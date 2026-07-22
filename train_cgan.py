"""
train_cgan.py
=============
Trains a Conditional GAN (cGAN) on logged gameplay runs (runs.csv) to
generate adaptive difficulty parameters for the roguelike.

Architecture
------------
  Condition vector  c  (6 floats) — the CombatMetrics feature vector:
      [melee_ratio, magic_ratio, hpk_melee, hpk_magic, dmg_ratio, death_rate]

  Generator  G(z, c)  → difficulty params p  (4 floats):
      [enemy_hp_mult, enemy_dmg_mult, enemy_speed_mult, spawn_count_norm]

  Discriminator  D(p, c) → real/fake logit

  LootGenerator  LG(z, archetype_soft_vec) → loot weights (4 floats)
      Conditioned on a 3-dim archetype soft-vector from the frozen classifier.

CUDA acceleration
-----------------
  * Default device is "auto": CUDA > MPS > CPU, selected automatically.
  * DataLoader uses pin_memory + persistent_workers when CUDA is available
    for faster host→GPU transfers.
  * torch.amp mixed-precision (autocast + GradScaler) on CUDA for ~2× speed
    with no loss in model quality.
  * torch.compile() (PyTorch ≥ 2.0) fuses GPU kernels; skipped gracefully on
    older versions or non-CUDA backends.
  * Cosine annealing LR scheduler on both G and D for better late convergence.

Usage
-----
  python train_cgan.py --csv runs.csv --out models/
  python train_cgan.py --csv runs.csv --out models/ --epochs 300 --batch 128
  python train_cgan.py --csv runs.csv --out models/ --device cuda   # explicit

  # "auto" is the default and picks the best available backend automatically.

Output (saved to --out directory)
------
  generator.pt          — trained generator weights
  loot_generator.pt     — trained loot generator weights
  classifier.pt         — trained archetype classifier weights
  scaler.json           — feature normalisation stats
  training_curve.png    — G/D loss plot (requires matplotlib)
  model_runtime.py      — drop-in runtime module for game.py / rogue.py

model_runtime.py interface
--------------------------
  class LoadedModels:
      @classmethod
      def load(cls, model_dir) -> LoadedModels: ...
      def generate(self, features: torch.Tensor) -> dict: ...
      def generate_loot_bias(self, features: torch.Tensor) -> list: ...
"""

import argparse
import json
import os
import sys
import math
import random
from typing import List, Tuple

import csv

# ---------------------------------------------------------------------------
# Optional imports — give clear errors if missing
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    print("ERROR: PyTorch is required.  pip install torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COND_DIM    = 6     # CombatMetrics feature vector length
NOISE_DIM   = 16    # latent noise fed to generator
PARAM_DIM   = 4     # [hp_mult, dmg_mult, speed_mult, spawn_count_norm]
ARCH_DIM    = 3     # archetype soft-vector dimension (Knight/Berserker/Sniper)
MIN_SAMPLES = 200   # warn below this threshold

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    """
    Resolve "auto" to the best available backend; pass explicit strings through.
    Prints a one-line summary of what was chosen and why.
    """
    if device_str != "auto":
        d = torch.device(device_str)
        print(f"Device: {d}  (explicit)")
        return d

    if torch.cuda.is_available():
        d = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device: {d}  ({name}, {mem:.1f} GB VRAM)  [auto-selected]")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        d = torch.device("mps")
        print(f"Device: {d}  (Apple Silicon)  [auto-selected]")
    else:
        d = torch.device("cpu")
        print(f"Device: {d}  (no GPU found)  [auto-selected]")

    return d

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def row_to_condition(row: dict) -> List[float]:
    """Reconstruct the 6-dim condition vector. Mirrors CombatMetrics.to_feature_vector()."""
    mk   = float(row.get("melee_kills",  0))
    mgk  = float(row.get("magic_kills",  0))
    mh   = float(row.get("melee_hits",   0))
    mgh  = float(row.get("magic_hits",   0))
    dt   = float(row.get("damage_taken", 0))
    dd   = float(row.get("damage_dealt", 1e-6))
    dths = float(row.get("deaths",       0))
    ta   = float(row.get("time_alive",   1e-6))

    total_k     = mk + mgk
    melee_ratio = (mk  / total_k) if total_k > 0 else 0.5
    magic_ratio = (mgk / total_k) if total_k > 0 else 0.5
    hpk_melee   = (mh  / mk)  if mk  > 0 else float(mh  + 1)
    hpk_magic   = (mgh / mgk) if mgk > 0 else float(mgh + 1)
    dmg_ratio   = dt / dd if dd > 1e-6 else 1.0
    death_rate  = (dths / max(ta, 1e-6)) * 60.0

    return [
        clamp(melee_ratio, 0.0, 1.0),
        clamp(magic_ratio, 0.0, 1.0),
        clamp(hpk_melee,   0.0, 10.0),
        clamp(hpk_magic,   0.0, 10.0),
        clamp(dmg_ratio,   0.0, 5.0),
        clamp(death_rate,  0.0, 5.0),
    ]


def row_to_params(row: dict) -> List[float]:
    """Extract 4-dim difficulty parameter vector."""
    hp    = float(row.get("enemy_hp_mult",    1.0))
    dmg   = float(row.get("enemy_dmg_mult",   1.0))
    spd   = float(row.get("enemy_speed_mult", 1.0))
    spawn = float(row.get("spawn_count",      4))
    spawn_norm = clamp((spawn - 2) / 8.0, 0.0, 1.0)
    return [
        clamp(hp,   0.1, 4.0),
        clamp(dmg,  0.1, 3.5),
        clamp(spd,  0.5, 2.5),
        spawn_norm,
    ]


def load_csv(path: str) -> Tuple[List[List[float]], List[List[float]]]:
    conditions, params = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                c = row_to_condition(row)
                p = row_to_params(row)
                # FIX: validate dimensions before appending so tensors stay rectangular
                if len(c) != COND_DIM or len(p) != PARAM_DIM:
                    continue
                conditions.append(c)
                params.append(p)
            except (ValueError, KeyError):
                continue
    return conditions, params

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_stats(data: List[List[float]]):
    import statistics
    cols  = list(zip(*data))
    means = [statistics.mean(c) for c in cols]
    stds  = [statistics.stdev(c) if len(c) > 1 else 1.0 for c in cols]
    stds  = [s if s > 1e-8 else 1.0 for s in stds]
    return means, stds


def normalise(data: List[List[float]], means, stds) -> List[List[float]]:
    return [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in data]

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """G(z, c) → p"""
    def __init__(self, noise_dim=NOISE_DIM, cond_dim=COND_DIM, param_dim=PARAM_DIM):
        super().__init__()
        inp = noise_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64,  128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64,  param_dim),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    """D(p, c) → real/fake logit"""
    def __init__(self, cond_dim=COND_DIM, param_dim=PARAM_DIM):
        super().__init__()
        inp = param_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 64),  nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64,  128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64,  1),
        )

    def forward(self, p, c):
        return self.net(torch.cat([p, c], dim=1))


class LootGenerator(nn.Module):
    """
    Produces a 4-dim loot weight vector [weapon_w, spell_w, armor_w, boots_w].
    Conditioned on the 3-dim archetype soft-vector from the frozen classifier.
    Sigmoid outputs; apply softmax at runtime for a categorical distribution.

    FIX: cond_dim is ARCH_DIM (3), not COND_DIM (6).  The training loop feeds
    the classifier's softmax output (shape (bs, 3)) — this must match the
    Linear input size, otherwise weights saved here won't load in model_runtime.
    """
    def __init__(self, noise_dim=8, cond_dim=ARCH_DIM, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),                   nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))

# ---------------------------------------------------------------------------
# Archetype classifier + loot targets
# ---------------------------------------------------------------------------

# Per-archetype loot weight targets [weapon, spell, armor, boots]
_LOOT_TARGETS = torch.tensor([
    [0.55, 0.05, 0.25, 0.15],   # 0 Knight
    [0.30, 0.05, 0.45, 0.20],   # 1 Berserker
    [0.05, 0.55, 0.20, 0.20],   # 2 Sniper
], dtype=torch.float32)


def derive_archetype_labels(C_norm: torch.Tensor, means, stds) -> torch.Tensor:
    """
    Heuristic pseudo-labels from normalised condition vectors.
    Denormalise the relevant columns first.
    Column order: [melee_ratio, magic_ratio, hpk_m, hpk_mg, dmg_ratio, death_rate]
    """
    m = torch.tensor(means, dtype=torch.float32)
    s = torch.tensor(stds,  dtype=torch.float32)
    raw = C_norm * s + m   # denormalise

    melee_ratio = raw[:, 0]
    magic_ratio = raw[:, 1]
    dmg_ratio   = raw[:, 4]
    death_rate  = raw[:, 5]

    labels = torch.zeros(len(C_norm), dtype=torch.long)
    labels[magic_ratio > 0.6] = 2
    labels[(melee_ratio > 0.6) & (dmg_ratio > 1.2) & (death_rate > 0.3)] = 1

    # FIX: clamp to valid index range so _LOOT_TARGETS[labels] never goes OOB
    labels = torch.clamp(labels, 0, _LOOT_TARGETS.shape[0] - 1)
    return labels


class ArchetypeClassifier(nn.Module):
    def __init__(self, cond_dim=COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),       nn.ReLU(),
            nn.Linear(16, ARCH_DIM),
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    conditions: List[List[float]],
    params:     List[List[float]],
    out_dir:    str,
    epochs:     int,
    batch_size: int,
    lr:         float,
    noise_dim:  int,
    device:     torch.device,
):
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = device.type == "cuda"
    use_amp  = use_cuda   # mixed precision only on CUDA

    # ------------------------------------------------------------------ data
    cond_means, cond_stds = compute_stats(conditions)
    par_means,  par_stds  = compute_stats(params)

    cond_norm = normalise(conditions, cond_means, cond_stds)
    par_norm  = normalise(params,     par_means,  par_stds)

    C = torch.tensor(cond_norm, dtype=torch.float32)
    P = torch.tensor(par_norm,  dtype=torch.float32)

    # FIX: validate tensor shapes before building dataset so any future
    # CSV changes surface a clear error rather than a cryptic AssertionError.
    assert C.shape[1] == COND_DIM,  f"Expected {COND_DIM} condition cols, got {C.shape[1]}"
    assert P.shape[1] == PARAM_DIM, f"Expected {PARAM_DIM} param cols,     got {P.shape[1]}"
    assert C.shape[0] == P.shape[0], \
        f"Row count mismatch: conditions={C.shape[0]}, params={P.shape[0]}"

    # Archetype labels + loot targets (CPU tensors; moved to device in loop)
    labels    = derive_archetype_labels(C, cond_means, cond_stds)
    loot_tgts = _LOOT_TARGETS[labels]   # (N, 4)

    # FIX: final sanity-check before TensorDataset — all four must agree on N
    N = C.shape[0]
    assert labels.shape[0]    == N, f"labels row count {labels.shape[0]} != {N}"
    assert loot_tgts.shape[0] == N, f"loot_tgts row count {loot_tgts.shape[0]} != {N}"
    print(f"Dataset shapes — C:{tuple(C.shape)}  P:{tuple(P.shape)}  "
          f"labels:{tuple(labels.shape)}  loot_tgts:{tuple(loot_tgts.shape)}")

    dataset    = TensorDataset(C, P, labels, loot_tgts)
    dataloader = DataLoader(
        dataset,
        batch_size   = batch_size,
        shuffle      = True,
        drop_last    = True,
        pin_memory   = use_cuda,
        num_workers  = min(4, os.cpu_count() or 1) if use_cuda else 0,
        persistent_workers = use_cuda,
    )

    # ------------------------------------------------------------------ models
    G   = Generator(noise_dim=noise_dim).to(device)
    D   = Discriminator().to(device)
    CLF = ArchetypeClassifier().to(device)
    LG  = LootGenerator(noise_dim=8).to(device)  # cond_dim=ARCH_DIM=3 by default

    # torch.compile (PyTorch ≥ 2.0) fuses kernels on CUDA for extra throughput.
    if use_cuda and hasattr(torch, "compile"):
        try:
            G   = torch.compile(G)
            D   = torch.compile(D)
            CLF = torch.compile(CLF)
            LG  = torch.compile(LG)
            print("torch.compile() enabled — GPU kernel fusion active")
        except Exception as e:
            print(f"torch.compile() skipped: {e}")

    opt_G   = optim.Adam(G.parameters(),   lr=lr,    betas=(0.5, 0.999))
    opt_D   = optim.Adam(D.parameters(),   lr=lr,    betas=(0.5, 0.999))
    opt_clf = optim.Adam(CLF.parameters(), lr=1e-3)
    opt_lg  = optim.Adam(LG.parameters(),  lr=1e-3)

    # Cosine annealing: smoothly decays LR to lr/100 over the full run
    sched_G  = optim.lr_scheduler.CosineAnnealingLR(opt_G,  T_max=epochs,             eta_min=lr / 100)
    sched_D  = optim.lr_scheduler.CosineAnnealingLR(opt_D,  T_max=epochs,             eta_min=lr / 100)
    sched_lg = optim.lr_scheduler.CosineAnnealingLR(opt_lg, T_max=max(40, epochs//2), eta_min=1e-5)

    criterion = nn.BCEWithLogitsLoss()
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    # Mixed-precision scaler (no-op when use_amp=False)
    scaler_G  = torch.amp.GradScaler(enabled=use_amp)
    scaler_D  = torch.amp.GradScaler(enabled=use_amp)
    scaler_lg = torch.amp.GradScaler(enabled=use_amp)

    g_losses, d_losses = [], []

    print(f"\nTraining cGAN — {len(conditions)} samples, {epochs} epochs, "
          f"batch {batch_size}, device {device}")
    print(f"  Mixed precision (AMP): {'on' if use_amp else 'off'}")
    print(f"  COND_DIM={COND_DIM}  NOISE={noise_dim}  PARAM_DIM={PARAM_DIM}  ARCH_DIM={ARCH_DIM}\n")

    # ------------------------------------------------------------------ phase 1: classifier
    print("Phase 1/3 — Archetype classifier…")
    clf_epochs = 25
    for ep in range(clf_epochs):
        CLF.train()
        for c_batch, _, lb, _ in dataloader:
            c_batch = c_batch.to(device, non_blocking=True)
            lb      = lb.to(device,      non_blocking=True)
            logits  = CLF(c_batch)
            loss    = ce_loss(logits, lb)
            opt_clf.zero_grad()
            loss.backward()
            opt_clf.step()
    CLF.eval()
    for p in CLF.parameters():
        p.requires_grad_(False)   # freeze; used only for conditioning
    print(f"  Classifier trained ({clf_epochs} epochs, frozen).")

    # ------------------------------------------------------------------ phase 2: cGAN
    print("\nPhase 2/3 — Difficulty cGAN…")
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        epoch_g = epoch_d = batches = 0.0

        for c_batch, p_real, _, _ in dataloader:
            c_batch = c_batch.to(device, non_blocking=True)
            p_real  = p_real.to(device,  non_blocking=True)

            bs          = c_batch.size(0)
            real_labels = torch.ones( bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # ---- Discriminator ----
            opt_D.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                d_real      = D(p_real, c_batch)
                loss_d_real = criterion(d_real, real_labels)
                z           = torch.randn(bs, noise_dim, device=device)
                p_fake      = G(z, c_batch).detach()
                d_fake      = D(p_fake, c_batch)
                loss_d_fake = criterion(d_fake, fake_labels)
                loss_d      = (loss_d_real + loss_d_fake) * 0.5
            scaler_D.scale(loss_d).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # ---- Generator ----
            opt_G.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z      = torch.randn(bs, noise_dim, device=device)
                p_fake = G(z, c_batch)
                d_out  = D(p_fake, c_batch)
                loss_g = criterion(d_out, real_labels)
            scaler_G.scale(loss_g).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            batches += 1

        sched_G.step()
        sched_D.step()

        avg_g = epoch_g / max(batches, 1)
        avg_d = epoch_d / max(batches, 1)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            lr_now = sched_G.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{epochs}  G={avg_g:.4f}  D={avg_d:.4f}  lr={lr_now:.2e}")

    # ------------------------------------------------------------------ phase 3: LootGenerator
    loot_epochs = max(40, epochs // 2)
    print(f"\nPhase 3/3 — Loot generator ({loot_epochs} epochs)…")
    LG.train()
    for ep in range(1, loot_epochs + 1):
        ep_loss = 0.0
        for c_batch, _, _, loot_batch in dataloader:
            c_batch    = c_batch.to(device,    non_blocking=True)
            loot_batch = loot_batch.to(device, non_blocking=True)

            # FIX: use torch.no_grad() since CLF is frozen — avoids stale graph issues
            with torch.no_grad():
                # c_soft is (bs, ARCH_DIM=3); this is what LootGenerator expects
                c_soft = torch.softmax(CLF(c_batch), dim=1)

            opt_lg.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                bs   = c_batch.size(0)
                z    = torch.randn(bs, 8, device=device)
                pred = LG(z, c_soft)          # (bs, 4)
                loss = mse_loss(pred, loot_batch)
            scaler_lg.scale(loss).backward()
            scaler_lg.step(opt_lg)
            scaler_lg.update()
            ep_loss = loss.item()

        sched_lg.step()
        if ep % 10 == 0 or ep == loot_epochs:
            print(f"  Epoch {ep:3d}/{loot_epochs}  MSE={ep_loss:.5f}  "
                  f"lr={sched_lg.get_last_lr()[0]:.2e}")

    # ------------------------------------------------------------------ save artefacts
    gen_path    = os.path.join(out_dir, "generator.pt")
    loot_path   = os.path.join(out_dir, "loot_generator.pt")
    clf_path    = os.path.join(out_dir, "classifier.pt")   # FIX: was never saved
    scaler_path = os.path.join(out_dir, "scaler.json")

    # Unwrap compiled models before saving state dicts
    def _state(m):
        return getattr(m, "_orig_mod", m).state_dict()

    torch.save(_state(G),   gen_path)
    torch.save(_state(LG),  loot_path)
    torch.save(_state(CLF), clf_path)   # FIX: save classifier so runtime can load it
    print(f"\nGenerator saved      → {gen_path}")
    print(f"LootGenerator saved  → {loot_path}")
    print(f"Classifier saved     → {clf_path}")

    scaler = {
        "cond_means": cond_means, "cond_stds": cond_stds,
        "par_means":  par_means,  "par_stds":  par_stds,
        "noise_dim":  noise_dim,
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"Scaler saved         → {scaler_path}")

    # Loss plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(g_losses, label="G loss")
        ax.plot(d_losses, label="D loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
        ax.set_title("cGAN training curve"); ax.legend()
        plot_path = os.path.join(out_dir, "training_curve.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"Loss plot            → {plot_path}")
    except ImportError:
        print("(matplotlib not installed — loss plot skipped)")

    write_model_runtime(out_dir, noise_dim)
    print(f"Runtime module       → {os.path.join(out_dir, 'model_runtime.py')}")
    print("\nTraining complete.")

# ---------------------------------------------------------------------------
# model_runtime.py writer
# Generates the drop-in module that rogue.py imports via LoadedModels.
# ---------------------------------------------------------------------------

MODEL_RUNTIME_TEMPLATE = '''"""
model_runtime.py  —  auto-generated by train_cgan.py
Exposes LoadedModels so rogue.py can call it without changes.
"""
import json
import os
import torch
import torch.nn as nn


COND_DIM   = 6
PARAM_DIM  = 4
ARCH_DIM   = 3   # archetype soft-vector: [Knight, Berserker, Sniper]
LOOT_DIM   = 4   # [weapon_w, spell_w, armor_w, boots_w]


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim=COND_DIM, param_dim=PARAM_DIM):
        super().__init__()
        inp = noise_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64,  128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64,  param_dim),
        )
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class LootGenerator(nn.Module):
    # FIX: cond_dim=ARCH_DIM (3) to match training — was incorrectly 6 in the
    # original template, causing a weight shape mismatch on load.
    def __init__(self, noise_dim=8, cond_dim=ARCH_DIM, out_dim=LOOT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),                   nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class ArchetypeClassifier(nn.Module):
    def __init__(self, cond_dim=COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),       nn.ReLU(),
            nn.Linear(16, ARCH_DIM),
        )
    def forward(self, x):
        return self.net(x)


class LoadedModels:
    def __init__(self, generator, loot_generator, classifier, scaler, device):
        self.generator      = generator
        self.loot_generator = loot_generator
        self.classifier     = classifier
        self.scaler         = scaler
        self.device         = device

    @classmethod
    def load(cls, model_dir: str) -> "LoadedModels":
        device      = _best_device()
        scaler_path = os.path.join(model_dir, "scaler.json")
        gen_path    = os.path.join(model_dir, "generator.pt")
        loot_path   = os.path.join(model_dir, "loot_generator.pt")
        clf_path    = os.path.join(model_dir, "classifier.pt")

        with open(scaler_path) as f:
            scaler = json.load(f)

        noise_dim = scaler["noise_dim"]

        gen = Generator(noise_dim=noise_dim)
        gen.load_state_dict(torch.load(gen_path, map_location="cpu"))
        gen.to(device).eval()

        loot_gen = LootGenerator()
        if os.path.exists(loot_path):
            loot_gen.load_state_dict(torch.load(loot_path, map_location="cpu"))
        loot_gen.to(device).eval()

        clf = ArchetypeClassifier()
        if os.path.exists(clf_path):
            clf.load_state_dict(torch.load(clf_path, map_location="cpu"))
        clf.to(device).eval()

        return cls(gen, loot_gen, clf, scaler, device)

    def _normalise_features(self, features: torch.Tensor) -> torch.Tensor:
        """Z-score normalise a (1, 6) feature tensor using saved scaler stats."""
        sc = self.scaler
        cm = torch.tensor(sc["cond_means"], dtype=torch.float32, device=self.device)
        cs = torch.tensor(sc["cond_stds"],  dtype=torch.float32, device=self.device)
        return (features.to(self.device) - cm) / cs

    def generate(self, features: torch.Tensor) -> dict:
        """
        features: (1, 6) CombatMetrics feature tensor
        Returns difficulty param dict with keys:
          enemy_hp_mult, enemy_dmg_mult, enemy_speed_mult, spawn_count
        """
        c  = self._normalise_features(features)
        sc = self.scaler
        noise_dim = sc["noise_dim"]
        z  = torch.randn(1, noise_dim, device=self.device)

        with torch.no_grad():
            p_norm = self.generator(z, c)

        pm = torch.tensor(sc["par_means"], dtype=torch.float32, device=self.device)
        ps = torch.tensor(sc["par_stds"],  dtype=torch.float32, device=self.device)
        p  = (p_norm * ps + pm).squeeze(0).tolist()

        hp_mult    = max(0.45, min(3.5,  p[0]))
        dmg_mult   = max(0.35, min(2.8,  p[1]))
        speed_mult = max(0.70, min(2.0,  p[2]))
        spawn_norm = max(0.0,  min(1.0,  p[3]))
        spawn_n    = max(2, min(10, int(round(2 + spawn_norm * 8))))

        return {
            "enemy_hp_mult":    round(hp_mult,    3),
            "enemy_dmg_mult":   round(dmg_mult,   3),
            "enemy_speed_mult": round(speed_mult, 3),
            "spawn_count":      spawn_n,
        }

    def generate_loot_bias(self, features: torch.Tensor) -> list:
        """
        features: (1, 6) CombatMetrics feature tensor
        Returns a 4-element list [weapon_w, spell_w, armor_w, boots_w] in (0,1).
        Pass through softmax in rogue.py to get a proper categorical distribution.
        """
        c_norm = self._normalise_features(features)

        with torch.no_grad():
            # Classifier produces (1, ARCH_DIM=3) soft-vector
            c_soft  = torch.softmax(self.classifier(c_norm), dim=1)
            z       = torch.randn(1, 8, device=self.device)
            weights = self.loot_generator(z, c_soft)   # (1, 4)

        return weights.squeeze(0).tolist()
'''


def write_model_runtime(out_dir: str, noise_dim: int):
    path = os.path.join(out_dir, "model_runtime.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(MODEL_RUNTIME_TEMPLATE)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train cGAN for roguelike DDA")
    parser.add_argument("--csv",    default="runs.csv",
                        help="Path to runs.csv")
    parser.add_argument("--out",    default="models",
                        help="Directory to save model artefacts")
    parser.add_argument("--epochs", type=int,   default=150,
                        help="cGAN training epochs")
    parser.add_argument("--batch",  type=int,   default=32,
                        help="Batch size")
    parser.add_argument("--lr",     type=float, default=2e-4,
                        help="Learning rate for G and D")
    parser.add_argument("--noise",  type=int,   default=NOISE_DIM,
                        help="Noise vector dimension")
    parser.add_argument("--device", default="auto",
                        help="torch device: auto | cpu | cuda | mps  (default: auto)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    conditions, params = load_csv(args.csv)

    if len(conditions) == 0:
        print("ERROR: No valid rows found in CSV.")
        sys.exit(1)

    if len(conditions) < MIN_SAMPLES:
        print(f"WARNING: Only {len(conditions)} samples (recommended ≥ {MIN_SAMPLES}).")
        print("         Model quality may be poor. Run more bot/human sessions first.\n")
    else:
        print(f"Loaded {len(conditions)} samples from {args.csv}")

    # FIX: validate that every row has the expected number of columns before
    # handing off to train() — catches CSV schema drift early with a clear message.
    bad_c = [i for i, c in enumerate(conditions) if len(c) != COND_DIM]
    bad_p = [i for i, p in enumerate(params)     if len(p) != PARAM_DIM]
    if bad_c:
        print(f"ERROR: {len(bad_c)} condition rows have wrong length (expected {COND_DIM}).")
        sys.exit(1)
    if bad_p:
        print(f"ERROR: {len(bad_p)} param rows have wrong length (expected {PARAM_DIM}).")
        sys.exit(1)

    device = resolve_device(args.device)

    train(
        conditions = conditions,
        params     = params,
        out_dir    = args.out,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        noise_dim  = args.noise,
        device     = device,
    )


if __name__ == "__main__":
    main()