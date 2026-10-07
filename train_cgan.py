from dataclasses import dataclass
from typing import List, Tuple
import argparse
import csv
import json
import os
import random

import dda_core as core
import telemetry

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = object

COND_DIM = core.COND_DIM      # 9
TARGET_DIM = telemetry.TARGET_DIM  # 4: hp_mult, dmg_mult, speed_mult, spawn_count

COND_COLUMNS = [
    "cond_melee_ratio", "cond_magic_ratio", "cond_hpk_melee", "cond_hpk_magic",
    "cond_dmg_ratio", "cond_hp_ratio", "cond_total_kills_norm",
    "cond_node_depth_norm", "cond_progress_norm",
]
TARGET_COLUMNS = ["applied_hp_mult", "applied_dmg_mult", "applied_speed_mult",
                   "applied_spawn_count"]

CHECKPOINT_SCHEMA_VERSION = "cgan_v2_checkpoint_1"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_rows(csv_path: str, manifest_path: str,
                       include_boss: bool = False, boss_only: bool = False) -> List[dict]:
    """
    Loads and filters rows. Raises telemetry.DatasetVersionError if the
    dataset doesn't validate against the current code (finding #9.1/#9.8).
    """
    telemetry.validate_dataset(csv_path, manifest_path)

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if boss_only:
                if r["room_kind"] != "boss":
                    continue
            else:
                if r["room_kind"] not in ("enemy", "boss"):
                    continue
                if r["room_kind"] == "boss" and not include_boss:
                    continue
            rows.append(r)
    return rows


def group_split_by_run_id(rows: List[dict], val_frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Group-aware split: all rows for a given run_id go entirely to train
    OR entirely to val, never split across both (prevents leakage)."""
    run_ids = sorted(set(r["run_id"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(run_ids)
    n_val_runs = max(1, int(len(run_ids) * val_frac))
    val_run_ids = set(run_ids[:n_val_runs])
    train_rows = [r for r in rows if r["run_id"] not in val_run_ids]
    val_rows = [r for r in rows if r["run_id"] in val_run_ids]
    return train_rows, val_rows


def rows_to_arrays(rows: List[dict]):
    cond = [[float(r[c]) for c in COND_COLUMNS] for r in rows]
    target = [[float(r[c]) for c in TARGET_COLUMNS] for r in rows]
    return cond, target


@dataclass
class Scaler:
    target_mean: List[float]
    target_std: List[float]

    @staticmethod
    def fit(targets: List[List[float]]) -> "Scaler":
        n = len(targets)
        dim = len(targets[0])
        mean = [sum(t[i] for t in targets) / n for i in range(dim)]
        var = [sum((t[i] - mean[i]) ** 2 for t in targets) / n for i in range(dim)]
        std = [max(v ** 0.5, 1e-6) for v in var]
        return Scaler(target_mean=mean, target_std=std)

    def transform(self, targets: List[List[float]]) -> List[List[float]]:
        return [[(t[i] - self.target_mean[i]) / self.target_std[i]
                  for i in range(len(t))] for t in targets]

    def inverse(self, scaled: List[float]) -> List[float]:
        return [scaled[i] * self.target_std[i] + self.target_mean[i]
                for i in range(len(scaled))]

    def to_json(self) -> dict:
        return {"target_mean": self.target_mean, "target_std": self.target_std,
                "target_columns": TARGET_COLUMNS, "cond_columns": COND_COLUMNS}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if torch is not None:
    class Generator(nn.Module):
        def __init__(self, cond_dim=COND_DIM, noise_dim=8, target_dim=TARGET_DIM, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cond_dim + noise_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, target_dim),
            )
            self.noise_dim = noise_dim

        def forward(self, cond, noise=None):
            if noise is None:
                noise = torch.randn(cond.shape[0], self.noise_dim, device=cond.device)
            x = torch.cat([cond, noise], dim=1)
            return self.net(x)

    class Discriminator(nn.Module):
        def __init__(self, cond_dim=COND_DIM, target_dim=TARGET_DIM, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cond_dim + target_dim, hidden), nn.LeakyReLU(0.2),
                nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1), nn.Sigmoid(),
            )

        def forward(self, cond, target):
            x = torch.cat([cond, target], dim=1)
            return self.net(x)
else:
    Generator = None
    Discriminator = None


def train(
    csv_path: str, manifest_path: str, out_dir: str,
    epochs: int = 30, batch_size: int = 32, val_frac: float = 0.2, seed: int = 0,
    include_boss: bool = False,
):
    if torch is None:
        raise RuntimeError("PyTorch is required to train (pip install torch).")

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    rows = load_dataset_rows(csv_path, manifest_path, include_boss=include_boss)
    if len(rows) < 20:
        raise RuntimeError(f"Only {len(rows)} combat rows after filtering -- too few to train on.")

    train_rows, val_rows = group_split_by_run_id(rows, val_frac=val_frac, seed=seed)
    if not train_rows or not val_rows:
        raise RuntimeError("Group split by run_id produced an empty train or val set -- need more runs.")

    train_cond, train_target = rows_to_arrays(train_rows)
    val_cond, val_target = rows_to_arrays(val_rows)

    scaler = Scaler.fit(train_target)
    train_target_s = scaler.transform(train_target)
    val_target_s = scaler.transform(val_target)

    train_cond_t = torch.tensor(train_cond, dtype=torch.float32)
    train_target_t = torch.tensor(train_target_s, dtype=torch.float32)
    val_cond_t = torch.tensor(val_cond, dtype=torch.float32)
    val_target_t = torch.tensor(val_target_s, dtype=torch.float32)

    gen = Generator()
    disc = Discriminator()
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-3, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    n = train_cond_t.shape[0]
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            cond_b = train_cond_t[idx]
            real_b = train_target_t[idx]
            bs = cond_b.shape[0]
            if bs < 2:
                continue

            real_labels = torch.ones(bs, 1)
            fake_labels = torch.zeros(bs, 1)

            fake_b = gen(cond_b).detach()
            d_real = disc(cond_b, real_b)
            d_fake = disc(cond_b, fake_b)
            d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            fake_b2 = gen(cond_b)
            d_fake2 = disc(cond_b, fake_b2)
            recon_loss = nn.functional.mse_loss(fake_b2, real_b)
            g_loss = bce(d_fake2, real_labels) + 0.5 * recon_loss
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            epoch_g_loss += float(g_loss.detach()); epoch_d_loss += float(d_loss.detach())
            n_batches += 1

        with torch.no_grad():
            val_pred = gen(val_cond_t)
            val_mse = float(nn.functional.mse_loss(val_pred, val_target_t))

        history.append({"epoch": epoch, "g_loss": epoch_g_loss / max(n_batches, 1),
                         "d_loss": epoch_d_loss / max(n_batches, 1), "val_mse": val_mse})

    manifest = telemetry.load_manifest(manifest_path)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": manifest["schema_version"],
        "dda_version": manifest["dda_version"],
        "config_hash": manifest["config_hash"],
        "cond_dim": COND_DIM,
        "target_dim": TARGET_DIM,
        "cond_columns": COND_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
    }
    ckpt_path = os.path.join(out_dir, "cgan_checkpoint.pt")
    torch.save(checkpoint, ckpt_path)

    scaler_path = os.path.join(out_dir, "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler.to_json(), f, indent=2)

    history_path = os.path.join(out_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")
    print(f"final val_mse={history[-1]['val_mse']:.4f}")
    print(f"checkpoint -> {ckpt_path}")
    print(f"scaler -> {scaler_path}")
    return ckpt_path, scaler_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="room_samples_v2.csv")
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="model_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-boss", action="store_true")
    args = ap.parse_args()

    manifest_path = args.manifest or (os.path.splitext(args.csv)[0] + ".manifest.json")
    train(args.csv, manifest_path, args.out_dir, epochs=args.epochs,
          batch_size=args.batch_size, val_frac=args.val_frac, seed=args.seed,
          include_boss=args.include_boss)


if __name__ == "__main__":
    main()
