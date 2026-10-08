"""
models.py
=========
Torch modules and losses for the outcome-guided models.

Enemy intervention:
  Q(x, a) -> predicted outcome vector   (OutcomeModel, use_action=True)
  Q(x)    -> state-only ablation        (OutcomeModel, use_action=False)
  G(x, z) -> candidate action           (Generator; inside the observed
                                          action-support box by construction)
  D(x, a) -> plausibility logit         (Discriminator)
  S       -> kNN action-support model   (SupportModel; OOD penalty + runtime guard)

Loot category:
  Q_loot(features) -> next-3-combat-room outcome profile (BoundedOutcomeNet)
  G_loot(x, e, z)  -> [P_weapon, P_spell, P_armor, P_boots] (LootGenerator)

Conventions
-----------
* x is the condition vector standardized with the TRAIN-split scaler.
* Enemy actions inside the networks are normalized to [0, 1]^4 over
  dda_core.ACTION_BOUNDS (ml_data.normalize_action).
* Every outcome network emits [binary_logit, continuous...] where each
  continuous output is BOUNDED BY ARCHITECTURE to its physical range
  (lo + (hi - lo) * sigmoid(.)), so no extrapolation can leave [lo, hi].
  `outcomes_physical()` turns the logit into a probability.

Enemy generator objective (generator_loss) -- no observed/heuristic action
enters it, so there is no reconstruction term:

  L_G = L_adv + lambda_balance * L_balance(Q(x, G))
        + lambda_diversity * L_diversity + lambda_support * L_support
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import balance
import dda_core as core
import ml_data

COND_DIM = core.COND_DIM
ACTION_DIM = core.ACTION_DIM
OUTCOME_DIM = balance.OUTCOME_DIM
N_CONT = OUTCOME_DIM - 1      # continuous outcomes (all but death)


def mlp(in_dim: int, hidden: int, layers: int, out_dim: int, act=nn.ReLU,
        dropout: float = 0.0) -> nn.Sequential:
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# Bounded outcome networks
# ---------------------------------------------------------------------------

class BoundedOutcomeNet(nn.Module):
    """Generic outcome net: output 0 is a binary logit; outputs 1.. are
    continuous outcomes squashed into their physical (lo, hi) ranges."""

    def __init__(self, in_dim: int, cont_ranges: Sequence[Sequence[float]], hidden=128, layers=3,
                 dropout=0.05):
        super().__init__()
        self.in_dim = in_dim
        self.cont_ranges = [list(map(float, r)) for r in cont_ranges]
        self.net = mlp(in_dim, hidden, layers, 1 + len(self.cont_ranges), act=nn.SiLU, dropout=dropout)
        lo = torch.tensor([r[0] for r in self.cont_ranges], dtype=torch.float32)
        hi = torch.tensor([r[1] for r in self.cont_ranges], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("span", hi - lo)
        self.hparams = {"in_dim": in_dim, "cont_ranges": self.cont_ranges, "hidden": hidden,
                        "layers": layers, "dropout": dropout}

    def head(self, raw):
        return torch.cat([raw[:, :1], self.lo + self.span * torch.sigmoid(raw[:, 1:])], dim=1)

    def forward_features(self, inp):
        return self.head(self.net(inp))

    def forward(self, inp):
        return self.forward_features(inp)


class OutcomeModel(BoundedOutcomeNet):
    def __init__(self, cond_dim=COND_DIM, action_dim=ACTION_DIM, hidden=128, layers=3,
                 dropout=0.05, use_action=True, cont_ranges=None):
        cont_ranges = cont_ranges or [list(r) for r in balance.output_ranges("enemy")[1:]]
        super().__init__(cond_dim + (action_dim if use_action else 0), cont_ranges, hidden, layers, dropout)
        self.use_action = use_action
        self.hparams = {"cond_dim": cond_dim, "action_dim": action_dim, "hidden": hidden,
                        "layers": layers, "dropout": dropout, "use_action": use_action,
                        "cont_ranges": self.cont_ranges}

    def forward(self, x_s, a_n=None):
        inp = torch.cat([x_s, a_n], dim=1) if self.use_action else x_s
        return self.forward_features(inp)


def outcome_loss(pred, y_bin, y_cont, cont_var, bin_weight: float = 1.0):
    """BCE on the binary outcome + variance-normalized MSE on the bounded
    continuous outcomes (physical units; cont_var from the TRAIN split)."""
    bce = F.binary_cross_entropy_with_logits(pred[:, 0], y_bin)
    mse = (((pred[:, 1:] - y_cont) ** 2) / cont_var).mean()
    return bin_weight * bce + mse, bce, mse


q_loss = outcome_loss


def outcomes_physical(pred):
    """[logit, continuous...] -> [probability, continuous...] (already bounded)."""
    return torch.cat([torch.sigmoid(pred[:, :1]), pred[:, 1:]], dim=1)


# ---------------------------------------------------------------------------
# Action support (OOD protection)
# ---------------------------------------------------------------------------

class SupportModel:
    """
    kNN support of the observed (x, a) training pairs in a standardized joint
    space (x already standardized; a_n standardized with train mean/std).
    distance(x, a) = mean Euclidean distance to the k nearest reference pairs.
    threshold = `quantile` of the reference set's own leave-one-out distances,
    so ~ (1 - quantile) of OBSERVED training pairs lie above it.
    """

    def __init__(self, ref, a_mean, a_std, k: int, threshold: float):
        self.ref = ref
        self.a_mean = a_mean
        self.a_std = a_std
        self.k = int(k)
        self.threshold = float(threshold)

    @staticmethod
    def fit(x_s, a_n, k: int, quantile: float, max_rows: int, seed: int = 0) -> "SupportModel":
        g = torch.Generator().manual_seed(seed)
        n = x_s.shape[0]
        idx = torch.randperm(n, generator=g)[:max_rows] if n > max_rows else torch.arange(n)
        x_s, a_n = x_s.detach().cpu(), a_n.detach().cpu()
        a_mean = a_n.mean(dim=0)
        a_std = a_n.std(dim=0).clamp_min(1e-3)
        ref = torch.cat([x_s[idx], (a_n[idx] - a_mean) / a_std], dim=1)
        d = torch.cdist(ref, ref)
        d.fill_diagonal_(float("inf"))
        loo = d.topk(k, largest=False).values.mean(dim=1)
        thr = float(torch.quantile(loo, quantile))
        return SupportModel(ref, a_mean, a_std, k, thr)

    def distance(self, x_s, a_n):
        ref = self.ref.to(x_s.device)
        q = torch.cat([x_s, (a_n - self.a_mean.to(x_s.device)) / self.a_std.to(x_s.device)], dim=1)
        return torch.cdist(q, ref).topk(self.k, largest=False).values.mean(dim=1)

    def penalty(self, x_s, a_n):
        """Smooth, zero inside the support threshold: mean(relu(d/thr - 1)^2)."""
        return (F.relu(self.distance(x_s, a_n) / self.threshold - 1.0) ** 2).mean()

    def to_state(self) -> dict:
        return {"ref": self.ref, "a_mean": self.a_mean, "a_std": self.a_std, "k": self.k,
                "threshold": self.threshold}

    @staticmethod
    def from_state(d: dict) -> "SupportModel":
        return SupportModel(d["ref"], d["a_mean"], d["a_std"], d["k"], d["threshold"])


def support_box(a_n, q_low: float, q_high: float):
    """Per-dimension [q_low, q_high] quantiles of observed normalized actions."""
    return torch.quantile(a_n, q_low, dim=0), torch.quantile(a_n, q_high, dim=0)


# ---------------------------------------------------------------------------
# Enemy generator / discriminator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """G(x, z) -> a_n inside [box_lo, box_hi] (the observed action-support box,
    a subset of the design bounds [0, 1]^4) by construction."""

    def __init__(self, cond_dim=COND_DIM, noise_dim=8, action_dim=ACTION_DIM, hidden=128, layers=3,
                 box_lo: Optional[Sequence[float]] = None, box_hi: Optional[Sequence[float]] = None):
        super().__init__()
        self.noise_dim = noise_dim
        box_lo = list(map(float, box_lo)) if box_lo is not None else [0.0] * action_dim
        box_hi = list(map(float, box_hi)) if box_hi is not None else [1.0] * action_dim
        self.hparams = {"cond_dim": cond_dim, "noise_dim": noise_dim, "action_dim": action_dim,
                        "hidden": hidden, "layers": layers, "box_lo": box_lo, "box_hi": box_hi}
        self.net = mlp(cond_dim + noise_dim, hidden, layers, action_dim, act=nn.ReLU)
        self.register_buffer("box_lo", torch.tensor(box_lo, dtype=torch.float32))
        self.register_buffer("box_span", torch.tensor(box_hi, dtype=torch.float32) - self.box_lo)

    def forward(self, x_s, z=None):
        if z is None:
            z = torch.randn(x_s.shape[0], self.noise_dim, device=x_s.device)
        return self.box_lo + self.box_span * torch.sigmoid(self.net(torch.cat([x_s, z], dim=1)))


class Discriminator(nn.Module):
    def __init__(self, cond_dim=COND_DIM, action_dim=ACTION_DIM, hidden=128, layers=2):
        super().__init__()
        self.hparams = {"cond_dim": cond_dim, "action_dim": action_dim, "hidden": hidden,
                        "layers": layers}
        self.net = mlp(cond_dim + action_dim, hidden, layers, 1, act=lambda: nn.LeakyReLU(0.2))

    def forward(self, x_s, a_n):
        return self.net(torch.cat([x_s, a_n], dim=1))   # logit


def discriminator_loss(disc, x_s, a_real_n, a_fake_n):
    real = disc(x_s, a_real_n)
    fake = disc(x_s, a_fake_n.detach())
    return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
            + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))


def diversity_loss(a1_n, a2_n, z1, z2, tau: float):
    """Mode-seeking hinge: penalizes near-identical outputs from different
    noise vectors for the same condition. Zero once the output spread per
    unit of noise spread exceeds tau, so it never pushes to extremes."""
    ratio = (a1_n - a2_n).abs().mean(dim=1) / ((z1 - z2).abs().mean(dim=1) + 1e-6)
    return F.relu(tau - ratio).mean()


GENERATOR_LOSS_TERMS = ("adv", "balance", "diversity", "support")


def generator_loss(gen: Generator, disc: Discriminator, q: OutcomeModel, support: SupportModel,
                   x_s, lambda_balance: float, lambda_diversity: float, diversity_tau: float,
                   lambda_support: float, exp_cfg: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    """
    L_G = L_adv + lambda_balance * L_balance + lambda_diversity * L_diversity
          + lambda_support * L_support

    Inputs are only the conditions x: observed/heuristic actions are never
    seen here, so G cannot be trained to reconstruct them. Q must be frozen
    by the caller; gradients flow through Q's input into G only.
    """
    n = x_s.shape[0]
    z1 = torch.randn(n, gen.noise_dim, device=x_s.device)
    z2 = torch.randn(n, gen.noise_dim, device=x_s.device)
    a1 = gen(x_s, z1)
    a2 = gen(x_s, z2)

    d_fake = disc(x_s, a1)
    l_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    l_bal = balance.balance_loss_torch(outcomes_physical(q(x_s, a1)), exp_cfg)
    l_div = diversity_loss(a1, a2, z1, z2, diversity_tau)
    l_sup = support.penalty(x_s, a1)

    total = l_adv + lambda_balance * l_bal + lambda_diversity * l_div + lambda_support * l_sup
    return {"total": total, "adv": l_adv, "balance": l_bal, "diversity": l_div, "support": l_sup}


def decode_actions(a_n) -> torch.Tensor:
    lo = torch.tensor(ml_data.ACTION_LO, dtype=a_n.dtype, device=a_n.device)
    hi = torch.tensor(ml_data.ACTION_HI, dtype=a_n.dtype, device=a_n.device)
    return lo + a_n * (hi - lo)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


# ---------------------------------------------------------------------------
# Enemy generator diagnostics (validation-time selection and evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generator_diagnostics(gen: Generator, q: OutcomeModel, support: SupportModel, x_s,
                          run_index: Optional[List[str]] = None,
                          reference_actions_n: Optional[Dict[str, torch.Tensor]] = None,
                          k: int = 8, seed: int = 0, exp_cfg: Optional[dict] = None) -> dict:
    """
    Metrics on a fixed set of conditions x_s (N, COND_DIM), in normalized
    action units ([0,1] per dimension):

      validity           share of generated actions that are finite and in
                         bounds after integer spawn rounding
      saturation         share of action components within 1% of a design bound
      balance_pred       mean Q-predicted balance loss of G's actions
      balance_ref        same for each reference action set on the SAME x
      ood_rate           share of G's actions whose support distance exceeds
                         the training threshold (ood_rate_ref for references)
      support_ratio      mean support distance / threshold of G's actions
      diversity          mean |a(z_i) - a(z_j)| across k noise draws per x
      sensitivity        mean |a(x + 1 std e_j, z) - a(x, z)| over features j
      jitter             mean |a_t - a_{t-1}| between consecutive rooms of a run
                         with the noise held fixed per run (state-driven change)
    """
    gen.eval()
    q.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x_s.shape[0]
    dev = x_s.device

    def z(m):
        return torch.randn(m, gen.noise_dim, generator=g).to(dev)

    samples = torch.stack([gen(x_s, z(n)) for _ in range(k)])           # (k, N, A)
    raw = decode_actions(samples)
    lo = torch.tensor(ml_data.ACTION_LO, device=dev)
    hi = torch.tensor(ml_data.ACTION_HI, device=dev)
    rounded = raw.clone()
    rounded[..., -1] = torch.round(rounded[..., -1])
    finite = torch.isfinite(raw).all(dim=-1)
    inb = ((rounded >= lo - 1e-6) & (rounded <= hi + 1e-6)).all(dim=-1)
    validity = float((finite & inb).float().mean())
    saturation = float(((samples < 0.01) | (samples > 0.99)).float().mean())

    bal = [float(balance.balance_loss_torch(outcomes_physical(q(x_s, samples[i])), exp_cfg))
           for i in range(k)]
    dist = torch.stack([support.distance(x_s, samples[i]) for i in range(k)])
    balance_ref, ood_ref = {}, {}
    for name, a_ref in (reference_actions_n or {}).items():
        balance_ref[name] = float(balance.balance_loss_torch(outcomes_physical(q(x_s, a_ref)), exp_cfg))
        ood_ref[name] = float((support.distance(x_s, a_ref) > support.threshold).float().mean())

    div = [float((samples[i] - samples[j]).abs().mean()) for i in range(k) for j in range(i + 1, k)]
    z0 = z(n)
    base = gen(x_s, z0)
    per_feat = []
    for j in range(x_s.shape[1]):
        xp = x_s.clone()
        xp[:, j] += 1.0
        per_feat.append(float((gen(xp, z0) - base).abs().mean()))

    jitter = float("nan")
    if run_index is not None and n > 1:
        # One noise vector per run, so jitter measures action changes driven
        # by STATE changes between consecutive rooms, not noise diversity.
        runs = {r: i for i, r in enumerate(dict.fromkeys(run_index))}
        z_run = z(len(runs))
        a_rt = gen(x_s, z_run[torch.tensor([runs[r] for r in run_index], device=dev)])
        steps = [float((a_rt[i] - a_rt[i - 1]).abs().mean())
                 for i in range(1, n) if run_index[i] == run_index[i - 1]]
        jitter = sum(steps) / len(steps) if steps else float("nan")

    mean_action = decode_actions(samples.mean(dim=(0, 1)).unsqueeze(0))[0].tolist()
    return {"n_conditions": n, "k": k, "validity": validity, "saturation": saturation,
            "balance_pred": sum(bal) / k, "balance_ref": balance_ref,
            "ood_rate": float((dist > support.threshold).float().mean()),
            "ood_rate_ref": ood_ref,
            "support_ratio": float((dist / support.threshold).mean()),
            "support_threshold": support.threshold,
            "diversity": sum(div) / max(len(div), 1),
            "sensitivity": sum(per_feat) / len(per_feat),
            "sensitivity_per_feature": dict(zip(ml_data.COND_COLUMNS, per_feat)),
            "jitter": jitter, "mean_action": dict(zip(core.ACTION_KEYS, mean_action))}


def selection_eligible(diag: dict, epoch: int, sel: dict) -> bool:
    """A checkpoint can only be selected after min_epoch and with non-trivial
    diversity and conditional sensitivity (rejects near-constant generators,
    e.g. an untrained network that happens to sit in a good region)."""
    return (epoch >= sel.get("min_epoch", 0) and diag["validity"] == 1.0
            and diag["diversity"] >= sel.get("min_diversity", 0.0)
            and diag["sensitivity"] >= sel.get("min_sensitivity", 0.0))


def selection_score(diag: dict, sel: dict) -> float:
    """Composite validation score (LOWER is better). Never the GAN loss."""
    import math
    jit = diag["jitter"] if math.isfinite(diag["jitter"]) else 0.0
    return (diag["balance_pred"]
            + sel["w_invalid"] * (1.0 - diag["validity"])
            + sel.get("w_ood", 0.0) * diag["ood_rate"]
            - sel["w_diversity"] * min(diag["diversity"], sel["diversity_cap"])
            - sel["w_sensitivity"] * min(diag["sensitivity"], sel["sensitivity_cap"])
            + sel["w_jitter"] * jit)


# ---------------------------------------------------------------------------
# Loot-category generator
# ---------------------------------------------------------------------------

class LootGenerator(nn.Module):
    """G_loot(x, e, z) -> category probabilities over dda_core.SLOTS.
    x = standardized condition, e = prior-equipment one-hot (known before the
    reward), z = noise. Output = floor + (1 - n_allowed*floor) * softmax over
    the ALLOWED categories (categories without enough training support get
    probability 0): finite, non-negative and summing to one by construction."""

    def __init__(self, in_dim: int, noise_dim=4, hidden=64, layers=2, min_prob=0.05,
                 allowed: Optional[Sequence[bool]] = None):
        super().__init__()
        n = len(core.SLOTS)
        allowed = [bool(a) for a in allowed] if allowed is not None else [True] * n
        if not any(allowed):
            raise ValueError("no loot category has training support")
        if min_prob * sum(allowed) >= 1.0:
            raise ValueError("min_prob too large for the number of allowed categories")
        self.noise_dim = noise_dim
        self.hparams = {"in_dim": in_dim, "noise_dim": noise_dim, "hidden": hidden, "layers": layers,
                        "min_prob": min_prob, "allowed": allowed}
        self.net = mlp(in_dim + noise_dim, hidden, layers, n, act=nn.ReLU)
        self.register_buffer("mask", torch.tensor([0.0 if a else float("-inf") for a in allowed]))
        self.register_buffer("allowed_f", torch.tensor([1.0 if a else 0.0 for a in allowed]))
        self.min_prob = float(min_prob)

    def forward(self, feats, z=None):
        if z is None:
            z = torch.randn(feats.shape[0], self.noise_dim, device=feats.device)
        sm = torch.softmax(self.net(torch.cat([feats, z], dim=1)) + self.mask, dim=1)
        k = self.allowed_f.sum()
        return self.min_prob * self.allowed_f + (1.0 - k * self.min_prob) * sm


def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
