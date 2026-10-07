"""
models.py
=========
Torch modules and losses for the outcome-guided cGAN:

  Q(x, a) -> predicted outcome vector   (OutcomeModel, use_action=True)
  Q(x)    -> state-only ablation        (OutcomeModel, use_action=False)
  G(x, z) -> candidate action           (Generator; bounded by construction)
  D(x, a) -> plausibility logit         (Discriminator)

Conventions
-----------
* x is the condition vector standardized with the TRAIN-split scaler.
* Actions inside the networks are normalized to [0, 1]^4 over
  dda_core.ACTION_BOUNDS (ml_data.normalize_action); G emits sigmoid outputs
  so every generated action is inside the design bounds.
* Q predicts [death_logit, post_hp_ratio, damage_share, combat_efficiency,
  duration_norm] where the 4 continuous outputs are standardized with the
  TRAIN-split outcome scaler. `outcomes_physical()` converts to the units
  used by balance.py (death as a probability).

The generator objective (generator_loss) is

  L_G = L_adv + lambda_balance * L_balance + lambda_diversity * L_diversity

and deliberately receives NO observed/heuristic action: there is no
reconstruction term anywhere in the final objective.
"""

from typing import Dict, List, Optional

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


class OutcomeModel(nn.Module):
    def __init__(self, cond_dim=COND_DIM, action_dim=ACTION_DIM, hidden=128, layers=3,
                 dropout=0.05, use_action=True):
        super().__init__()
        self.use_action = use_action
        self.hparams = {"cond_dim": cond_dim, "action_dim": action_dim, "hidden": hidden,
                        "layers": layers, "dropout": dropout, "use_action": use_action}
        in_dim = cond_dim + (action_dim if use_action else 0)
        self.net = mlp(in_dim, hidden, layers, OUTCOME_DIM, act=nn.SiLU, dropout=dropout)

    def forward(self, x_s, a_n=None):
        inp = torch.cat([x_s, a_n], dim=1) if self.use_action else x_s
        return self.net(inp)


def q_loss(pred, y_death, y_cont_s, death_weight: float = 1.0):
    """BCE on death + MSE on standardized continuous outcomes (observed data only)."""
    bce = F.binary_cross_entropy_with_logits(pred[:, 0], y_death)
    mse = F.mse_loss(pred[:, 1:], y_cont_s)
    return death_weight * bce + mse, bce, mse


class OutcomeScaler:
    """Torch view of the continuous-outcome Standardizer."""

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def to(self, device):
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self


def outcomes_physical(pred, out_scaler: OutcomeScaler):
    """(N, 5) raw Q output -> (N, 5) [death_prob, continuous in physical units]."""
    death = torch.sigmoid(pred[:, :1])
    cont = pred[:, 1:] * out_scaler.std + out_scaler.mean
    return torch.cat([death, cont], dim=1)


class Generator(nn.Module):
    def __init__(self, cond_dim=COND_DIM, noise_dim=8, action_dim=ACTION_DIM, hidden=128, layers=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.hparams = {"cond_dim": cond_dim, "noise_dim": noise_dim, "action_dim": action_dim,
                        "hidden": hidden, "layers": layers}
        self.net = mlp(cond_dim + noise_dim, hidden, layers, action_dim, act=nn.ReLU)

    def forward(self, x_s, z=None):
        if z is None:
            z = torch.randn(x_s.shape[0], self.noise_dim, device=x_s.device)
        return torch.sigmoid(self.net(torch.cat([x_s, z], dim=1)))   # a_n in (0, 1)^4


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
    """Mode-seeking hinge: penalizes near-identical actions from different
    noise vectors for the same condition. Zero once the action spread per
    unit of noise spread exceeds tau, so it never pushes to extremes."""
    ratio = (a1_n - a2_n).abs().mean(dim=1) / ((z1 - z2).abs().mean(dim=1) + 1e-6)
    return F.relu(tau - ratio).mean()


def generator_loss(gen: Generator, disc: Discriminator, q: OutcomeModel,
                   out_scaler: OutcomeScaler, x_s, lambda_balance: float,
                   lambda_diversity: float, diversity_tau: float,
                   exp_cfg: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    """
    L_G = L_adv + lambda_balance * L_balance + lambda_diversity * L_diversity.

    Inputs are only the conditions x: the observed/heuristic actions are
    never seen here, so G cannot be trained to reconstruct them. Q must be
    frozen (requires_grad=False) by the caller; gradients flow through Q's
    input into G only.
    """
    n = x_s.shape[0]
    z1 = torch.randn(n, gen.noise_dim, device=x_s.device)
    z2 = torch.randn(n, gen.noise_dim, device=x_s.device)
    a1 = gen(x_s, z1)
    a2 = gen(x_s, z2)

    d_fake = disc(x_s, a1)
    l_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

    y_hat = outcomes_physical(q(x_s, a1), out_scaler)
    l_bal = balance.balance_loss_torch(y_hat, exp_cfg)

    l_div = diversity_loss(a1, a2, z1, z2, diversity_tau)

    total = l_adv + lambda_balance * l_bal + lambda_diversity * l_div
    return {"total": total, "adv": l_adv, "balance": l_bal, "diversity": l_div}


GENERATOR_LOSS_TERMS = ("adv", "balance", "diversity")


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
# Generator diagnostics (validation-time model selection and evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generator_diagnostics(gen: Generator, q: OutcomeModel, out_scaler: OutcomeScaler, x_s,
                          run_index: Optional[List[str]] = None,
                          reference_actions_n: Optional[Dict[str, torch.Tensor]] = None,
                          k: int = 8, seed: int = 0, exp_cfg: Optional[dict] = None) -> dict:
    """
    Metrics on a fixed set of conditions x_s (N, COND_DIM), all in normalized
    action units ([0,1] per dimension):

      validity           share of generated actions that are finite and in
                         bounds after integer spawn rounding
      saturation         share of action components within 1% of a bound
      balance_pred       mean Q-predicted balance loss of G's actions
      balance_ref        same for each reference action set (e.g. observed,
                         heuristic, fixed) on the SAME conditions
      diversity          mean |a(z_i) - a(z_j)| across k noise draws per x
      sensitivity        mean |a(x + 1 std e_j, z) - a(x, z)| over features j
      jitter             mean |a_t - a_{t-1}| between consecutive rooms of a run
                         (fresh noise per room, as at runtime)
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

    bal = [float(balance.balance_loss_torch(outcomes_physical(q(x_s, samples[i]), out_scaler), exp_cfg))
           for i in range(k)]
    balance_pred = sum(bal) / k
    balance_ref = {}
    for name, a_ref in (reference_actions_n or {}).items():
        balance_ref[name] = float(balance.balance_loss_torch(
            outcomes_physical(q(x_s, a_ref), out_scaler), exp_cfg))

    div = []
    for i in range(k):
        for j in range(i + 1, k):
            div.append(float((samples[i] - samples[j]).abs().mean()))
    diversity = sum(div) / max(len(div), 1)

    z0 = z(n)
    base = gen(x_s, z0)
    per_feat = []
    for j in range(x_s.shape[1]):
        xp = x_s.clone()
        xp[:, j] += 1.0
        per_feat.append(float((gen(xp, z0) - base).abs().mean()))
    sensitivity = sum(per_feat) / len(per_feat)

    jitter = float("nan")
    if run_index is not None and n > 1:
        a_rt = gen(x_s, z(n))
        steps = [float((a_rt[i] - a_rt[i - 1]).abs().mean())
                 for i in range(1, n) if run_index[i] == run_index[i - 1]]
        jitter = sum(steps) / len(steps) if steps else float("nan")

    mean_action = decode_actions(samples.mean(dim=(0, 1), keepdim=False).unsqueeze(0))[0].tolist()
    return {"n_conditions": n, "k": k, "validity": validity, "saturation": saturation,
            "balance_pred": balance_pred, "balance_ref": balance_ref, "diversity": diversity,
            "sensitivity": sensitivity,
            "sensitivity_per_feature": dict(zip(ml_data.COND_COLUMNS, per_feat)),
            "jitter": jitter, "mean_action": dict(zip(core.ACTION_KEYS, mean_action))}


def selection_score(diag: dict, sel: dict) -> float:
    """Composite validation score (LOWER is better). Never the GAN loss."""
    import math
    jit = diag["jitter"] if math.isfinite(diag["jitter"]) else 0.0
    return (diag["balance_pred"]
            + sel["w_invalid"] * (1.0 - diag["validity"])
            - sel["w_diversity"] * min(diag["diversity"], sel["diversity_cap"])
            - sel["w_sensitivity"] * min(diag["sensitivity"], sel["sensitivity_cap"])
            + sel["w_jitter"] * jit)
