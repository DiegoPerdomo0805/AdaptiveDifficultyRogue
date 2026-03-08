import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# ----------------------------
# Conditioning vector = your thesis metrics
# ----------------------------
def build_features(df: pd.DataFrame) -> np.ndarray:
    # Convert raw counters into stable features
    melee_k = df["melee_kills"].to_numpy(dtype=np.float32)
    magic_k = df["magic_kills"].to_numpy(dtype=np.float32)
    melee_h = df["melee_hits"].to_numpy(dtype=np.float32)
    magic_h = df["magic_hits"].to_numpy(dtype=np.float32)
    dmg_taken = df["damage_taken"].to_numpy(dtype=np.float32)
    dmg_dealt = df["damage_dealt"].to_numpy(dtype=np.float32)
    deaths = df["deaths"].to_numpy(dtype=np.float32)
    time_alive = df["time_alive"].to_numpy(dtype=np.float32)

    total_k = melee_k + magic_k
    melee_ratio = np.where(total_k > 0, melee_k / total_k, 0.5)
    magic_ratio = np.where(total_k > 0, magic_k / total_k, 0.5)

    hpk_melee = np.where(melee_k > 0, melee_h / melee_k, melee_h + 1.0)
    hpk_magic = np.where(magic_k > 0, magic_h / magic_k, magic_h + 1.0)

    dmg_ratio = np.where(dmg_dealt > 1e-6, dmg_taken / dmg_dealt, 1.0)
    death_rate = np.where(time_alive > 1e-6, (deaths / time_alive) * 60.0, 0.0)  # per minute

    # clamp
    hpk_melee = np.clip(hpk_melee, 0.0, 10.0)
    hpk_magic = np.clip(hpk_magic, 0.0, 10.0)
    dmg_ratio = np.clip(dmg_ratio, 0.0, 5.0)
    death_rate = np.clip(death_rate, 0.0, 5.0)

    X = np.stack([melee_ratio, magic_ratio, hpk_melee, hpk_magic, dmg_ratio, death_rate], axis=1)
    return X.astype(np.float32)


# ----------------------------
# Targets to generate (content tuning vector)
# ----------------------------
def build_targets(df: pd.DataFrame) -> np.ndarray:
    # y = [enemy_hp_mult, enemy_dmg_mult, enemy_speed_mult, spawn_count_norm]
    hp = df["enemy_hp_mult"].to_numpy(dtype=np.float32)
    dmg = df["enemy_dmg_mult"].to_numpy(dtype=np.float32)
    spd = df["enemy_speed_mult"].to_numpy(dtype=np.float32)
    spawn = df["spawn_count"].to_numpy(dtype=np.float32)

    # normalize spawn to 0..1 (assume 1..12 typical)
    spawn_norm = np.clip((spawn - 1.0) / 11.0, 0.0, 1.0)

    # clamp multipliers to sane training ranges
    hp = np.clip(hp, 0.5, 3.0)
    dmg = np.clip(dmg, 0.5, 3.0)
    spd = np.clip(spd, 0.5, 2.5)

    # scale to 0..1 for sigmoid generator convenience
    # these min/max must match runtime decode
    hp01 = (hp - 0.5) / (3.0 - 0.5)
    dmg01 = (dmg - 0.5) / (3.0 - 0.5)
    spd01 = (spd - 0.5) / (2.5 - 0.5)

    Y = np.stack([hp01, dmg01, spd01, spawn_norm], axis=1).astype(np.float32)
    return Y


# ----------------------------
# Archetype classifier (simple, learned from labels derived from behavior)
# ----------------------------
def derive_archetype_labels(X: np.ndarray) -> np.ndarray:
    """
    Essentials-first: pseudo-labels from heuristics.
    Later: replace with actual user-study labels or clustering.
    Labels:
      0 Knight-ish: melee heavy + low dmg_ratio
      1 Berserker-ish: melee heavy + high dmg_ratio + higher death_rate
      2 Spell Sniper-ish: magic heavy
    """
    melee_ratio = X[:, 0]
    magic_ratio = X[:, 1]
    dmg_ratio = X[:, 4]
    death_rate = X[:, 5]

    labels = np.zeros(len(X), dtype=np.int64)
    labels[magic_ratio > 0.6] = 2
    labels[(melee_ratio > 0.6) & (dmg_ratio > 1.2) & (death_rate > 0.3)] = 1
    return labels


class ArchetypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------
# cGAN
# ----------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=10, cond_dim=3, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Sigmoid(),  # outputs in 0..1
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, in_dim=4, cond_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, y, c):
        return self.net(torch.cat([y, c], dim=1))


def one_hot(labels: torch.Tensor, n=3) -> torch.Tensor:
    return torch.eye(n, device=labels.device)[labels]


def train_models(df: pd.DataFrame, out_dir: str, epochs: int = 80, batch_size: int = 64, device: str = "cpu"):
    X = build_features(df)
    Y = build_targets(df)
    labels = derive_archetype_labels(X)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    L_t = torch.tensor(labels, dtype=torch.long, device=device)

    dataset = TensorDataset(X_t, Y_t, L_t)
    n_train = int(0.85 * len(dataset))
    train_set, val_set = random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 1) classifier
    clf = ArchetypeClassifier().to(device)
    opt_c = optim.Adam(clf.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for ep in range(25):
        clf.train()
        for xb, _, lb in train_loader:
            logits = clf(xb)
            loss = ce(logits, lb)
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()

    # 2) cGAN
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    for ep in range(epochs):
        G.train(); D.train()
        for xb, yb, _ in train_loader:
            with torch.no_grad():
                c_logits = clf(xb)
                c = torch.softmax(c_logits, dim=1)  # soft condition (3)

            bs = xb.size(0)
            real = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            # D step
            z = torch.randn(bs, 10, device=device)
            y_fake = G(z, c).detach()
            d_real = D(yb, c)
            d_fake = D(y_fake, c)
            loss_d = bce(d_real, real) + bce(d_fake, fake)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # G step
            z = torch.randn(bs, 10, device=device)
            y_gen = G(z, c)
            d_gen = D(y_gen, c)
            loss_g = bce(d_gen, real)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if ep % 10 == 0 or ep == epochs - 1:
            print(f"Epoch {ep:03d} | D {loss_d.item():.4f} | G {loss_g.item():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    torch.save(clf.state_dict(), os.path.join(out_dir, "classifier.pt"))
    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
    print("Saved:", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV produced by game.py (runs.csv)")
    ap.add_argument("--out", default="models", help="output dir for .pt weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if len(df) < 200:
        print(f"[WARN] Only {len(df)} rows. GANs like data. Expect mediocre outputs until you log more runs.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_models(df, args.out, epochs=args.epochs, batch_size=args.batch, device=device)

if __name__ == "__main__":
    main()