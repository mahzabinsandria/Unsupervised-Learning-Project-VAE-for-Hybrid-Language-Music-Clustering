from __future__ import annotations
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa
import matplotlib.pyplot as plt
import umap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from clustering import kmeans_labels, agglomerative_labels, dbscan_labels
from evaluation import eval_internal, eval_with_labels


ROOT = Path(__file__).resolve().parents[1]
DATA_LYRICS = ROOT / "data" / "lyrics"
RESULTS = ROOT / "results"
LATENT_DIR = RESULTS / "latent_visualization"
METRICS_CSV = RESULTS / "clustering_metrics.csv"


# -------------------------
# utils
# -------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LATENT_DIR.mkdir(parents=True, exist_ok=True)

def umap_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)

def save_scatter(X2: np.ndarray, labels: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(7,6))
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=12)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path.as_posix(), dpi=200)
    plt.close()

def update_metrics(task: str, rows: list[dict]):
    """Write/replace rows for this task into results/clustering_metrics.csv."""
    new_df = pd.DataFrame(rows)
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        old = old[old["task"] != task].copy() if "task" in old.columns else old
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(METRICS_CSV, index=False)
    print("Saved:", METRICS_CSV)

def load_easy_csv() -> pd.DataFrame:
    path = DATA_LYRICS / "tracks_easy.csv"
    df = pd.read_csv(path)
    df["lyrics"] = df["lyrics"].fillna("").astype(str)
    df["language"] = df["language"].fillna("unknown").astype(str)
    return df

def load_av_csv() -> pd.DataFrame:
    path = DATA_LYRICS / "tracks_av.csv"
    df = pd.read_csv(path)
    df["lyrics"] = df["lyrics"].fillna("").astype(str)
    if "language" not in df.columns:
        df["language"] = "unknown"
    if "genre" not in df.columns:
        df["genre"] = "unknown"
    df["language"] = df["language"].fillna("unknown").astype(str)
    df["genre"] = df["genre"].fillna("unknown").astype(str)
    return df

class TfidfEmbedder:
    def __init__(self, max_features=5000, min_df=2):
        self.vec = TfidfVectorizer(max_features=max_features, min_df=min_df, stop_words="english", lowercase=True)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        X = self.vec.fit_transform(texts)
        return X.toarray().astype(np.float32)

# -------------------------
# audio features
# -------------------------
def mel_spec(path: str, sr=22050, n_mels=64, max_frames=256) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S = librosa.power_to_db(S, ref=np.max)
    if S.shape[1] < max_frames:
        pad = max_frames - S.shape[1]
        S = np.pad(S, ((0,0),(0,pad)), mode="constant", constant_values=S.min())
    else:
        S = S[:, :max_frames]
    S = (S - S.mean()) / (S.std() + 1e-6)
    return S.astype(np.float32)

def mfcc_stats(path: str, sr=22050, n_mfcc=20) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0).astype(np.float32)

def one_hot(codes: np.ndarray) -> np.ndarray:
    n = int(codes.max()) + 1
    out = np.zeros((len(codes), n), dtype=np.float32)
    out[np.arange(len(codes)), codes] = 1.0
    return out

# -------------------------
# models
# -------------------------
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class TextVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim=32, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, in_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

class MultiModalVAE(nn.Module):
    def __init__(self, text_dim: int, latent_dim=32, n_mels=64, max_frames=256, cond_dim=0):
        super().__init__()
        self.n_mels, self.max_frames, self.cond_dim = n_mels, max_frames, cond_dim

        self.a_enc = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,1,n_mels,max_frames)
            self.a_flat = self.a_enc(dummy).numel()

        self.a_fc = nn.Sequential(nn.Linear(self.a_flat, 256), nn.ReLU())
        self.t_fc = nn.Sequential(nn.Linear(text_dim + cond_dim, 256), nn.ReLU(),
                                  nn.Linear(256,256), nn.ReLU())
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

        self.a_dec_fc = nn.Sequential(nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
                                      nn.Linear(256, self.a_flat), nn.ReLU())
        self.a_dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,stride=2,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,stride=2,padding=1),
        )
        self.t_dec = nn.Sequential(nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
                                   nn.Linear(256,256), nn.ReLU(),
                                   nn.Linear(256, text_dim))

    def forward(self, a, t, cond=None):
        if cond is None:
            cond = torch.zeros((a.size(0), self.cond_dim), device=a.device)

        ah = self.a_fc(self.a_enc(a).view(a.size(0), -1))
        t_in = torch.cat([t, cond], dim=1) if self.cond_dim else t
        th = self.t_fc(t_in)

        h = torch.cat([ah, th], dim=1)
        mu, logvar = self.mu(h), self.logvar(h)
        z = reparameterize(mu, logvar)

        z_in = torch.cat([z, cond], dim=1) if self.cond_dim else z
        a_flat = self.a_dec_fc(z_in).view(a.size(0), 64, self.n_mels//8, self.max_frames//8)
        a_hat = self.a_dec(a_flat)[:, :, :self.n_mels, :self.max_frames]
        t_hat = self.t_dec(z_in)
        return a_hat, t_hat, mu, logvar

class MultiModalAE(nn.Module):
    def __init__(self, text_dim: int, latent_dim=32, n_mels=64, max_frames=256, cond_dim=0):
        super().__init__()
        self.n_mels, self.max_frames, self.cond_dim = n_mels, max_frames, cond_dim

        self.a_enc = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,1,n_mels,max_frames)
            self.a_flat = self.a_enc(dummy).numel()

        self.a_fc = nn.Sequential(nn.Linear(self.a_flat, 256), nn.ReLU())
        self.t_fc = nn.Sequential(nn.Linear(text_dim + cond_dim, 256), nn.ReLU(),
                                  nn.Linear(256,256), nn.ReLU())
        self.z_fc = nn.Linear(512, latent_dim)

        self.a_dec_fc = nn.Sequential(nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
                                      nn.Linear(256, self.a_flat), nn.ReLU())
        self.a_dec = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,stride=2,padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,stride=2,padding=1),
        )
        self.t_dec = nn.Sequential(nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
                                   nn.Linear(256,256), nn.ReLU(),
                                   nn.Linear(256, text_dim))

    def encode(self, a, t, cond=None):
        if cond is None:
            cond = torch.zeros((a.size(0), self.cond_dim), device=a.device)
        ah = self.a_fc(self.a_enc(a).view(a.size(0), -1))
        t_in = torch.cat([t, cond], dim=1) if self.cond_dim else t
        th = self.t_fc(t_in)
        return self.z_fc(torch.cat([ah, th], dim=1))

    def forward(self, a, t, cond=None):
        if cond is None:
            cond = torch.zeros((a.size(0), self.cond_dim), device=a.device)
        z = self.encode(a, t, cond)
        z_in = torch.cat([z, cond], dim=1) if self.cond_dim else z
        a_flat = self.a_dec_fc(z_in).view(a.size(0), 64, self.n_mels//8, self.max_frames//8)
        a_hat = self.a_dec(a_flat)[:, :, :self.n_mels, :self.max_frames]
        t_hat = self.t_dec(z_in)
        return a_hat, t_hat, z


# -------------------------
# training
# -------------------------
def train_text_vae(X: np.ndarray, epochs=10, batch=64, lr=1e-3, beta=1.0, device="cpu") -> np.ndarray:
    model = TextVAE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(X, dtype=torch.float32).to(device)

    for _ in range(epochs):
        model.train()
        idx = torch.randperm(x.size(0), device=device)
        for i in range(0, x.size(0), batch):
            b = idx[i:i+batch]
            xb = x[b]
            x_hat, mu, logvar = model(xb)
            recon = F.mse_loss(x_hat, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        h = model.enc(x)
        z = model.mu(h)
    return z.detach().cpu().numpy()

def train_mm_vae(A, T, cond, epochs=25, batch=16, lr=1e-3, beta=4.0, device="cpu"):
    cond_dim = 0 if cond is None else cond.shape[1]
    model = MultiModalVAE(T.shape[1], cond_dim=cond_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    a = torch.tensor(A, dtype=torch.float32).to(device)
    t = torch.tensor(T, dtype=torch.float32).to(device)
    c = None if cond is None else torch.tensor(cond, dtype=torch.float32).to(device)

    for _ in range(epochs):
        model.train()
        idx = torch.randperm(a.size(0), device=device)
        for i in range(0, a.size(0), batch):
            b = idx[i:i+batch]
            ab, tb = a[b], t[b]
            cb = None if c is None else c[b]
            a_hat, t_hat, mu, logvar = model(ab, tb, cb)
            loss_a = F.mse_loss(a_hat, ab)
            loss_t = F.mse_loss(t_hat, tb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_a + loss_t + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        ah = model.a_fc(model.a_enc(a).view(a.size(0), -1))
        t_in = t if c is None else torch.cat([t, c], dim=1)
        th = model.t_fc(t_in)
        z = model.mu(torch.cat([ah, th], dim=1))
    return model, z.detach().cpu().numpy()

def train_mm_ae(A, T, cond, epochs=25, batch=16, lr=1e-3, device="cpu"):
    cond_dim = 0 if cond is None else cond.shape[1]
    model = MultiModalAE(T.shape[1], cond_dim=cond_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    a = torch.tensor(A, dtype=torch.float32).to(device)
    t = torch.tensor(T, dtype=torch.float32).to(device)
    c = None if cond is None else torch.tensor(cond, dtype=torch.float32).to(device)

    for _ in range(epochs):
        model.train()
        idx = torch.randperm(a.size(0), device=device)
        for i in range(0, a.size(0), batch):
            b = idx[i:i+batch]
            ab, tb = a[b], t[b]
            cb = None if c is None else c[b]
            a_hat, t_hat, _ = model(ab, tb, cb)
            loss = F.mse_loss(a_hat, ab) + F.mse_loss(t_hat, tb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(a, t, c)
    return model, z.detach().cpu().numpy()


# -------------------------
# tasks
# -------------------------
def run_easy(k=8):
    ensure_dirs()
    df = load_easy_csv()
    X = TfidfEmbedder(5000).fit_transform(df["lyrics"].tolist())

    # PCA baseline
    Xp = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    y_pca = kmeans_labels(Xp, k)
    m_pca = eval_internal(Xp, y_pca)

    # VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z = train_text_vae(X, device=device)
    y_vae = kmeans_labels(Z, k)
    m_vae = eval_internal(Z, y_vae)

    U = umap_2d(Z)
    save_scatter(U, y_vae, "Easy: VAE latent + KMeans (UMAP)", LATENT_DIR / "easy_umap.png")

    update_metrics("easy", [
        {"task":"easy", "method":"PCA+KMeans", **m_pca},
        {"task":"easy", "method":"VAE+KMeans", **m_vae},
    ])

def run_medium(k=6):
    ensure_dirs()
    df = load_av_csv().sample(n=min(120, len(load_av_csv())), random_state=42).reset_index(drop=True)

    T = TfidfEmbedder(3000).fit_transform(df["lyrics"].tolist())
    A = np.stack([mel_spec(p) for p in tqdm(df["audio_path"].tolist(), desc="Mel")], axis=0)[:, None, :, :]

    # partial labels: language
    lang_codes, _ = pd.factorize(df["language"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, Z = train_mm_vae(A, T, cond=None, beta=1.0, device=device)

    y_k = kmeans_labels(Z, k)
    y_a = agglomerative_labels(Z, k)
    y_d = dbscan_labels(Z)

    save_scatter(umap_2d(Z), y_k, "Medium: mmVAE latent + KMeans (UMAP)", LATENT_DIR / "medium_umap.png")

    rows = []
    for name, y in [("KMeans", y_k), ("Agglomerative", y_a), ("DBSCAN", y_d)]:
        rows.append({
            "task":"medium",
            "method":f"mmVAE+{name}",
            **eval_internal(Z, y),
            **eval_with_labels(lang_codes, y)
        })
    update_metrics("medium", rows)

    # reconstruction example (mel)
    model.eval()
    with torch.no_grad():
        a0 = torch.tensor(A[:1], dtype=torch.float32).to(device)
        t0 = torch.tensor(T[:1], dtype=torch.float32).to(device)
        a_hat, _, _, _ = model(a0, t0, None)
        recon = a_hat.detach().cpu().numpy()[0,0]
        orig = A[0,0]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(orig, aspect="auto"); plt.title("Original mel"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(recon, aspect="auto"); plt.title("Reconstructed mel"); plt.axis("off")
    plt.tight_layout()
    plt.savefig((LATENT_DIR / "medium_mel_recon.png").as_posix(), dpi=200)
    plt.close()

def run_hard(k=6, beta=4.0):
    ensure_dirs()
    df = load_av_csv().sample(n=min(140, len(load_av_csv())), random_state=42).reset_index(drop=True)

    # genre conditioning + labels
    top = df["genre"].value_counts().head(6).index
    df["genre_s"] = df["genre"].where(df["genre"].isin(top), other="other")
    y_true, _ = pd.factorize(df["genre_s"])
    cond = one_hot(y_true)

    T = TfidfEmbedder(3000).fit_transform(df["lyrics"].tolist())
    A = np.stack([mel_spec(p) for p in tqdm(df["audio_path"].tolist(), desc="Mel")], axis=0)[:, None, :, :]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Beta-VAE (main)
    vae_model, Z_vae = train_mm_vae(A, T, cond=cond, beta=beta, device=device)
    y_vae = kmeans_labels(Z_vae, k)

    # AE baseline
    ae_model, Z_ae = train_mm_ae(A, T, cond=cond, device=device)
    y_ae = kmeans_labels(Z_ae, k)

    # PCA baseline (concat)
    audio_pool = A.reshape(A.shape[0], -1)
    Xcat = np.concatenate([audio_pool, T], axis=1)
    Xp = PCA(n_components=min(50, Xcat.shape[1]), random_state=42).fit_transform(Xcat)
    y_pca = kmeans_labels(Xp, k)

    # MFCC baseline
    mfcc = np.stack([mfcc_stats(p) for p in tqdm(df["audio_path"].tolist(), desc="MFCC")], axis=0)
    y_mfcc = kmeans_labels(mfcc, k)

    # plots
    save_scatter(umap_2d(Z_ae),  y_ae,  "Hard: AE latent + KMeans (UMAP)", LATENT_DIR / "hard_ae_umap.png")
    save_scatter(umap_2d(Z_vae), y_vae, f"Hard: Beta-VAE(beta={beta}) + KMeans (UMAP)", LATENT_DIR / "hard_vae_umap.png")

    rows = [
        {"task":"hard", "method":"PCA+KMeans", **eval_internal(Xp, y_pca), **eval_with_labels(y_true, y_pca)},
        {"task":"hard", "method":"MFCC+KMeans", **eval_internal(mfcc, y_mfcc), **eval_with_labels(y_true, y_mfcc)},
        {"task":"hard", "method":"AE+KMeans", **eval_internal(Z_ae, y_ae), **eval_with_labels(y_true, y_ae)},
        {"task":"hard", "method":f"Beta-VAE(beta={beta})+KMeans", **eval_internal(Z_vae, y_vae), **eval_with_labels(y_true, y_vae)},
    ]
    update_metrics("hard", rows)

    # reconstruction example (original vs AE vs VAE)
    ae_model.eval()
    vae_model.eval()
    with torch.no_grad():
        a0 = torch.tensor(A[:1], dtype=torch.float32).to(device)
        t0 = torch.tensor(T[:1], dtype=torch.float32).to(device)
        c0 = torch.tensor(cond[:1], dtype=torch.float32).to(device)

        a_hat_ae, _, _ = ae_model(a0, t0, c0)
        a_hat_vae, _, _, _ = vae_model(a0, t0, c0)

        orig = A[0,0]
        reco_ae = a_hat_ae.detach().cpu().numpy()[0,0]
        reco_vae = a_hat_vae.detach().cpu().numpy()[0,0]

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.imshow(orig, aspect="auto"); plt.title("Original mel"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(reco_ae, aspect="auto"); plt.title("AE recon"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(reco_vae, aspect="auto"); plt.title("Beta-VAE recon"); plt.axis("off")
    plt.tight_layout()
    plt.savefig((LATENT_DIR / "hard_mel_recon.png").as_posix(), dpi=200)
    plt.close()


def main():
    seed_all(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["easy","medium","hard","all"], required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--beta", type=float, default=4.0)
    args = ap.parse_args()

    if args.task == "easy":
        run_easy(k=args.k)
    elif args.task == "medium":
        run_medium(k=args.k)
    elif args.task == "hard":
        run_hard(k=args.k, beta=args.beta)
    else:
        run_easy(k=8)
        run_medium(k=6)
        run_hard(k=6, beta=args.beta)

    print("Done.")
    print("Metrics:", METRICS_CSV)
    print("Plots:", LATENT_DIR)


if __name__ == "__main__":
    main()
