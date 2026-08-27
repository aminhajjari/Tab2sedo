"""
Table2Image v2 — Feature-Derived Image target.

Replaces the arbitrary class-matched FashionMNIST image target with an image
whose pixels/regions correspond to tabular features, placed on a 2-D grid so that
correlated / label-relevant features are spatially adjacent (spectral ordering
+ Hilbert curve over feature blocks). The visual modality therefore genuinely
encodes tabular structure, which addresses the "the image is artificial"
reviewer objection and lifts the accuracy ceiling.

Key changes vs. the original Table2Image script:
  * Feature-image target (no MNIST / FashionMNIST).
  * Layout similarity mixes feature-feature correlation and label MI:
        S_ij = alpha * |corr(x_i, x_j)| + (1 - alpha) * sqrt(MI_i_norm * MI_j_norm)
    then a Fiedler-vector spectral ordering + Hilbert curve places similar
    features adjacently on the grid.
  * Features are tiled into power-of-2 blocks (filling the grid, CNN-friendly);
    if there are more features than block slots, features are binned along the
    spectral order and averaged.
  * CNN classifies the RAW feature image (best accuracy); reconstruction is
    auxiliary regularization. Reconstructed-image classification is available
    as an ablation (--classify_recon 1).
  * Gated fusion of tabular and image branches + supervised contrastive latent
    regularization are kept (these are the real differentiators over the
    published Table2Image baseline).
  * No leakage: imputer / scaler / MI / layout / missing-column drops /
    categorical encoders are fit on the TRAIN fold only; a held-out validation
    fold is used for model selection, and the test fold is reported once.
  * "VIF" mislabel removed: feature importance now correctly named MI-guided.

Usage:
    python table2image_v2.py --data path/to/dataset.csv \
        --image_size 32 --sigma 1.0 --layout_alpha 0.5 --classify_recon 0

Ablations to run for a paper:
    --sigma 0 / 0.5 / 1.0          (blur effect)
    --layout_alpha 0 / 0.5 / 1.0   (MI-only / mixed / corr-only layout)
    --classify_recon 0 / 1        (raw feature image vs reconstruction)
    --con_weight 0 / 0.5          (contrastive loss on/off)
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import scipy.io.arff as arff

# ============================ ARGUMENTS ============================
parser = argparse.ArgumentParser(description="Table2Image v2 (feature-derived images)")
parser.add_argument("--data", type=str, required=True, help="Path to dataset (csv/arff/data)")
parser.add_argument("--save_dir", type=str, default="./t2i_v2_output", help="Output directory")
parser.add_argument("--num_images", type=int, default=20, help="Number of sample images to save")
parser.add_argument("--image_size", type=int, default=32, help="Image side (power of 2, e.g. 16/32)")
parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur sigma (0 disables)")
parser.add_argument("--layout_alpha", type=float, default=0.5,
                    help="Mix of |corr| vs label-MI in layout (0=MI only, 1=corr only)")
parser.add_argument("--con_weight", type=float, default=0.5, help="Weight of supervised contrastive loss")
parser.add_argument("--classify_recon", type=int, default=0,
                    help="1=classify reconstructed image, 0=classify raw feature image")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of train used for validation")
args = parser.parse_args()

EPOCH = args.epochs
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
SIGMA = args.sigma
ALPHA_LAYOUT = args.layout_alpha
CON_WEIGHT = args.con_weight
CLASSIFY_RECON = bool(args.classify_recon)
NUM_IMAGES_TO_SAVE = min(args.num_images, 20)

file_name = os.path.basename(os.path.dirname(args.data)) or os.path.basename(args.data)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TABLE2IMAGE v2 - Feature-Derived Image target")
print("=" * 70)
print(f"Dataset: {file_name} | Image: {IMAGE_SIZE}x{IMAGE_SIZE} | sigma={SIGMA} | alpha={ALPHA_LAYOUT}")
print(f"Device: {DEVICE} | Classify reconstruction: {CLASSIFY_RECON}")
print("=" * 70)

assert (IMAGE_SIZE & (IMAGE_SIZE - 1)) == 0, "image_size must be a power of 2"


# ============================ DATA LOADING ============================
def load_dataset(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".csv":
        return pd.read_csv(file_path), None
    if file_ext == ".arff":
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = df[col].str.decode("utf-8")
                except AttributeError:
                    pass
        return df, meta
    if file_ext == ".data":
        for sep in [",", " ", "\t", ";"]:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None)
                if df.shape[1] > 1:
                    return df, None
            except Exception:
                continue
        raise Exception("Could not determine delimiter for .data file")
    raise ValueError(f"Unsupported file format: {file_ext}")


df, arff_meta = load_dataset(args.data)
df = df.replace(["?", "", " ", "nan", "NaN", "NA", "null", "None", "-"], np.nan)

# target detection
target_col = None
if arff_meta is not None:
    try:
        target_col = list(arff_meta.names())[-1]
    except Exception:
        target_col = None
if target_col is None or target_col not in df.columns:
    candidates = ["target", "class", "outcome", "Class", "binaryClass", "status", "Target",
                  "label", "Label", "diagnosis", "y", "Utility", "signal", "click"]
    target_col = next((c for c in df.columns if c in candidates), None) or df.columns[-1]

# encode target (label identity only; fit on full data is acceptable)
y = LabelEncoder().fit_transform(df[target_col].astype(str))
num_classes = len(np.unique(y))
assert 2 <= num_classes <= 20, f"Need 2..20 classes, got {num_classes}"

# features
X_df = df.drop(columns=[target_col]).copy()

# SPLIT FIRST (train/val/test) to avoid leakage
X_tr_df, X_te_df, y_tr, y_te = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
X_tr_df, X_va_df, y_tr, y_va = train_test_split(X_tr_df, y_tr, test_size=args.val_frac,
                                                random_state=42, stratify=y_tr)

# drop high-missing columns based on TRAIN only
missing_pct = X_tr_df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
if cols_to_drop:
    print(f"[INFO] dropping {len(cols_to_drop)} high-missing columns (train-based)")
X_tr_df = X_tr_df.drop(columns=cols_to_drop)
X_va_df = X_va_df.drop(columns=cols_to_drop)
X_te_df = X_te_df.drop(columns=cols_to_drop)

# encode categoricals with TRAIN-only OrdinalEncoder (unknown -> -1)
cat_cols = [c for c in X_tr_df.columns if not pd.api.types.is_numeric_dtype(X_tr_df[c])]
if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1).fit(X_tr_df[cat_cols].astype(str))
    for d in (X_tr_df, X_va_df, X_te_df):
        d[cat_cols] = enc.transform(d[cat_cols].astype(str))
# to numeric
for d in (X_tr_df, X_va_df, X_te_df):
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")

X_tr = X_tr_df.to_numpy(dtype=np.float32)
X_va = X_va_df.to_numpy(dtype=np.float32)
X_te = X_te_df.to_numpy(dtype=np.float32)

# impute + scale on TRAIN only
imputer = SimpleImputer(strategy="median").fit(X_tr)
X_tr = imputer.transform(X_tr)
X_va = imputer.transform(X_va)
X_te = imputer.transform(X_te)
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr).astype(np.float32)
X_va = scaler.transform(X_va).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)

n_cont_features = X_tr.shape[1]
tab_latent_size = n_cont_features + 4
print(f"[INFO] train={X_tr.shape[0]} val={X_va.shape[0]} test={X_te.shape[0]} "
      f"features={n_cont_features} classes={num_classes}")


# ============================ HILBERT CURVE ============================
def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def d2xy(n, d):
    x = y = 0
    t = d
    s = 1
    while s < (1 << n):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_curve(n):
    n = int(n)
    return np.array([d2xy(n, d) for d in range(1 << (2 * n))], dtype=np.int64)


# ============================ FEATURE LAYOUT ============================
def compute_layout(X_train, mi_values, alpha):
    """Fiedler-vector spectral ordering so correlated / label-relevant features
    are adjacent. Returns the 1-D feature ordering."""
    F = X_train.shape[1]
    if F == 1:
        return np.array([0])
    corr = np.corrcoef(X_train.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    S_corr = np.abs(corr)
    np.fill_diagonal(S_corr, 0.0)
    mi = np.clip(np.asarray(mi_values, dtype=np.float64), 1e-6, None)
    mi = mi / (mi.max() + 1e-8)  # normalize so MI scale doesn't dominate corr
    S_mi = np.sqrt(np.outer(mi, mi))
    np.fill_diagonal(S_mi, 0.0)
    S = alpha * S_corr + (1.0 - alpha) * S_mi
    np.fill_diagonal(S, 0.0)
    L = np.diag(S.sum(axis=1)) - S
    try:
        w, V = eigh(L)
        order = np.argsort(V[:, 1])
    except Exception:
        order = np.arange(F)
    return order


def build_feature_images(X, order, image_size, sigma):
    """Tile each feature into a block that fills the grid; blocks laid out
    along a Hilbert curve (in spectral order) so correlated features are
    adjacent. If features outnumber block slots, bin along the spectral order
    and average. Pixel intensity = sigmoid(feature value). Optional blur."""
    N, F = X.shape
    vals = 1.0 / (1.0 + np.exp(-X))  # (N, F) in (0, 1)
    nb = 1
    while nb * nb < F:
        nb *= 2
    nb = max(nb, 2)
    nb = min(nb, image_size)
    bs = image_size // nb  # block pixel side (both powers of 2 -> divides evenly)
    n_h = int(np.log2(nb))
    hcurve = hilbert_curve(n_h)  # (nb*nb, 2) positions on nb x nb grid
    n_slots = len(hcurve)
    img = np.zeros((N, image_size, image_size), dtype=np.float32)
    if F <= n_slots:
        for j in range(F):
            bx, by = int(hcurve[j, 0]), int(hcurve[j, 1])
            img[:, bx * bs:(bx + 1) * bs, by * bs:(by + 1) * bs] = vals[:, order[j]][:, None, None]
    else:
        # bin features into n_slots groups along the spectral order
        groups = np.array_split(order, n_slots)
        for j, g in enumerate(groups):
            if len(g) == 0:
                continue
            bx, by = int(hcurve[j, 0]), int(hcurve[j, 1])
            img[:, bx * bs:(bx + 1) * bs, by * bs:(by + 1) * bs] = vals[:, g].mean(axis=1)[:, None, None]
    if sigma and sigma > 0:
        for i in range(N):
            img[i] = gaussian_filter(img[i], sigma=sigma)
    return img


# MI on TRAIN only
print("[INFO] computing label MI (train only)...")
mi_values = mutual_info_classif(X_tr, y_tr, random_state=42)
mi_values = np.clip(mi_values, 1e-3, None)

order = compute_layout(X_tr, mi_values, ALPHA_LAYOUT)
train_imgs = build_feature_images(X_tr, order, IMAGE_SIZE, SIGMA)
val_imgs = build_feature_images(X_va, order, IMAGE_SIZE, SIGMA)
test_imgs = build_feature_images(X_te, order, IMAGE_SIZE, SIGMA)
print(f"[INFO] feature images: train={train_imgs.shape} val={val_imgs.shape} test={test_imgs.shape}")


# ============================ DATASET ============================
class FeatureImageDataset(Dataset):
    def __init__(self, X_tab, imgs, labels):
        self.X = torch.tensor(X_tab, dtype=torch.float32)
        self.img = torch.tensor(imgs, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        assert len(self.X) == len(self.img) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.img[i], self.y[i]


train_ds = FeatureImageDataset(X_tr, train_imgs, y_tr)
val_ds = FeatureImageDataset(X_va, val_imgs, y_va)
test_ds = FeatureImageDataset(X_te, test_imgs, y_te)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)


# ============================ MODEL ============================
IMG_DIM = IMAGE_SIZE * IMAGE_SIZE


class MIGuidedInit(nn.Module):
    """Initialize first-layer weights from per-feature label MI (replaces the
    mislabeled 'VIF' init). MI weights are frozen at init, then trained."""
    def __init__(self, input_dim, mi_values):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)
        imp = torch.tensor(mi_values, dtype=torch.float32)
        imp = imp / (imp.mean() + 1e-6)
        with torch.no_grad():
            for i in range(self.fc1.weight.data.shape[0]):
                self.fc1.weight.data[i, :] = imp[i % len(imp)] / (input_dim + 4)

    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class TabMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        return z, self.fc2(z)


class ImageClassifierHead(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()
        h = img_size // 4  # after two MaxPool2d(2)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * h * h, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CAEWithTabEmbedding(nn.Module):
    def __init__(self, input_dim, tab_latent_size, num_classes, img_dim,
                 latent_size=8, mi_values=None):
        super().__init__()
        self.mlp = TabMLP(input_dim, tab_latent_size, num_classes)
        self.mi_model = MIGuidedInit(input_dim, mi_values) if mi_values is not None else None
        self.encoder = nn.Sequential(
            nn.Linear(img_dim + tab_latent_size + input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size + input_dim, 128), nn.ReLU(),
            nn.Linear(128, img_dim), nn.Sigmoid(),
        )
        self.classifier = ImageClassifierHead(num_classes, img_size=int(img_dim ** 0.5))
        self.gate = nn.Sequential(
            nn.Linear(tab_latent_size + num_classes, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, img_flat, tab_data):
        mi_emb = self.mi_model(tab_data) if self.mi_model is not None else tab_data
        tab_emb, tab_pred = self.mlp(tab_data)
        z = self.encoder(torch.cat([img_flat, tab_emb, mi_emb], dim=1))
        recon = self.decoder(torch.cat([z, tab_emb, mi_emb], dim=1))
        cls_input = recon if CLASSIFY_RECON else img_flat
        side = int(IMG_DIM ** 0.5)
        img_pred = self.classifier(cls_input.view(-1, 1, side, side))
        a = self.gate(torch.cat([tab_emb, img_pred], dim=1))
        fused = a * img_pred + (1 - a) * tab_pred
        return recon, tab_pred, img_pred, fused, z


def supcon_loss(z, labels, temperature=0.1):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    eye = torch.eye(mask.shape[0], device=z.device)
    mask = mask * (1 - eye)
    exp_sim = torch.exp(sim) * (1 - eye)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    return -(mask * log_prob).sum(1).div(mask.sum(1) + 1e-8).mean()


model = CAEWithTabEmbedding(
    input_dim=n_cont_features, tab_latent_size=tab_latent_size,
    num_classes=num_classes, img_dim=IMG_DIM, latent_size=8,
    mi_values=mi_values,
).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
print(f"[INFO] model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def loss_fn(recon, img_target, tab_pred, tab_lbl, img_pred, img_lbl, fused, z):
    bce = F.mse_loss(recon, img_target)
    tab_l = F.cross_entropy(tab_pred, tab_lbl)
    img_l = F.cross_entropy(img_pred, img_lbl)
    fused_l = F.cross_entropy(fused, tab_lbl)
    con_l = supcon_loss(z, tab_lbl)
    return bce + tab_l + img_l + fused_l + CON_WEIGHT * con_l


def train_epoch(loader):
    model.train()
    tot = 0.0
    for x_tab, img, lbl in loader:
        x_tab, img, lbl = x_tab.to(DEVICE), img.to(DEVICE), lbl.to(DEVICE)
        img_flat = img.view(-1, IMG_DIM)
        optimizer.zero_grad()
        recon, tab_pred, img_pred, fused, z = model(img_flat, x_tab)
        loss = loss_fn(recon, img_flat, tab_pred, lbl, img_pred, lbl, fused, z)
        loss.backward()
        optimizer.step()
        tot += loss.item()
    return tot / max(1, len(loader))


@torch.no_grad()
def eval_loader(loader):
    model.eval()
    correct_tab = correct_img = correct_fused = total = 0
    tab_probs, fused_probs, labels_all = [], [], []
    for x_tab, img, lbl in loader:
        x_tab, img, lbl = x_tab.to(DEVICE), img.to(DEVICE), lbl.to(DEVICE)
        img_flat = img.view(-1, IMG_DIM)
        recon, tab_pred, img_pred, fused, z = model(img_flat, x_tab)
        correct_tab += (tab_pred.argmax(1) == lbl).sum().item()
        correct_img += (img_pred.argmax(1) == lbl).sum().item()
        correct_fused += (fused.argmax(1) == lbl).sum().item()
        total += lbl.size(0)
        tab_probs.append(F.softmax(tab_pred, 1).cpu().numpy())
        fused_probs.append(F.softmax(fused, 1).cpu().numpy())
        labels_all.append(lbl.cpu().numpy())
    labels_all = np.concatenate(labels_all)
    tab_probs = np.vstack(tab_probs)
    fused_probs = np.vstack(fused_probs)

    def auc(probs):
        try:
            return (roc_auc_score(labels_all, probs[:, 1]) if num_classes == 2
                    else roc_auc_score(labels_all, probs, multi_class="ovr", average="macro"))
        except Exception:
            return 0.0
    return (100 * correct_tab / total, 100 * correct_img / total,
            100 * correct_fused / total, auc(tab_probs), auc(fused_probs))


# ============================ TRAINING (val-based model selection) ============================
best_val_acc = -1.0
best_state = None
best_epoch = 0
for ep in range(1, EPOCH + 1):
    tr = train_epoch(train_loader)
    vta, via, vfa, vtauc, vfauc = eval_loader(val_loader)
    if vfa > best_val_acc:
        best_val_acc = vfa
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = ep
    if ep == 1 or ep % 10 == 0:
        print(f"[Epoch {ep:3d}] train_loss={tr:.4f} | val tab={vta:.2f}% img={via:.2f}% "
              f"fused={vfa:.2f}% | valAUC={vfauc:.4f}")

# restore best-by-val weights, evaluate on TEST once
model.load_state_dict(best_state)
tta, tia, tfa, ttauc, tfauc = eval_loader(test_loader)

print("\n" + "=" * 70)
print(f"Best val epoch: {best_epoch} (val fused acc {best_val_acc:.2f}%)")
print(f"TEST  | tab={tta:.2f}% img={tia:.2f}% fused={tfa:.2f}% | "
      f"tabAUC={ttauc:.4f} fusedAUC={tfauc:.4f}")
print("=" * 70)


# ============================ SAVE SAMPLE IMAGES ============================
os.makedirs(args.save_dir, exist_ok=True)
n = min(NUM_IMAGES_TO_SAVE, len(test_ds))
side = int(IMG_DIM ** 0.5)
fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
for i in range(n):
    x_tab, img, lbl = test_ds[i]
    with torch.no_grad():
        recon, *_ = model(img.view(1, -1).to(DEVICE), x_tab.view(1, -1).to(DEVICE))
    axes[0, i].imshow(img.numpy(), cmap="gray")
    axes[0, i].set_title(f"feat cls {int(lbl)}", fontsize=7)
    axes[0, i].axis("off")
    axes[1, i].imshow(recon.view(side, side).cpu().numpy(), cmap="gray")
    axes[1, i].set_title("recon", fontsize=7)
    axes[1, i].axis("off")
plt.suptitle(f"{file_name} - feature images (top) vs reconstructions (bottom)", fontsize=10)
plt.tight_layout()
grid_path = os.path.join(args.save_dir, f"{file_name}_feature_images.png")
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[INFO] saved sample grid -> {grid_path}")

results = {
    "dataset": file_name, "image_size": IMAGE_SIZE, "sigma": SIGMA,
    "layout_alpha": ALPHA_LAYOUT, "classify_recon": CLASSIFY_RECON,
    "features": n_cont_features, "classes": num_classes,
    "best_val_epoch": best_epoch,
    "test_tab_accuracy": round(tta, 4), "test_img_accuracy": round(tia, 4),
    "test_fused_accuracy": round(tfa, 4),
    "test_tab_auc": round(ttauc, 4), "test_fused_auc": round(tfauc, 4),
    "timestamp": datetime.now().isoformat(),
}
with open(os.path.join(args.save_dir, f"{file_name}_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
