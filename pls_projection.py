"""
Supervised Decorrelated Projection (SDP) for Tab2sedo / Table2Image-VIF.

Contribution summary
--------------------
1. PLS-DA warm start: the tabular input (p features) is mapped to k supervised
   latent components with a projection initialised from PLS-DA weights, so the
   KAN/CVAE width scales with k instead of p (parameters O(k^2) not O(p^2)).
2. End-to-end refinement: the projection is a trainable layer (not a frozen
   preprocessing step); column norms are fixed at their PLS values so the score
   scale stays inside the KAN spline grid.
3. Differentiable VIF regulariser: VIF_j = [R^-1]_jj where R is the batch
   correlation matrix of the latent scores. Penalising mean(log VIF) keeps the
   components decorrelated while they are fine-tuned, turning the original
   one-off VIF heuristic into a training-time constraint on the representation.
4. Adaptive k: smallest k whose held-out R^2_Y (on a split of the training
   set) reaches a fraction `var_keep` of the best held-out R^2_Y up to k_max.
5. Attribution back-projection: KAN component relevance is mapped back to the
   original features through |W|, and compared with classical PLS VIP scores.

PCA is provided with the same interface as the unsupervised ablation baseline.
"""
import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


# --------------------------------------------------------------------------- #
#  Differentiable VIF
# --------------------------------------------------------------------------- #
def batch_vif(T: torch.Tensor, shrink: float = 0.05) -> torch.Tensor:
    """Per-component VIF of a batch of latent scores T [B, k]: diag(R^-1).

    R is shrunk towards I so it stays invertible for small batches.
    """
    B, k = T.shape
    Tc = T - T.mean(0, keepdim=True)
    Tn = Tc / (Tc.std(0, keepdim=True) + 1e-6)
    R = (Tn.t() @ Tn) / (B - 1)
    eye = torch.eye(k, device=T.device, dtype=T.dtype)
    R = (1.0 - shrink) * R + shrink * eye
    return torch.diagonal(torch.linalg.inv(R))


def vif_loss(T: torch.Tensor, shrink: float = 0.05) -> torch.Tensor:
    """mean(log VIF) >= 0; zero iff the batch scores are uncorrelated."""
    B, k = T.shape
    if k < 2 or B < k + 2:            # too few rows for a meaningful estimate
        return T.new_zeros(())
    return torch.log(batch_vif(T, shrink).clamp_min(1.0)).mean()


# --------------------------------------------------------------------------- #
#  Fitting the initial projection
# --------------------------------------------------------------------------- #
def _pls_r2y_curve(T, Q, Yc):
    """Cumulative explained Y-variance for the first 1..k PLS components."""
    ss_tot = (Yc ** 2).sum()
    r2 = []
    for j in range(1, T.shape[1] + 1):
        Yhat = T[:, :j] @ Q[:, :j].T
        r2.append(1.0 - ((Yc - Yhat) ** 2).sum() / ss_tot)
    return np.array(r2)


def vip_scores(pls: PLSRegression, k: int) -> np.ndarray:
    """Classical PLS Variable Importance in Projection for the first k comps."""
    W = pls.x_weights_[:, :k]                         # [p, k]
    T = pls.x_scores_[:, :k]
    Q = pls.y_loadings_[:, :k]
    ss = (T ** 2).sum(0) * (Q ** 2).sum(0)            # Y-SS explained per comp
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    p = W.shape[0]
    return np.sqrt(p * (Wn ** 2 @ ss) / (ss.sum() + 1e-12))


def fit_projection(method, X_train, y_train, num_classes, k=None, k_max=32,
                   var_keep=0.95, max_fit_rows=20000, seed=42):
    """Fit PLS-DA or PCA on the (already standardised) training set only.

    Returns (W [p,k], x_mean [p], info dict).
    """
    n, p = X_train.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, max_fit_rows, replace=False) if n > max_fit_rows else np.arange(n)
    Xf, yf = X_train[idx], y_train[idx]

    k_cap = int(max(1, min(p, len(Xf) - 1, k if k else k_max)))
    k_min = int(min(k_cap, max(2, num_classes - 1)))
    info = {"method": method, "p": p}

    if method == "pls":
        Y = np.eye(num_classes)[yf]
        if k is None:
            # choose k on a held-out part of the TRAINING set (train R2_Y is
            # optimistic when p >> n), then refit on all training rows
            perm = rng.permutation(len(Xf))
            n_val = max(num_classes * 2, int(0.2 * len(Xf)))
            va, fi = perm[:n_val], perm[n_val:]
            k_try = int(max(1, min(k_cap, len(fi) - 1)))
            pls_cv = PLSRegression(n_components=k_try, scale=False, max_iter=1000).fit(Xf[fi], Y[fi])
            Tva = (Xf[va] - pls_cv._x_mean) @ pls_cv.x_rotations_
            Yva_c = Y[va] - pls_cv._y_mean
            r2_val = _pls_r2y_curve(Tva, pls_cv.y_loadings_, Yva_c)
            best = r2_val.max()
            k_sel = int(np.argmax(r2_val >= var_keep * best) + 1) if best > 0 else k_min
            k_sel = int(min(max(k_sel, k_min), k_cap))
            info["r2y_val_curve"] = [round(float(v), 4) for v in r2_val]
        else:
            k_sel = k_cap
        pls = PLSRegression(n_components=k_sel, scale=False, max_iter=1000).fit(Xf, Y)
        r2 = _pls_r2y_curve(pls.x_scores_, pls.y_loadings_, Y - Y.mean(0))
        W = pls.x_rotations_
        x_mean = pls._x_mean
        info.update(k=k_sel, r2y=float(r2[-1]),
                    r2y_val=float(info["r2y_val_curve"][k_sel - 1]) if "r2y_val_curve" in info else None,
                    vip=vip_scores(pls, k_sel).tolist())
    elif method == "pca":
        pca = PCA(n_components=k_cap, random_state=seed).fit(Xf)
        cum = np.cumsum(pca.explained_variance_ratio_)
        if k is None:
            k_sel = int(np.argmax(cum >= var_keep * cum[-1]) + 1)
            k_sel = max(k_sel, k_min)
        else:
            k_sel = k_cap
        W = pca.components_[:k_sel].T
        x_mean = pca.mean_
        info.update(k=k_sel, r2x=float(cum[k_sel - 1]))
    else:
        raise ValueError(f"unknown projection method {method}")

    return W.astype(np.float32), np.asarray(x_mean, dtype=np.float32), info


# --------------------------------------------------------------------------- #
#  Trainable projection layer
# --------------------------------------------------------------------------- #
class SupervisedProjection(nn.Module):
    """T = ((x - mu) @ W - t_mu) / t_sd, with W = c * V / ||V||_col.

    V is trainable (direction), column norms c are fixed at their PLS/PCA values,
    and (t_mu, t_sd) are the training-score statistics at initialisation, so the
    output starts as z-scored, (near-)orthogonal components.
    """

    def __init__(self, W, x_mean, X_train_ref, trainable=True):
        super().__init__()
        W = torch.as_tensor(W, dtype=torch.float32)
        self.V = nn.Parameter(W.clone(), requires_grad=trainable)
        self.register_buffer("col_norm", W.norm(dim=0).clamp_min(1e-8))
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        with torch.no_grad():
            T0 = (torch.as_tensor(X_train_ref, dtype=torch.float32) - self.x_mean) @ W
            self.register_buffer("t_mu", T0.mean(0))
            self.register_buffer("t_sd", T0.std(0).clamp_min(1e-6))
        self.in_features, self.out_features = W.shape

    def weight(self):
        return self.V / self.V.norm(dim=0, keepdim=True).clamp_min(1e-8) * self.col_norm

    def forward(self, x):
        dev = x.device
        w = self.weight().to(dev)
        return ((x - self.x_mean.to(dev)) @ w - self.t_mu.to(dev)) / self.t_sd.to(dev)

    @torch.no_grad()
    def backproject(self, comp_scores: torch.Tensor) -> torch.Tensor:
        """Map k component relevances (sum 1) to p feature relevances (sum 1)."""
        A = self.weight().abs() / self.t_sd                 # effective loading
        A = A / (A.sum(0, keepdim=True) + 1e-12)            # each column sums to 1
        s = A @ comp_scores
        return s / (s.sum() + 1e-12)

    @torch.no_grad()
    def transform_numpy(self, X, device="cpu"):
        return self.forward(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
