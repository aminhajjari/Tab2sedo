"""
self_explain.py  --  "X-Row": X-Node-style self-explanation for Tab2sedo
========================================================================

Adapts X-Node (Sengupta & Rekik, GRAIL@MICCAI 2025) to tabular rows and fuses it
with the KAN interpretability already in Tab2sedo.

    row x --PLS/SDP--> T --KAN branch------------------\
                        |                                >-- Final KAN --> y_hat
    noise --CVAE-------> generated image --CNN---------/|
                        |                                |
                        +--kNN graph over TRAIN rows     |
                              -> context c_i ---> KAN Reasoner -> e_i ---/
                                                    |-> decode c_hat (faithful to context)
                                                    |-> decode h_hat (faithful to model)

Per sample, one forward pass gives a *three-level* explanation:
  1. WHICH MODALITY decided   : local shares of tab / image / context in the Final KAN
  2. WHICH FEATURES           : per-sample KAN relevance, back-projected through PLS
  3. WHY (relational)         : which context cues (neighbour purity, outlierness, ...)
                                drove e_i, plus the nearest training rows as prototypes
The record can optionally be narrated by an LLM *after* training (never in the loop).

Differences from X-Node (deliberate fixes):
  * Reasoner is a KAN, not an MLP -> the explanation generator itself is inspectable.
  * X-Node's alignment term ||e - c||^2 forces d_e = d_c and makes e ~ c. Here e is
    low-dimensional and faithfulness to c is a reconstruction ||dec(e) - c||^2.
  * No own-label features (X-Node's 2-hop agreement uses the node's true label,
    which is unavailable at test time). Neighbours are TRAINING rows only, and
    training rows exclude themselves -> inductive, no label leakage.
  * No betweenness / eigenvector centrality / community ID: O(N*E) or categorical,
    and not meaningful on a kNN graph of i.i.d. rows. Replaced with cheap cues that are.

Requires: torch, numpy, scikit-learn, and kan_hybrid.py from this repo.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from kan_hybrid import KAN, HybridKAN


# --------------------------------------------------------------------------- #
#  1. Neighbourhood context vector c_i  (X-Node Sec. 3B, rewritten for rows)
# --------------------------------------------------------------------------- #
class NeighborhoodContext:
    """
    Builds a kNN graph over training rows in a (preferably PLS/SDP) latent space
    and turns each row's neighbourhood into an interpretable context vector.

    Use the *initial* PLS scores (projection before training) so the graph is fixed;
    VIF-decorrelated components make Euclidean distance meaningful, which ties
    this module to your SDP contribution.
    """

    SCALAR_NAMES = [
        "nbr_purity",        # similarity-weighted share of the majority class among k nbrs
        "nbr_entropy",       # normalised entropy of neighbour labels (0 = unanimous)
        "hop2_purity",       # share of 2-hop nbrs carrying the 1-hop majority label
        "mean_similarity",   # mean Gaussian similarity to neighbours (X-Node's avg edge weight)
        "lof_ratio",         # own k-distance / nbrs' k-distance (>1 = local outlier)
        "centroid_margin",   # (d_2nd - d_1st)/(d_2nd + d_1st) to class centroids
    ]

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, T_train: np.ndarray, y_train: np.ndarray, num_classes: int):
        self.T = np.asarray(T_train, dtype=np.float32)
        self.y = np.asarray(y_train, dtype=np.int64)
        self.C = num_classes
        k = min(self.k, len(self.T) - 1)
        self.k_eff = k
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(self.T)
        d, idx = self.nn.kneighbors(self.T)                  # col 0 is the row itself
        self.train_nbr = idx[:, 1:]
        self.train_kdist = d[:, 1:].mean(1) + 1e-8
        self.bandwidth = float(np.median(d[:, 1:])) + 1e-8
        self.centroids = np.stack([
            self.T[self.y == c].mean(0) if np.any(self.y == c) else np.zeros(self.T.shape[1])
            for c in range(self.C)
        ]).astype(np.float32)
        self.names = self.SCALAR_NAMES + [f"support_class_{c}" for c in range(self.C)]
        # standardisation of c (fit on train context) so values sit inside the KAN grid
        raw = self._raw(self.T, is_train=True)
        self.mu, self.sd = raw.mean(0), raw.std(0) + 1e-6
        return self

    def _raw(self, T, is_train):
        k = self.k_eff
        if is_train:
            d, idx = self.nn.kneighbors(T, n_neighbors=k + 1)
            d, idx = d[:, 1:], idx[:, 1:]                    # exclude self
        else:
            d, idx = self.nn.kneighbors(T, n_neighbors=k)
        w = np.exp(-(d / self.bandwidth) ** 2)               # [N, k]
        lab = self.y[idx]                                    # [N, k]

        support = np.zeros((len(T), self.C), dtype=np.float32)
        for c in range(self.C):
            support[:, c] = (w * (lab == c)).sum(1)
        support /= support.sum(1, keepdims=True) + 1e-12
        maj = support.argmax(1)
        purity = support.max(1)
        entropy = -(support * np.log(support + 1e-12)).sum(1) / np.log(max(self.C, 2))

        hop2 = self.train_nbr[idx].reshape(len(T), -1)       # [N, k*k]
        hop2_purity = (self.y[hop2] == maj[:, None]).mean(1)

        lof = (d.mean(1) + 1e-8) / self.train_kdist[idx].mean(1)

        dc = np.linalg.norm(T[:, None, :] - self.centroids[None], axis=2)  # [N, C]
        dcs = np.sort(dc, 1)
        margin = (dcs[:, 1] - dcs[:, 0]) / (dcs[:, 1] + dcs[:, 0] + 1e-8) if self.C > 1 \
            else np.zeros(len(T))

        scalars = np.stack([purity, entropy, hop2_purity, w.mean(1), lof, margin], 1)
        self._last_idx = idx
        return np.concatenate([scalars, support], 1).astype(np.float32)

    def transform(self, T, is_train=False, standardise=True):
        raw = self._raw(np.asarray(T, dtype=np.float32), is_train)
        return (raw - self.mu) / self.sd if standardise else raw

    def neighbours(self, T, n=3):
        """Nearest training rows (indices, labels) -> example-based explanation."""
        _, idx = self.nn.kneighbors(np.asarray(T, dtype=np.float32), n_neighbors=n)
        return idx, self.y[idx]


# --------------------------------------------------------------------------- #
#  2. KAN Reasoner  (X-Node Sec. 3C/3D, with the MLP replaced by a KAN)
# --------------------------------------------------------------------------- #
class KANReasoner(nn.Module):
    def __init__(self, d_ctx: int, d_expl: int, d_emb: int, grid_range=(-5.0, 5.0)):
        super().__init__()
        self.kan = KAN([d_ctx, d_expl], grid_range=grid_range)
        self.ctx_decoder = nn.Linear(d_expl, d_ctx)        # e -> c_hat
        self.emb_decoder = nn.Linear(d_expl, d_emb)        # e -> h_hat

    def forward(self, c):
        return 3.0 * torch.tanh(self.kan(c) / 3.0)         # soft-bounded, stays in the grid

    def faithfulness_loss(self, e, c, h, alpha=0.5, beta=0.5):
        l_ctx = F.mse_loss(self.ctx_decoder(e), c)
        l_emb = F.mse_loss(self.emb_decoder(e), h.detach())  # don't let h collapse onto e
        return alpha * l_ctx + beta * l_emb


# --------------------------------------------------------------------------- #
#  3. HybridKAN with a third (context) block in the Final KAN  (X-Node Sec. 3F)
# --------------------------------------------------------------------------- #
class XHybridKAN(HybridKAN):
    """Final KAN input = [kan_out | cnn_out | e]. branch_weights -> 3 shares."""

    def __init__(self, *args, ctx_dim: int, **kw):
        super().__init__(*args, **kw)
        self.ctx_dim = ctx_dim
        grid_range = tuple(self.final_kan.layers[0].grid[0, [3, -4]].tolist())
        n_out = self.final_kan.layers[-1].out_features
        self.final_kan = KAN([self.kan_dim + self.cnn_dim + ctx_dim, n_out],
                             grid_range=grid_range)

    def fuse(self, x_tab, x_img, e):
        return torch.cat([super().fuse(x_tab, x_img), e], dim=1)

    def forward(self, x_tab, x_img, e):
        return self.final_kan(self.fuse(x_tab, x_img, e))


# --------------------------------------------------------------------------- #
#  4. Per-sample (local) KAN relevance -- your feature_score() is global
# --------------------------------------------------------------------------- #
def _edges(layer, x):
    """Exact per-sample edge contributions [B, out, in]; sum over in = layer output."""
    base = layer.base_activation(x)
    edge = base.unsqueeze(1) * layer.base_weight.unsqueeze(0)
    return edge + torch.einsum("bic,oic->boi", layer.b_splines(x), layer.spline_weight)


@torch.no_grad()
def kan_local_relevance(kan: KAN, x: torch.Tensor, r_out: torch.Tensor, eps=1e-6):
    """
    LRP-epsilon through a KAN. r_out: [B, out] relevance at the KAN output.
    First layer is exact (KAN is additive over inputs); deeper layers approximate.
    Returns signed [B, in]; sum over inputs ~= sum of r_out.
    """
    acts = [x]
    for layer in kan.layers[:-1]:
        acts.append(layer(acts[-1]))
    r = r_out
    for layer, a in zip(reversed(kan.layers), reversed(acts)):
        E = _edges(layer, a)                                 # [B, out, in]
        z = E.sum(-1, keepdim=True)
        z = z + eps * torch.where(z >= 0, 1.0, -1.0)
        r = (E / z * r.unsqueeze(-1)).sum(1)                 # [B, in]
    return r


@torch.no_grad()
def projection_local_relevance(proj, x, r_comp, eps=1e-6):
    """Signed per-sample back-projection of component relevance [B,k] -> features [B,p]."""
    W = proj.weight()                                        # [p, k]
    Z = (x - proj.x_mean).unsqueeze(-1) * (W / proj.t_sd).unsqueeze(0)   # [B, p, k]
    z = Z.sum(1, keepdim=True)
    z = z + eps * torch.where(z >= 0, 1.0, -1.0)
    return (Z / z * r_comp.unsqueeze(1)).sum(-1)


@torch.no_grad()
def explain_batch(model, x_raw, x_img, c, target=None):
    """
    model: CAE with .proj (or None), .hybrid (XHybridKAN), .reasoner (KANReasoner).
    Returns a dict of per-sample explanations for the fused prediction.
    """
    tab = model.proj(x_raw) if model.proj is not None else x_raw
    e = model.reasoner(c)
    h = model.hybrid
    fused_in = h.fuse(tab, x_img, e)
    logits = h.final_kan(fused_in)
    pred = logits.argmax(1) if target is None else target
    r_out = F.one_hot(pred, logits.shape[1]).float() * logits      # relevance = target logit

    r_in = kan_local_relevance(h.final_kan, fused_in, r_out)       # [B, kan+cnn+ctx]
    a, b = h.kan_dim, h.kan_dim + h.cnn_dim
    blocks = torch.stack([r_in[:, :a].sum(1), r_in[:, a:b].sum(1), r_in[:, b:].sum(1)], 1)
    share = blocks.abs() / (blocks.abs().sum(1, keepdim=True) + 1e-12)   # local MDR

    r_feat = kan_local_relevance(h.kan_branch, tab, r_in[:, :a])   # [B, k]
    if model.proj is not None:
        r_feat = projection_local_relevance(model.proj, x_raw, r_feat)   # [B, p]
    r_ctx = kan_local_relevance(model.reasoner.kan, c, r_in[:, b:])      # [B, d_ctx]

    return dict(pred=pred, prob=F.softmax(logits, 1).gather(1, pred[:, None]).squeeze(1),
                modality_share=share, feature_relevance=r_feat, context_relevance=r_ctx)


# --------------------------------------------------------------------------- #
#  5. Optional, post-hoc LLM narration (X-Node Sec. 3E), grounded and checkable
# --------------------------------------------------------------------------- #
def build_prompt(i, expl, ctx_raw, ctx_names, feat_names, x_orig, class_names,
                 nbr_labels=None, top=5):
    p = int(expl["pred"][i]); share = expl["modality_share"][i].tolist()
    rf = expl["feature_relevance"][i]; rc = expl["context_relevance"][i]
    tf = torch.argsort(rf.abs(), descending=True)[:top].tolist()
    tc = torch.argsort(rc.abs(), descending=True)[:3].tolist()
    lines = [f"- {feat_names[j]} = {x_orig[j]:.3g} (relevance {rf[j]:+.3f})" for j in tf]
    ctxl = [f"- {ctx_names[j]} = {ctx_raw[j]:.3g} (relevance {rc[j]:+.3f})" for j in tc]
    nb = "" if nbr_labels is None else \
        f"Nearest training rows have labels: {[class_names[int(l)] for l in nbr_labels]}\n"
    return (
        "You explain one prediction of a tabular classifier. Use ONLY the numbers below; "
        "do not invent features or values. Say which evidence supports and which opposes the "
        "prediction, and flag low confidence or outlier status.\n"
        f"Prediction: {class_names[p]} (p={float(expl['prob'][i]):.2f})\n"
        f"Decision share: features {share[0]:.0%}, generated image {share[1]:.0%}, "
        f"neighbourhood context {share[2]:.0%}\n"
        "Top features:\n" + "\n".join(lines) + "\n"
        "Top context cues:\n" + "\n".join(ctxl) + "\n" + nb
    )
