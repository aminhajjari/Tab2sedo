"""
kan_hybrid.py
=============
A self-contained KAN branch + CNN-KAN fusion head, extracted from the design in
Mondragon-Ruiz et al. (2026), "Interpretable CNN-KAN hybrid architectures for
tabular data with synthetic image encoding".

What the paper's repo actually contains: a vendored copy of pykan plus notebooks.
The reusable idea is the architecture, not the KAN code. That idea is:

    tabular x  --> KAN branch      --\
                                      >--> fusion --> Final KAN --> prediction
    image   I  --> CNN image encoder -/

    interpretability:
        KAN branch  -> feature_score (spline-based, native)
        CNN branch  -> Grad-CAM -> mapped back to features
        Final KAN   -> branch weights w_kan / w_cnn
        GFS_i = w_kan * s_kan_i + w_cnn * s_cnn_i

This file reimplements the KAN as a plain nn.Module (efficient B-spline
formulation) instead of pykan, because:
  * pykan is built around full-batch LBFGS and caches activations for plotting;
    it is slow and memory-hungry inside a mini-batch Adam loop over many datasets.
  * as a plain nn.Module it composes with whatever optimizer / AMP / DDP you
    already use.
It still exposes `feature_score()`, which is the only pykan feature the paper's
interpretability pipeline depends on.

Requires: torch only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  KAN
# --------------------------------------------------------------------------- #
class KANLayer(nn.Module):
    """
    One Kolmogorov-Arnold layer: every input->output edge is a learnable
    univariate function, parameterised as (base activation) + (B-spline).

        y_o = sum_i [ w_base[o,i] * silu(x_i) + sum_c w_spline[o,i,c] * B_c(x_i) ]

    Args:
        in_features, out_features: layer widths.
        grid_size: number of spline intervals (pykan's `grid`). 5 is a good default.
        spline_order: B-spline degree (pykan's `k`). 3 = cubic, as in the paper.
        grid_range: domain the splines cover. Keep your inputs inside it.
                    If you MinMax-scale features to [0, 1], (-1, 1) is safe.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        # [in_features, grid_size + 2*spline_order + 1] -- buffer, not a parameter
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        self.base_weight.data.mul_(scale_base)
        nn.init.normal_(self.spline_weight, std=scale_spline / (in_features ** 0.5))

        # filled by forward() when attribution is on; shape [out_features, in_features]
        self.edge_relevance: torch.Tensor | None = None

    # ---- B-spline basis ---------------------------------------------------- #
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in] -> bases: [B, in, grid_size + spline_order]"""
        grid = self.grid                     # [in, G + 2k + 1]
        x = x.unsqueeze(-1)                  # [B, in, 1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor, attribute: bool = False) -> torch.Tensor:
        """
        x: [B, in_features] -> [B, out_features]

        attribute=True additionally materialises the per-edge contribution tensor
        [B, out, in] and stores its across-batch std in `self.edge_relevance`.
        That costs B*out*in memory, so leave it False during training and turn it
        on only for a dedicated attribution pass.
        """
        base = self.base_activation(x)                       # [B, in]
        bases = self.b_splines(x)                            # [B, in, C]

        if not attribute:
            out = F.linear(base, self.base_weight)
            out = out + F.linear(
                bases.view(x.size(0), -1),
                self.spline_weight.view(self.out_features, -1),
            )
            return out

        edge = base.unsqueeze(1) * self.base_weight.unsqueeze(0)          # [B, out, in]
        edge = edge + torch.einsum("bic,oic->boi", bases, self.spline_weight)
        # how much this edge actually moves across the data = its relevance
        self.edge_relevance = edge.std(dim=0).detach()                    # [out, in]
        return edge.sum(dim=-1)

    def regularisation_loss(self, l1: float = 1.0, entropy: float = 1.0) -> torch.Tensor:
        """pykan-style sparsity penalty. Optional; pass lamb * this into your loss."""
        w = self.spline_weight.abs().mean(dim=-1)            # [out, in]
        l1_term = w.sum()
        p = w / (l1_term + 1e-8)
        entropy_term = -(p * (p + 1e-8).log()).sum()
        return l1 * l1_term + entropy * entropy_term


class KAN(nn.Module):
    """A stack of KANLayers. `width=[8, 16, 1]` mirrors pykan's `KAN(width=...)`."""

    def __init__(
        self,
        width: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            KANLayer(width[i], width[i + 1], grid_size, spline_order, grid_range=grid_range)
            for i in range(len(width) - 1)
        )

    def forward(self, x: torch.Tensor, attribute: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attribute=attribute)
        return x

    @torch.no_grad()
    def feature_score(self, x: torch.Tensor, normalise: bool = True) -> torch.Tensor:
        """
        Native KAN attribution, equivalent in spirit to pykan's `feature_score`.

        Runs one attribution forward pass on `x`, then propagates relevance from
        the output nodes back to the input nodes through the per-edge relevance
        matrices:  s_in = E^T @ s_out.

        Returns: [in_features] non-negative scores, summing to 1 if normalise.
        """
        was_training = self.training
        self.eval()
        self.forward(x, attribute=True)

        scores = torch.ones(
            self.layers[-1].out_features, device=x.device, dtype=x.dtype
        )
        for layer in reversed(self.layers):
            scores = layer.edge_relevance.t() @ scores       # [out] -> [in]

        if was_training:
            self.train()
        scores = scores.abs()
        return scores / (scores.sum() + 1e-12) if normalise else scores

    def regularisation_loss(self, l1: float = 1.0, entropy: float = 1.0) -> torch.Tensor:
        return sum(layer.regularisation_loss(l1, entropy) for layer in self.layers)


# --------------------------------------------------------------------------- #
#  Hybrid: your image encoder + KAN branch + fusion + Final KAN
# --------------------------------------------------------------------------- #
class HybridKAN(nn.Module):
    """
    Paper's CNN-KAN, with your own CNN plugged in.

    Args:
        image_encoder: any nn.Module mapping [B, C, H, W] -> [B, img_feat_dim].
                       Use your existing CNN with its classifier head removed.
                       The paper's encoder ends in LayerNorm + Sigmoid + Flatten,
                       which bounds the visual features so they don't swamp the
                       symbolic ones -- worth keeping if you write your own.
        img_feat_dim: width of that encoder's output. Pass None to infer it with
                      a dummy forward (then also pass img_shape).
        n_features:   number of raw tabular columns.
        n_outputs:    1 for regression / binary, n_classes for multi-class.
        fusion:       'direct' | 'bottleneck' | 'scaled' | 'gated'
                      (paper's Strategy 1 / 2 / 3, plus plain concatenation)

    Fusion exists because the flattened CNN output is typically 100-1000x wider
    than the KAN output, so plain concatenation lets the visual branch dominate
    the Final KAN. That is the "modality collapse" the paper's Section 4.2 is about.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        n_features: int,
        n_outputs: int = 1,
        img_feat_dim: int | None = None,
        img_shape: tuple[int, int, int] | None = None,
        kan_neurons: int = 8,
        grid_size: int = 5,
        spline_order: int = 3,
        fusion: str = "bottleneck",
        bottleneck_dim: int = 4,
        alpha: float = 0.1,
        gate_hidden: int = 32,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        assert fusion in {"direct", "bottleneck", "scaled", "gated"}
        self.fusion = fusion
        self.alpha = alpha
        self.image_encoder = image_encoder

        if img_feat_dim is None:
            assert img_shape is not None, "give img_feat_dim or img_shape"
            with torch.no_grad():
                img_feat_dim = image_encoder(torch.zeros(2, *img_shape)).shape[1]
        self.img_feat_dim = img_feat_dim

        # symbolic branch on the raw tabular features
        self.kan_branch = KAN([n_features, kan_neurons], grid_size, spline_order, grid_range)

        if fusion == "bottleneck":
            self.cnn_proj = nn.Linear(img_feat_dim, bottleneck_dim)
            cnn_dim = bottleneck_dim
        else:
            self.cnn_proj = nn.Identity()
            cnn_dim = img_feat_dim

        if fusion == "gated":
            self.gate_net = nn.Sequential(
                nn.Linear(cnn_dim + kan_neurons, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )

        self.kan_dim, self.cnn_dim = kan_neurons, cnn_dim
        # Final KAN sees [kan_part | cnn_part]; index 0..kan_dim-1 is symbolic
        self.final_kan = KAN(
            [kan_neurons + cnn_dim, n_outputs], grid_size, spline_order, grid_range
        )

    def fuse(self, x_tab: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        kan_out = self.kan_branch(x_tab)
        cnn_out = self.cnn_proj(self.image_encoder(x_img))

        if self.fusion == "scaled":
            cnn_out = cnn_out * self.alpha
        elif self.fusion == "gated":
            g = self.gate_net(torch.cat([kan_out, cnn_out], dim=1))   # [B, 1] in [0,1]
            kan_out, cnn_out = kan_out * (1 - g), cnn_out * g
        return torch.cat([kan_out, cnn_out], dim=1)

    def forward(self, x_tab: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        return self.final_kan(self.fuse(x_tab, x_img))

    # ---- interpretability -------------------------------------------------- #
    @torch.no_grad()
    def branch_weights(self, x_tab: torch.Tensor, x_img: torch.Tensor) -> tuple[float, float]:
        """
        Modality Dominance Ratio (paper Sec. 7.1): the share of the Final KAN's
        input relevance that comes from the symbolic vs. the visual block.
        Returns (w_kan, w_cnn), summing to 1.
        """
        s = self.final_kan.feature_score(self.fuse(x_tab, x_img), normalise=True)
        w_kan = s[: self.kan_dim].sum().item()
        return w_kan, 1.0 - w_kan

    @torch.no_grad()
    def kan_feature_score(self, x_tab: torch.Tensor) -> torch.Tensor:
        """Per-input-feature symbolic relevance, [n_features], sums to 1."""
        return self.kan_branch.feature_score(x_tab)


# --------------------------------------------------------------------------- #
#  Grad-CAM on the image branch, and the Global Feature Score
# --------------------------------------------------------------------------- #
def grad_cam(
    model: HybridKAN,
    conv_layer: nn.Module,
    x_tab: torch.Tensor,
    x_img: torch.Tensor,
    target_index: int | None = None,
) -> torch.Tensor:
    """
    Grad-CAM on `conv_layer` (usually the LAST conv of your image encoder --
    note the paper hooks the first, which is coarser but higher-resolution).
    Returns a [H, W] map in [0, 1], upsampled to the input image size.
    """
    acts, grads = {}, {}
    h1 = conv_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = conv_layer.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("v", go[0].detach())
    )
    try:
        model.zero_grad(set_to_none=True)
        out = model(x_tab, x_img)
        target = out.sum() if target_index is None else out[:, target_index].sum()
        target.backward()

        a, g = acts["v"], grads["v"]                      # [B, K, h, w]
        weights = g.mean(dim=(2, 3), keepdim=True)        # [B, K, 1, 1]
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x_img.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.mean(0).squeeze(0)                      # average over the batch
        cam = cam - cam.min()
        return cam / (cam.max() + 1e-12)
    finally:
        h1.remove()
        h2.remove()


def cam_to_feature_scores(
    cam: torch.Tensor,
    coords: dict[str, tuple[int, int]],
    feature_names: list[str],
    zoom: int = 1,
    normalise: bool = True,
) -> torch.Tensor:
    """
    Map a [H, W] saliency map back onto tabular features.

    `coords` maps feature name -> (row, col) in the *pre-zoom* layout, i.e. the
    positions TINTOlib / IGTD / REFINED assigned. Padding pixels (Ex1, Ex2, ...)
    must simply be absent from `coords`; excluding them is what the paper calls
    preserving semantic validity.
    """
    scores = torch.zeros(len(feature_names), device=cam.device, dtype=cam.dtype)
    for i, name in enumerate(feature_names):
        if name not in coords:
            continue
        r, c = coords[name]
        scores[i] = cam[r * zoom : (r + 1) * zoom, c * zoom : (c + 1) * zoom].mean()
    return scores / (scores.sum() + 1e-12) if normalise else scores


def global_feature_score(
    kan_scores: torch.Tensor,
    cnn_scores: torch.Tensor,
    w_kan: float,
    w_cnn: float,
) -> torch.Tensor:
    """
    GFS_i = w_kan * s_kan_i + w_cnn * s_cnn_i   (paper, Sec. 4.3.3)

    Both score vectors must already be normalised to sum to 1 over the same
    feature ordering, and w_kan + w_cnn must be 1. The result is then a proper
    convex combination, so it sums to 1 and is comparable across models.
    """
    assert kan_scores.shape == cnn_scores.shape, "feature orderings must match"
    return w_kan * kan_scores + w_cnn * cnn_scores


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N_FEAT, N_CLS = 32, 12, 3
    IMG = (3, 16, 16)

    encoder = nn.Sequential(
        nn.Conv2d(IMG[0], 16, 3, padding=1),
        nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.GroupNorm(1, 32), nn.Sigmoid(),
        nn.AdaptiveAvgPool2d(2), nn.Flatten(),
    )

    for mode in ("direct", "bottleneck", "scaled", "gated"):
        model = HybridKAN(encoder, N_FEAT, N_CLS, img_shape=IMG, fusion=mode)
        xt, xi = torch.rand(B, N_FEAT), torch.rand(B, *IMG)
        y = model(xt, xi)
        wk, wc = model.branch_weights(xt, xi)
        ks = model.kan_feature_score(xt)
        cam = grad_cam(model, encoder[4], xt, xi, target_index=0)
        cs = cam_to_feature_scores(
            cam,
            {f"f{i}": (i // 4, i % 4) for i in range(N_FEAT)},
            [f"f{i}" for i in range(N_FEAT)],
            zoom=4,
        )
        gfs = global_feature_score(ks, cs, wk, wc)
        print(
            f"{mode:<11} out={tuple(y.shape)}  w_kan={wk:.3f} w_cnn={wc:.3f}  "
            f"kan_sum={ks.sum():.3f}  gfs_sum={gfs.sum():.3f}  cam={tuple(cam.shape)}"
        )
