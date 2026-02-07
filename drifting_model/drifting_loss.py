"""Drifting loss computation for ImageNet training.

Implements:
- compute_V: Algorithm 2 from the paper (batch-normalized kernel drift)
- Feature normalization: normalize features so average pairwise distance = sqrt(C)
- Drift normalization: normalize drift so E[||V||^2 / C] ≈ 1
- Multi-temperature aggregation
- CFG weighting for unconditional samples
"""

import torch
import torch.nn.functional as F


def compute_V(x, y_pos, y_neg, temperature, cfg_weights=None, mask_self=True):
    """Compute the drifting field V (Algorithm 2 from the paper).

    Args:
        x: [N, D] generated samples (these are the points we compute V for)
        y_pos: [N_pos, D] positive (real) samples
        y_neg: [N_neg, D] negative (generated) samples.
               First N entries of y_neg correspond to x (self-pairs to mask).
        temperature: scalar temperature for the kernel
        cfg_weights: [N_neg] optional per-sample weights for CFG.
                     Unconditional samples get weight w, conditional get weight 1.
        mask_self: if True, mask the first N entries of y_neg as self-pairs

    Returns:
        V: [N, D] drift vectors
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    # Pairwise distances
    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]

    # Mask self-distances (x[i] == y_neg[i] for i < N)
    if mask_self and N <= N_neg:
        mask = torch.zeros(N, N_neg, device=x.device, dtype=x.dtype)
        mask[:, :N].fill_diagonal_(1e6)
        dist_neg = dist_neg + mask

    # Logits
    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature

    # Apply CFG weights to logits (weight unconditional samples)
    if cfg_weights is not None:
        logit_neg = logit_neg + cfg_weights.unsqueeze(0).log()

    # Concatenate for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

    # Normalize along both dimensions (double normalization)
    A_row = logit.softmax(dim=-1)     # normalize across y
    A_col = logit.softmax(dim=-2)     # normalize across x
    A = (A_row * A_col).sqrt()

    # Split back
    A_pos = A[:, :N_pos]   # [N, N_pos]
    A_neg = A[:, N_pos:]   # [N, N_neg]

    # Compute weights
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]

    # Compute drift
    drift_pos = W_pos @ y_pos  # [N, D]
    drift_neg = W_neg @ y_neg  # [N, D]

    V = drift_pos - drift_neg
    return V


def _feature_normalize(feat_gen, feat_all, eps=1e-8):
    """Feature normalization: scale features so average pairwise distance = sqrt(C).

    Args:
        feat_gen: [N, D] generated sample features
        feat_all: [M, D] all sample features (gen + pos + uncond) for distance computation

    Returns:
        S: scalar normalization scale (with stop-gradient)
    """
    C = feat_gen.shape[-1]

    # Compute pairwise distances between gen and all
    dists = torch.cdist(feat_gen, feat_all)  # [N, M]
    mean_dist = dists.mean()

    # S = (1/sqrt(C)) * mean_dist
    S = mean_dist / (C ** 0.5)
    S = S.detach().clamp(min=eps)  # stop-gradient
    return S


def _drift_normalize(V, eps=1e-8):
    """Drift normalization: scale so E[||V||^2 / C] ≈ 1.

    Args:
        V: [N, D] drift vectors

    Returns:
        lambda_j: scalar normalization scale
    """
    C = V.shape[-1]
    # lambda = sqrt(E[||V||^2 / C])
    lam = (V.pow(2).sum(dim=-1).mean() / C).sqrt()
    return lam.clamp(min=eps)


def _compute_single_feature_loss(feat_gen, feat_pos, feat_neg, feat_unc,
                                  temperatures, cfg_weights):
    """Compute drifting loss for a single feature group.

    Per-location features are handled by iterating over spatial locations.
    Normalization is shared across all locations within the same feature map.

    Args:
        feat_gen: [N_neg, num_vecs, C] generated features
        feat_pos: [N_pos, num_vecs, C] positive features
        feat_neg: [N_neg, num_vecs, C] negative features (= gen)
        feat_unc: [N_unc, num_vecs, C] unconditional features
        temperatures: list of temperature values
        cfg_weights: [N_unc] weights for unconditional samples

    Returns:
        loss: scalar
    """
    N_neg, num_vecs, C = feat_gen.shape
    N_pos = feat_pos.shape[0]
    N_unc = feat_unc.shape[0]

    # Shared feature normalization across all spatial locations
    # Concatenate all locations for distance computation
    gen_flat = feat_gen.reshape(N_neg * num_vecs, C)
    pos_flat = feat_pos.reshape(N_pos * num_vecs, C)
    unc_flat = feat_unc.reshape(N_unc * num_vecs, C)
    all_flat = torch.cat([gen_flat, pos_flat, unc_flat], dim=0)
    S = _feature_normalize(gen_flat, all_flat)

    # Build per-sample cfg weights (1 for gen, w for uncond)
    if cfg_weights is not None:
        combined_w = torch.cat([
            torch.ones(N_neg, device=feat_gen.device),
            cfg_weights,
        ])
    else:
        combined_w = None

    # Per-location drift computation with shared normalization and drift norm
    total_loss = torch.tensor(0.0, device=feat_gen.device)

    for loc in range(num_vecs):
        g = feat_gen[:, loc, :] / S  # [N_neg, C]
        p = feat_pos[:, loc, :] / S  # [N_pos, C]
        n = feat_neg[:, loc, :] / S  # [N_neg, C]
        u = feat_unc[:, loc, :] / S  # [N_unc, C]

        y_neg = torch.cat([n, u], dim=0)  # [N_neg + N_unc, C]

        V_agg = torch.zeros_like(g)
        for tau in temperatures:
            tau_eff = tau * (C ** 0.5)
            V_tau = compute_V(g, p, y_neg, tau_eff, combined_w)
            lam = _drift_normalize(V_tau)
            V_agg = V_agg + V_tau / lam

        target = (g + V_agg).detach()
        total_loss = total_loss + F.mse_loss(g, target)

    return total_loss


def compute_drifting_loss(
    gen_features,
    pos_features,
    neg_features,
    uncond_features,
    temperatures,
    cfg_weights,
    gen_latent=None,
    pos_latent=None,
    neg_latent=None,
    uncond_latent=None,
):
    """Compute the full drifting loss across all multi-scale features.

    Each feature group is a tensor [B, num_vecs, C] where num_vecs is the
    number of spatial locations (or 1 for global features). The normalization
    scale is shared across locations within the same feature map, but
    compute_V is called per location.

    Args:
        gen_features: list of [N_neg, num_vecs, C] tensors for generated samples
        pos_features: list of [N_pos, num_vecs, C] tensors for positive samples
        neg_features: list of [N_neg, num_vecs, C] tensors for negative samples (= gen)
        uncond_features: list of [N_unc, num_vecs, C] tensors for unconditional samples
        temperatures: list of temperature values [0.02, 0.05, 0.2]
        cfg_weights: [N_unc] per-sample weights for CFG
        gen_latent: [N_neg, latent_dim] raw latent for vanilla drifting loss (optional)
        pos_latent: [N_pos, latent_dim] raw latent (optional)
        neg_latent: [N_neg, latent_dim] raw latent (= gen_latent) (optional)
        uncond_latent: [N_unc, latent_dim] raw latent (optional)

    Returns:
        total_loss: scalar loss
    """
    total_loss = torch.tensor(0.0, device=gen_features[0].device)

    for j in range(len(gen_features)):
        loss_j = _compute_single_feature_loss(
            gen_features[j], pos_features[j],
            neg_features[j], uncond_features[j],
            temperatures, cfg_weights,
        )
        total_loss = total_loss + loss_j

    # Vanilla drifting loss on raw latent (without feature encoder)
    if gen_latent is not None and pos_latent is not None:
        N_neg_lat = gen_latent.shape[0]
        latent_dim = gen_latent.shape[-1]

        # Treat the raw latent as a single "feature" with 1 vector of dim latent_dim
        gen_lat_2d = gen_latent.reshape(N_neg_lat, -1).unsqueeze(1)     # [N, 1, D]
        pos_lat_2d = pos_latent.reshape(pos_latent.shape[0], -1).unsqueeze(1)
        neg_lat_2d = neg_latent.reshape(neg_latent.shape[0], -1).unsqueeze(1)

        if uncond_latent is not None:
            unc_lat_2d = uncond_latent.reshape(uncond_latent.shape[0], -1).unsqueeze(1)
        else:
            unc_lat_2d = torch.zeros(0, 1, gen_lat_2d.shape[-1], device=gen_latent.device)

        loss_lat = _compute_single_feature_loss(
            gen_lat_2d, pos_lat_2d, neg_lat_2d, unc_lat_2d,
            temperatures, cfg_weights,
        )
        total_loss = total_loss + loss_lat

    return total_loss
