"""Drifting loss computation for ImageNet training.

Implements:
- compute_V: Algorithm 2 from the paper (batch-normalized kernel drift)
- Feature normalization: normalize features so average pairwise distance = sqrt(C)
- Drift normalization: normalize drift so E[||V||^2 / C] ≈ 1
- Multi-temperature aggregation
- CFG weighting for unconditional samples

Vectorized: compute_V operates on a batched location dimension [L, N, D]
using torch.cdist's batch support, eliminating the per-location Python loop.
"""

import torch
import torch.nn.functional as F


def compute_V(x, y_pos, y_neg, temperature, cfg_weights=None, mask_self=True):
    """Compute the drifting field V (Algorithm 2 from the paper).

    Supports both unbatched [N, D] and batched [L, N, D] inputs.
    When batched, L is the number of spatial locations processed in parallel.

    Args:
        x: [N, D] or [L, N, D] generated samples
        y_pos: [N_pos, D] or [L, N_pos, D] positive (real) samples
        y_neg: [N_neg, D] or [L, N_neg, D] negative (generated) samples.
               First N entries of y_neg correspond to x (self-pairs to mask).
        temperature: scalar temperature for the kernel
        cfg_weights: [N_neg] optional per-sample weights for CFG.
                     Unconditional samples get weight w, conditional get weight 1.
        mask_self: if True, mask the first N entries of y_neg as self-pairs

    Returns:
        V: [N, D] or [L, N, D] drift vectors
    """
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
        y_pos = y_pos.unsqueeze(0)
        y_neg = y_neg.unsqueeze(0)

    L, N, D = x.shape
    N_pos = y_pos.shape[1]
    N_neg = y_neg.shape[1]

    # Pairwise distances — cdist supports batch dim [L, N, M]
    dist_pos = torch.cdist(x, y_pos)  # [L, N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [L, N, N_neg]

    # Mask self-distances (x[i] == y_neg[i] for i < N)
    if mask_self and N <= N_neg:
        # Build [N, N_neg] mask once, broadcast over L
        mask = torch.zeros(N, N_neg, device=x.device, dtype=x.dtype)
        mask[:, :N].fill_diagonal_(1e6)
        dist_neg = dist_neg + mask.unsqueeze(0)  # [1, N, N_neg] broadcast

    # Logits
    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature

    # Apply CFG weights to logits (weight unconditional samples)
    if cfg_weights is not None:
        logit_neg = logit_neg + cfg_weights.log().unsqueeze(0).unsqueeze(0)  # [1, 1, N_neg]

    # Concatenate for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=2)  # [L, N, N_pos + N_neg]

    # Normalize along both dimensions (double normalization)
    A_row = logit.softmax(dim=-1)     # normalize across y
    A_col = logit.softmax(dim=-2)     # normalize across x
    A = (A_row * A_col).sqrt()

    # Split back
    A_pos = A[:, :, :N_pos]   # [L, N, N_pos]
    A_neg = A[:, :, N_pos:]   # [L, N, N_neg]

    # Compute weights
    W_pos = A_pos * A_neg.sum(dim=2, keepdim=True)  # [L, N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=2, keepdim=True)  # [L, N, N_neg]

    # Compute drift: [L, N, N_pos] @ [L, N_pos, D] → [L, N, D]
    drift_pos = torch.bmm(W_pos, y_pos)
    drift_neg = torch.bmm(W_neg, y_neg)

    V = drift_pos - drift_neg  # [L, N, D]

    if not batched:
        V = V.squeeze(0)
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

    Normalization is shared across all spatial locations within the same
    feature map (per appendix_impl.tex:264-267).

    Args:
        V: [L, N, D] drift vectors (L locations, N samples, D dims)

    Returns:
        lambda_j: scalar normalization scale (shared across all L locations)
    """
    C = V.shape[-1]
    # Flatten locations and samples: compute E over all L*N vectors
    # lambda = sqrt(E[||V||^2 / C])
    lam = (V.pow(2).sum(dim=-1).mean() / C).sqrt()
    return lam.detach().clamp(min=eps)


def _compute_single_feature_loss(feat_gen, feat_pos, feat_neg, feat_unc,
                                  temperatures, cfg_weights):
    """Compute drifting loss for a single feature group.

    Vectorized: all spatial locations are processed in a single batched call
    to compute_V, with normalization shared across locations.

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

    # Reshape to [L, N, C] for batched compute_V (L = num_vecs)
    # Transpose from [N, L, C] to [L, N, C]
    g = feat_gen.permute(1, 0, 2) / S  # [L, N_neg, C]
    p = feat_pos.permute(1, 0, 2) / S  # [L, N_pos, C]
    n = feat_neg.permute(1, 0, 2) / S  # [L, N_neg, C]
    u = feat_unc.permute(1, 0, 2) / S  # [L, N_unc, C]

    # Concatenate negatives: [L, N_neg + N_unc, C]
    y_neg = torch.cat([n, u], dim=1)

    # Batched drift computation — one call per temperature
    V_agg = torch.zeros_like(g)  # [L, N_neg, C]
    for tau in temperatures:
        tau_eff = tau * (C ** 0.5)
        V_tau = compute_V(g, p, y_neg, tau_eff, combined_w)  # [L, N_neg, C]
        # Drift normalization shared across all L locations (fixes per-location bug)
        lam = _drift_normalize(V_tau)
        V_agg = V_agg + V_tau / lam

    # MSE loss with stop-gradient target
    target = (g + V_agg).detach()  # [L, N_neg, C]
    loss = F.mse_loss(g, target) * num_vecs  # multiply by L to match original sum

    return loss


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
    scale is shared across locations within the same feature map.

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
    total_loss = torch.tensor(0.0)
    if len(gen_features) > 0:
        total_loss = total_loss.to(gen_features[0].device)
    elif gen_latent is not None:
        total_loss = total_loss.to(gen_latent.device)
    else:
        # Fallback (should not happen in valid usage)
        total_loss = total_loss.to(torch.device("cpu"))

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
