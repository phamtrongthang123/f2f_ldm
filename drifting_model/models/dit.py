"""DiT-style generator for drifting models.

Architecture per Appendix B.2 of the paper:
- Patchify + linear projection
- 2D RoPE on Q,K
- adaLN-zero conditioning (class label + CFG alpha + style)
- RMSNorm, SwiGLU FFN, QK-Norm
- In-context conditioning tokens + register tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def _sinusoidal_embedding(values, dim):
    """Sinusoidal positional embedding for scalar values.

    Args:
        values: [B] scalar values (e.g. CFG alpha)
        dim: embedding dimension

    Returns:
        [B, dim] embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=values.device, dtype=values.dtype) / half)
    args = values[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def _build_rope_cache(seq_h, seq_w, dim, device, dtype=torch.float32):
    """Build 2D RoPE frequency cache.

    Returns cos, sin each of shape [seq_h*seq_w, dim//2].
    We split dim equally for h and w axes.
    """
    assert dim % 4 == 0, "RoPE dim must be divisible by 4 for 2D"
    half = dim // 4  # per-axis dim

    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=dtype) / half))

    h_pos = torch.arange(seq_h, device=device, dtype=dtype)
    w_pos = torch.arange(seq_w, device=device, dtype=dtype)

    h_freqs = torch.outer(h_pos, freqs)  # [H, half]
    w_freqs = torch.outer(w_pos, freqs)  # [W, half]

    # Expand to grid
    h_freqs = h_freqs[:, None, :].expand(-1, seq_w, -1).reshape(-1, half)  # [H*W, half]
    w_freqs = w_freqs[None, :, :].expand(seq_h, -1, -1).reshape(-1, half)  # [H*W, half]

    freqs_all = torch.cat([h_freqs, w_freqs], dim=-1)  # [H*W, dim//2]
    return freqs_all.cos(), freqs_all.sin()


def _apply_rope(x, cos, sin):
    """Apply rotary embedding to x.

    x: [B, heads, seq, head_dim]
    cos, sin: [seq, head_dim//2]
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:x.shape[2], :d2].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :d2].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # QK-Norm
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, N, heads, head_dim]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE only to the patch tokens (last tokens in sequence)
        # The first tokens are register/context tokens without RoPE
        n_prefix = N - rope_cos.shape[0]
        if n_prefix > 0:
            q_prefix, q_patch = q[:, :, :n_prefix], q[:, :, n_prefix:]
            k_prefix, k_patch = k[:, :, :n_prefix], k[:, :, n_prefix:]
            q_patch = _apply_rope(q_patch, rope_cos, rope_sin)
            k_patch = _apply_rope(k_patch, rope_cos, rope_sin)
            q = torch.cat([q_prefix, q_patch], dim=2)
            k = torch.cat([k_prefix, k_patch], dim=2)
        else:
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class DiTBlock(nn.Module):
    """DiT transformer block with adaLN-zero modulation."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim)

        # adaLN-zero: produces (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Initialize alpha (gate) projections to zero for residual init
        nn.init.zeros_(self.adaLN_modulation[1].weight[-2 * dim:])
        nn.init.zeros_(self.adaLN_modulation[1].bias[-2 * dim:])

    def forward(self, x, cond, rope_cos, rope_sin):
        mod = self.adaLN_modulation(cond)  # [B, 6*dim]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Attention branch
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h = self.attn(h, rope_cos, rope_sin)
        x = x + alpha1.unsqueeze(1) * h

        # FFN branch
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h

        return x


# ---------------------------------------------------------------------------
# DiT Generator
# ---------------------------------------------------------------------------

class DiTGenerator(nn.Module):
    """DiT-based generator for drifting models.

    Maps (noise, class_label, cfg_alpha, style_indices) → latent.
    """

    def __init__(
        self,
        input_size=32,
        input_channels=4,
        patch_size=2,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
        num_register_tokens=16,
        num_style_tokens=32,
        style_codebook_size=64,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (input_size // patch_size) ** 2
        self.grid_size = input_size // patch_size
        self.num_register_tokens = num_register_tokens
        self.num_style_tokens = num_style_tokens
        self.num_context_tokens = 16  # in-context conditioning tokens

        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * patch_size * input_channels, hidden_dim)

        # Class embedding
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

        # CFG alpha embedding
        self.alpha_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Style codebook
        self.style_codebook = nn.Embedding(style_codebook_size, hidden_dim)

        # In-context conditioning tokens
        self.context_pos_embed = nn.Parameter(torch.randn(1, self.num_context_tokens, hidden_dim) * 0.02)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        # Register tokens
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])

        # Output
        self.final_norm = RMSNorm(hidden_dim)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim, patch_size * patch_size * input_channels)

        # Build RoPE cache
        self._rope_cos = None
        self._rope_sin = None

        self._init_weights()

    def _init_weights(self):
        # Initialize patch embed and output proj
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _get_rope(self, device, dtype):
        if self._rope_cos is None or self._rope_cos.device != device:
            head_dim = self.hidden_dim // (self.hidden_dim // (self.hidden_dim // 12))
            # num_heads = hidden_dim // head_dim, head_dim = hidden_dim // num_heads
            # For simplicity, use hidden_dim // num_heads where num_heads = blocks[0].attn.num_heads
            head_dim = self.blocks[0].attn.head_dim
            self._rope_cos, self._rope_sin = _build_rope_cache(
                self.grid_size, self.grid_size, head_dim, device, dtype
            )
        return self._rope_cos, self._rope_sin

    def patchify(self, x):
        """x: [B, C, H, W] → [B, num_patches, patch_dim]"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.num_patches, p * p * C)
        return x

    def unpatchify(self, x):
        """x: [B, num_patches, patch_dim] → [B, C, H, W]"""
        B = x.shape[0]
        p = self.patch_size
        g = self.grid_size
        C = self.input_channels
        x = x.reshape(B, g, g, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)
        return x

    def forward(self, noise, class_labels, cfg_alpha, style_indices=None):
        """
        Args:
            noise: [B, C, H, W] Gaussian noise (input_channels, input_size, input_size)
            class_labels: [B] integer class labels
            cfg_alpha: [B] CFG strength values
            style_indices: [B, num_style_tokens] random indices into codebook, or None

        Returns:
            latent: [B, C, H, W] generated latent
        """
        B = noise.shape[0]
        device = noise.device
        dtype = noise.dtype

        # Build conditioning vector
        cond = self.class_embed(class_labels)  # [B, D]
        cond = cond + self.alpha_embed(_sinusoidal_embedding(cfg_alpha, self.hidden_dim))  # [B, D]

        if style_indices is not None:
            style = self.style_codebook(style_indices)  # [B, num_style_tokens, D]
            cond = cond + style.sum(dim=1)  # [B, D]

        # Patchify noise
        tokens = self.patch_embed(self.patchify(noise))  # [B, num_patches, D]

        # Build in-context tokens: projected cond + positional embeddings
        ctx = self.context_proj(cond).unsqueeze(1) + self.context_pos_embed  # [B, 16, D]

        # Register tokens
        reg = self.register_tokens.expand(B, -1, -1)  # [B, 16, D]

        # Prepend context and register tokens
        tokens = torch.cat([ctx, reg, tokens], dim=1)  # [B, 16+16+256, D]

        # Get RoPE for patch tokens only
        rope_cos, rope_sin = self._get_rope(device, dtype)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, cond, rope_cos, rope_sin)

        # Extract patch tokens
        patch_tokens = tokens[:, self.num_context_tokens + self.num_register_tokens:]

        # Final output
        mod = self.final_adaLN(cond)
        gamma, beta = mod.chunk(2, dim=-1)
        patch_tokens = self.final_norm(patch_tokens)
        patch_tokens = patch_tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        patch_tokens = self.output_proj(patch_tokens)

        return self.unpatchify(patch_tokens)


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------

def dit_b2(**kwargs):
    return DiTGenerator(
        hidden_dim=768, depth=12, num_heads=12, patch_size=2, **kwargs
    )


def dit_l2(**kwargs):
    return DiTGenerator(
        hidden_dim=1024, depth=24, num_heads=16, patch_size=2, **kwargs
    )
