import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
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
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Standard attention (no RoPE for now, rely on PE)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim)

        # adaLN-zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight[-2 * dim:])
        nn.init.zeros_(self.adaLN_modulation[1].bias[-2 * dim:])

    def forward(self, x, cond):
        mod = self.adaLN_modulation(cond)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h = self.attn(h)
        x = x + alpha1.unsqueeze(1) * h

        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h

        return x

class DriftingPolicy(nn.Module):
    """1D DiT-based generator for robotics control."""
    
    def __init__(
        self,
        action_dim,
        obs_dim,
        horizon=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        
        # Action embedding
        self.input_proj = nn.Linear(action_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, embed_dim) * 0.02)
        
        # Observation conditioning projection
        self.obs_proj = nn.Linear(obs_dim, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # Output
        self.final_norm = RMSNorm(embed_dim)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        self.output_proj = nn.Linear(embed_dim, action_dim)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, noise, obs):
        """
        Args:
            noise: [B, T, action_dim] Gaussian noise
            obs: [B, obs_dim] Observation vector
            
        Returns:
            action: [B, T, action_dim]
        """
        x = self.input_proj(noise) + self.pos_embed
        
        cond = self.obs_proj(obs) # [B, embed_dim]
        
        for block in self.blocks:
            x = block(x, cond)
            
        mod = self.final_adaLN(cond)
        gamma, beta = mod.chunk(2, dim=-1)
        
        x = self.final_norm(x)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.output_proj(x)
        
        return x
