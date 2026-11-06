import torch
import torch.nn as nn
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def precompute_rope_freqs_2d(dim: int, seq_len: int, theta: float = 10000.0):
    # Split the feature dimension between height and width coordinates.
    dim = dim // 2
    T = int(seq_len ** 0.5)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(T, dtype=torch.float32)
    freqs_h = torch.einsum('i,j->ij', positions, freqs)
    freqs_w = torch.einsum('i,j->ij', positions, freqs)
    angles = torch.cat([
        freqs_h[:, None, :].repeat(1, T, 1),
        freqs_w[None, :, :].repeat(T, 1, 1),
    ], dim=-1)
    return torch.cos(angles).reshape(seq_len, dim), torch.sin(angles).reshape(seq_len, dim)


def apply_rotary_pos_emb(x, rope_cos_sin):
    rope_cos, rope_sin = rope_cos_sin
    x_float = x.to(torch.float32)
    x_reshaped = x_float.reshape(x_float.shape[:-1] + (-1, 2))
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
    cos = rope_cos.to(x.device).unsqueeze(0).unsqueeze(2)
    sin = rope_sin.to(x.device).unsqueeze(0).unsqueeze(2)
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.reshape(x.shape).to(x.dtype)


def apply_rotary_pos_emb_partial(x, rope_cos_sin):
    # Rotate only the trailing patch tokens and leave the prefix unchanged.
    rope_cos, rope_sin = rope_cos_sin
    x_float = x.to(torch.float32)
    x_reshaped = x_float.reshape(x_float.shape[:-1] + (-1, 2))
    num_patches = rope_cos.shape[0]
    cos = rope_cos.to(x.device).unsqueeze(0).unsqueeze(2)
    sin = rope_sin.to(x.device).unsqueeze(0).unsqueeze(2)

    prefix = x_reshaped[:, :-num_patches]
    patches = x_reshaped[:, -num_patches:]
    x1, x2 = patches[..., 0], patches[..., 1]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([prefix, rotated], dim=1).reshape(x.shape).to(x.dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
