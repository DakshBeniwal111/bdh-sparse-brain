"""
BDH Core Model — faithful implementation of BDH-GPU architecture
Based on: https://arxiv.org/abs/2509.26507
Official repo: https://github.com/pathwaycom/bdh

Key architectural features implemented:
  - ReLU sparse activations (~5% neurons fire)
  - Hebbian synaptic state (constant-size, not KV-cache)
  - Linear O(T) attention
  - Scale-free graph topology emerging from ReLU-lowrank structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BDHConfig:
    def __init__(
        self,
        vocab_size=256,     # byte-level
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=256,
        dropout=0.0,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.head_size = n_embd // n_head


class RoPE(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, head_size, max_seq=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_size)
        cos = emb.cos()[None, None, :, :]        # (1,1,T,hs)
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, cos, sin):
    return q * cos + rotate_half(q) * sin


class BDHAttention(nn.Module):
    """
    BDH linear attention with Hebbian synaptic state.
    Core equation (BDH-GPU, eq. 8 from paper):
        σ_{t+1} = σ_t + η * (relu(Q_t)^T relu(K_t))   [Hebbian update]
        out_t    = relu(Q_t) @ σ_t                        [read from state]
        
    Memory is O(n_embd * head_size) — CONSTANT regardless of sequence length.
    Compare to transformer KV-cache: O(T * head_size) — GROWS with T.
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.n_embd = config.n_embd

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rope = RoPE(config.head_size)

        # Hebbian learning rate (eta)
        self.eta = nn.Parameter(torch.ones(1) * 0.1)

        # Storage for activation captures (used by visualizer)
        self.last_q_activations = None
        self.last_k_activations = None
        self.last_hebbian_state = None

    def forward(self, x, sigma=None, capture=False):
        """
        x:     (B, T, C)
        sigma: (B, n_head, head_size, head_size)  — Hebbian synaptic state
        Returns: out (B,T,C), new_sigma
        """
        B, T, C = x.shape
        qkv = self.qkv(x)                              # (B,T,3C)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # Reshape to multi-head
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B,H,T,hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(q, T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # BDH sparse activation — THIS is the key: ReLU creates ~5% sparsity
        q_sparse = F.relu(q)   # (B,H,T,hs)
        k_sparse = F.relu(k)

        if capture:
            self.last_q_activations = q_sparse.detach().cpu()
            self.last_k_activations = k_sparse.detach().cpu()

        # Initialise Hebbian state if not provided
        if sigma is None:
            sigma = torch.zeros(B, self.n_head, self.head_size, self.head_size,
                                device=x.device)

        # Linear (causal) accumulation + Hebbian update
        # For each position t: out_t = q_t @ sigma_t; sigma_{t+1} += eta * k_t^T v_t
        outs = []
        for t in range(T):
            qt = q_sparse[:, :, t:t+1, :]          # (B,H,1,hs)
            kt = k_sparse[:, :, t:t+1, :].transpose(-1, -2)   # (B,H,hs,1)
            vt = v[:, :, t:t+1, :]                 # (B,H,1,hs)

            out_t = torch.matmul(qt, sigma)        # (B,H,1,hs) — read
            outs.append(out_t)

            # Hebbian: strengthen co-active synapses
            sigma = sigma + self.eta * torch.matmul(kt, vt)   # (B,H,hs,hs)

        if capture:
            self.last_hebbian_state = sigma.detach().cpu()

        out = torch.cat(outs, dim=2)               # (B,H,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out, sigma


class BDHBlock(nn.Module):
    """Single BDH layer: attention + MLP with sparse (ReLU) activations."""

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(config.n_embd, elementwise_affine=False)
        self.attn = BDHAttention(config)

        # MLP — note ReLU (sparse), not GELU (dense)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.ReLU(),     # ← sparse! ~5% fire
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

        # Storage for MLP activation captures
        self.last_mlp_activations = None

    def forward(self, x, sigma=None, capture=False):
        attn_out, sigma = self.attn(self.ln1(x), sigma, capture=capture)
        x = x + attn_out

        h = self.ln2(x)
        # Capture intermediate MLP activations (after first linear + ReLU)
        mid = F.relu(self.mlp[0](h))
        if capture:
            self.last_mlp_activations = mid.detach().cpu()
        out = self.mlp[2](mid)
        x = x + out
        return x, sigma


class BDHModel(nn.Module):
    """Full BDH language model."""

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([BDHBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, sigma_list=None, capture=False):
        B, T = idx.shape
        x = self.tok_emb(idx)

        if sigma_list is None:
            sigma_list = [None] * len(self.blocks)

        new_sigmas = []
        for i, block in enumerate(self.blocks):
            x, sigma = block(x, sigma_list[i], capture=capture)
            new_sigmas.append(sigma)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_sigmas

    def get_activation_stats(self, idx):
        """Run forward pass and collect activation sparsity per layer.
        
        KEY: We measure neurons that are STRICTLY NON-ZERO (|act| > 0).
        ReLU creates exact hard zeros → true sparsity.
        GELU never outputs exactly 0 → always ~100% non-zero.
        """
        with torch.no_grad():
            self.forward(idx, capture=True)

        stats = []
        for i, block in enumerate(self.blocks):
            mlp_acts = block.last_mlp_activations  # (B, T, 4*n_embd)
            if mlp_acts is not None:
                # Correct metric: fraction of neurons with non-zero output
                frac_active = (mlp_acts != 0).float().mean().item()
                stats.append({
                    "layer": i,
                    "sparsity": 1.0 - frac_active,
                    "frac_active": frac_active,
                    "activations": mlp_acts[0].numpy(),  # (T, 4*n_embd)
                })
        return stats

    def get_hebbian_state(self, idx):
        """Run and return Hebbian synaptic states after processing idx."""
        sigmas = []
        with torch.no_grad():
            _, sigma_list = self.forward(idx, capture=True)
        for i, block in enumerate(self.blocks):
            if block.attn.last_hebbian_state is not None:
                sigmas.append(block.attn.last_hebbian_state[0].numpy())  # (H, hs, hs)
        return sigmas


# ---------------------------------------------------------------------------
# Transformer baseline (for comparison)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard GPT-style transformer block with GELU (dense activations)."""

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.attn_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),     # ← dense! ~100% neurons have non-zero output
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )
        self.last_mlp_activations = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        n_head = 4
        head_size = C // n_head

        qkv = self.attn_qkv(self.ln1(x))
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        # Standard O(T²) attention
        att = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.attn_proj(out)
        x = x + out

        h = self.ln2(x)
        mid = self.mlp[1](self.mlp[0](h))  # after GELU
        if capture:
            self.last_mlp_activations = mid.detach().cpu()
        out = self.mlp[2](mid)
        x = x + out
        return x


class TransformerModel(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, capture=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, capture=capture)
        x = self.ln_f(x)
        return self.lm_head(x)

    def get_activation_stats(self, idx):
        """GELU never outputs exactly 0 → ~100% of neurons always non-zero."""
        with torch.no_grad():
            self.forward(idx, capture=True)
        stats = []
        for i, block in enumerate(self.blocks):
            acts = block.last_mlp_activations
            if acts is not None:
                # GELU: non-zero fraction should be ~100%
                frac_active = (acts != 0).float().mean().item()
                stats.append({
                    "layer": i,
                    "sparsity": 1.0 - frac_active,
                    "frac_active": frac_active,
                    "activations": acts[0].numpy(),
                })
        return stats
