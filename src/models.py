from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ff_multiplier: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = MultiHeadSelfAttention(model_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * ff_multiplier, model_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in)
        x = x + self.dropout1(attn_out)
        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout2(ffn_out)
        return x


class WiFiTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.ff_multiplier = int(ff_multiplier)
        self.token_projection = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, input_dim + 1, model_dim) * 0.02)
        self.token_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(model_dim, num_heads, dropout, ff_multiplier) for _ in range(num_layers)])
        self.post_norm = nn.LayerNorm(model_dim)
        self.latent_head = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, input_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_missing_indicator(self, x: torch.Tensor) -> torch.Tensor:
        return (x <= 1e-6).float()

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(-1)
        missing = self._build_missing_indicator(x).unsqueeze(-1)
        token_input = torch.cat([value, missing], dim=-1)
        tokens = self.token_projection(token_input)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.size(1), :]
        tokens = self.token_dropout(tokens)
        return tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        encoded = self.post_norm(tokens)
        cls_repr = encoded[:, 0, :]
        mean_repr = encoded[:, 1:, :].mean(dim=1)
        latent = self.latent_head(torch.cat([cls_repr, mean_repr], dim=-1))
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decoder(latent)
        return reconstruction
