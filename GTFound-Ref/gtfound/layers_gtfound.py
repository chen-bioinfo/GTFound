"""Transformer blocks used by GTFound inference."""

import torch
from torch import nn
from torch.nn import functional as F


class GTFoundLayer(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        ffn_embedding_dim=3072,
        num_attention_heads=12,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
    ):
        super().__init__()
        self.self_attention_layer = SelfAttention(
            embedding_dim, num_attention_heads, dropout=attention_dropout
        )
        self.cross_attention_layer = CrossAttention(
            embedding_dim, num_attention_heads, dropout=attention_dropout
        )
        self.feed_forward_layer = FeedForwardNetwork(
            embedding_dim, ffn_embedding_dim, activation_dropout=activation_dropout
        )
        # Keep both module aliases for state_dict compatibility.
        self.self_attention_norm = NormalizedResidualBlock(
            self.self_attention_layer, embedding_dim, dropout
        )
        self.cross_attention_norm = NormalizedResidualBlock(
            self.cross_attention_layer, embedding_dim, dropout
        )
        self.feed_forward_norm = NormalizedResidualBlock(
            self.feed_forward_layer, embedding_dim, dropout
        )

    def forward(self, x, ref, ld_bias, padding_mask=None):
        x = self.self_attention_norm(
            x, LD_data=ld_bias, key_padding_mask=padding_mask
        )
        x = self.cross_attention_norm(x, ref=ref)
        return self.feed_forward_norm(x)


class NormalizedResidualBlock(nn.Module):
    def __init__(self, layer, embedding_dim, dropout=0.1):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, **kwargs):
        return x + self.dropout(self.layer(self.layer_norm(x), **kwargs))


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, activation_dropout=0.1):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(activation_dropout)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.activation_fn(self.fc1(x))))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, LD_data, key_padding_mask=None):
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        bias = LD_data.unsqueeze(1).to(dtype=q.dtype, device=q.device)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, length):
                raise ValueError("key_padding_mask must have shape [batch, sites]")
            padding_bias = torch.zeros((batch, 1, 1, length), dtype=q.dtype, device=q.device)
            padding_bias.masked_fill_(
                key_padding_mask[:, None, None, :].to(device=q.device),
                torch.finfo(q.dtype).min,
            )
            bias = bias + padding_bias
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(output)


class CrossAttention(nn.Module):
    """Each target site attends to the same site in all reference haplotypes."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, ref):
        batch, sites, _ = x.shape
        if ref.ndim != 4:
            raise ValueError("ref must have shape [batch, references, sites, width]")
        reference_count = ref.size(1)
        q = self.q_proj(x).view(batch, sites, self.num_heads, self.head_dim)
        q = q.unsqueeze(3).reshape(batch * sites, self.num_heads, 1, self.head_dim)
        k = self.k_proj(ref).view(batch, reference_count, sites, self.num_heads, self.head_dim)
        v = self.v_proj(ref).view(batch, reference_count, sites, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 3, 1, 4).reshape(batch * sites, self.num_heads, reference_count, self.head_dim)
        v = v.permute(0, 2, 3, 1, 4).reshape(batch * sites, self.num_heads, reference_count, self.head_dim)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        output = output.view(batch, sites, self.num_heads, self.head_dim).flatten(2)
        return self.out_proj(output)
