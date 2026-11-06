"""Masked allele prediction head used by GTFound."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def gelu(x):
    """Exact GELU activation used by this model."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GTFoundLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        return F.linear(x, self.weight) + self.bias
