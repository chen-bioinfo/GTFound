"""Validate aligned Base-model windows and encode genomic positions."""

import math

import torch


INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def load_windows(path, training=False):
    """Load a tensor-only dictionary, rejecting invalid or misaligned inputs."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("input must be a dictionary of tensors")
    required = {"tokens", "positions", "ld_bias"}
    if required - data.keys():
        raise ValueError(f"missing input tensors: {sorted(required - data.keys())}")
    if any(not isinstance(data[key], torch.Tensor) for key in required):
        raise ValueError("tokens, positions, and ld_bias must be tensors")
    tokens, positions, ld_bias = (data[key] for key in ("tokens", "positions", "ld_bias"))
    if tokens.ndim != 2 or min(tokens.shape) < 1 or tokens.dtype not in INTEGER_DTYPES:
        raise ValueError("tokens must be an integer tensor [N, L] with N, L > 0")
    n, length = tokens.shape
    valid = (tokens == 0) | (tokens == 1)
    if not training:
        valid |= tokens == 5
    if not valid.all():
        raise ValueError("training tokens must be 0/1; inference also permits 5 ([MASK])")
    if positions.shape == (length,):
        positions = positions.expand(n, -1)
    if positions.shape != (n, length) or positions.dtype not in INTEGER_DTYPES:
        raise ValueError("positions must be integer genomic coordinates [L] or [N, L]")
    if (positions < 1).any() or (positions[:, 1:] < positions[:, :-1]).any():
        raise ValueError("positions must be positive and sorted within each window")
    if ld_bias.shape != (n, length, length) or not ld_bias.is_floating_point():
        raise ValueError("ld_bias must be a floating-point tensor [N, L, L]")
    if not torch.isfinite(ld_bias).all():
        raise ValueError("ld_bias must contain only finite values")
    return {"tokens": tokens.long(), "positions": positions.long(), "ld_bias": ld_bias.float()}


def position_encoding(positions, hidden_size):
    """Encode one-based genomic coordinates as [batch, sites, hidden_size]."""
    if hidden_size < 2 or hidden_size % 2:
        raise ValueError("hidden_size must be a positive even number")
    positions = positions.to(dtype=torch.float32)
    div = torch.exp(torch.arange(0, hidden_size, 2, device=positions.device)
                    * -(math.log(10000.0) / hidden_size))
    angles = positions.unsqueeze(-1) * div
    output = torch.empty((*positions.shape, hidden_size), device=positions.device)
    output[..., 0::2] = angles.sin()
    output[..., 1::2] = angles.cos()
    return output


def mask_training_tokens(tokens, rate):
    """Mask at least one site per row and ignore unselected sites in MLM loss."""
    if not 0 < rate <= 1:
        raise ValueError("mask rate must be in (0, 1]")
    selected = torch.rand(tokens.shape, device=tokens.device) < rate
    empty = ~selected.any(dim=1)
    selected[empty, 0] = True
    labels = torch.where(selected, tokens, 3)
    return tokens.masked_fill(selected, 5), labels
