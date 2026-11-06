"""Load the GTFound haplotype imputation model from a model directory."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .modeling_gtfound import GTFoundModel
from .vocab_gtfound import GTFoundVocab


def load_gtfound(model_dir=None, device="cpu"):
    model_dir = Path(model_dir or Path(__file__).resolve().parents[1])
    required = ["config.json", "vocab.txt", "model.safetensors"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"GTFound model files missing in {model_dir}: {missing}. "
            "Use --model-dir to point to the downloaded model repository."
        )
    config = json.loads((model_dir / "config.json").read_text())
    vocab = GTFoundVocab(model_dir / "vocab.txt")
    model = GTFoundModel(
        vocab=vocab,
        embed_dim=config["embed_dim"],
        num_attention_heads=config["num_attention_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, vocab


def position_encoding(genomic_positions, hidden_size=768):
    """Return sin/cos encoding of one-based genomic POS values."""
    import math

    pos = torch.as_tensor(genomic_positions, dtype=torch.float32)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if pos.ndim != 2:
        raise ValueError("genomic_positions must have shape [B, L] or [L]")
    div = torch.exp(
        torch.arange(0, hidden_size, 2, dtype=torch.float32)
        * -(math.log(10000.0) / hidden_size)
    )
    out = torch.zeros((*pos.shape, hidden_size), dtype=torch.float32)
    out[..., 0::2] = torch.sin(pos.unsqueeze(-1) * div)
    out[..., 1::2] = torch.cos(pos.unsqueeze(-1) * div)
    return out


def predict(model, tokens, references, positions, ld_bias):
    """Return [B,L,2] probabilities for alleles 0 and 1."""
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    references = torch.as_tensor(references, dtype=torch.long)
    positions = torch.as_tensor(positions)
    ld_bias = torch.as_tensor(ld_bias)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [B, L]")
    batch, length = tokens.shape
    if references.ndim != 3 or references.shape[0] != batch or references.shape[2] != length:
        raise ValueError("references must have shape [B, R, L]")
    if positions.ndim == 1:
        positions = positions.expand(batch, -1)
    if positions.shape != (batch, length):
        raise ValueError("positions must have shape [B, L] or [L]")
    if ld_bias.shape != (batch, length, length):
        raise ValueError("ld_bias must have shape [B, L, L]")
    if not torch.all((tokens == 0) | (tokens == 1) | (tokens == 5)):
        raise ValueError("tokens may contain only 0, 1, or 5 ([MASK])")
    if not torch.all((references == 0) | (references == 1)):
        raise ValueError("references may contain only 0 or 1")
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(
            tokens=tokens.to(device),
            positions_embedding=position_encoding(
                positions, hidden_size=model.embed_tokens.embedding_dim
            ).to(device),
            ref=references.to(device),
            LD_data=ld_bias.to(device=device, dtype=torch.float32),
        )
        return torch.softmax(logits[..., :2], dim=-1)
