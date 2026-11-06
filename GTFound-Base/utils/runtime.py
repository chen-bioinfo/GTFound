"""Shared configuration and checkpoint loading for the public Base interfaces."""

import json
from pathlib import Path

import torch

from model import BertConfig, BertForMLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "bert_1KGPMLM" / "config.json"


def resolve_device(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices are cpu and cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_config(path=DEFAULT_CONFIG, device="cpu", batch_size=1):
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    return config_from_dict(values, device, batch_size)


def config_from_dict(values, device="cpu", batch_size=1):
    config = BertConfig.from_dict(values)
    defaults = {"attention": "", "use_torch_multi_head": False,
                "use_embedding_weight": True, "pooler_type": "all_token_average",
                "all_possible_position": None, "legacy_attention_broadcast": False}
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    if config.vocab_size != 6 or config.pad_token_id != 3:
        raise ValueError("Base requires vocab_size=6 and pad_token_id=3; see docs/MIGRATION.md")
    if (config.hidden_size < 2 or config.hidden_size % 2
            or config.num_attention_heads < 1
            or config.hidden_size % config.num_attention_heads
            or config.num_hidden_layers < 1 or config.max_position_embeddings < 512):
        raise ValueError("require even hidden_size divisible by attention heads, layers >= 1, and max_position_embeddings >= 512")
    if config.attention or config.use_torch_multi_head:
        raise ValueError("the public Base workflow supports the default LD attention implementation")
    config.device = resolve_device(str(device))
    config.batch_size = batch_size
    return config


def serializable_config(config):
    values = config.to_dict()
    values.pop("device", None)
    values.pop("all_possible_position", None)
    return values


def load_checkpoint(path, device="cpu", batch_size=1):
    """Load a public training checkpoint containing config and model_state_dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "model_state_dict"} <= checkpoint.keys():
        raise ValueError("expected a Base checkpoint with config and model_state_dict; see docs/MIGRATION.md")
    config = config_from_dict(checkpoint["config"], device, batch_size)
    model = BertForMLM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(config.device), config, checkpoint
