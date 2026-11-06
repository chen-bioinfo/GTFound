"""Small synthetic model artifacts for public-interface tests."""

import json

import pytest
import torch
from safetensors.torch import save_file

from gtfound import GTFoundModel, GTFoundVocab


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "vocab.txt").write_text("0\n1\n2\n[PAD]\n[UNK]\n[MASK]\n", encoding="utf-8")
    config = {"embed_dim": 24, "num_attention_heads": 4, "num_layers": 1, "dropout": 0.0}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.manual_seed(1)
    model = GTFoundModel(GTFoundVocab(path / "vocab.txt"), **config)
    # Clone shared parameters to retain every checkpoint alias in this fixture.
    save_file({key: value.clone() for key, value in model.state_dict().items()},
              str(path / "model.safetensors"))
    return path
