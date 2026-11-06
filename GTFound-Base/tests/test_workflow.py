"""Behavioral regressions for attention, imputation, and training resume."""

import pytest
import torch

from examples.make_synthetic_input import make_inputs
from inference import run
from model import BertForMLM
from model.BasicBert.Bert1 import BertAttention
from model_train import main as train_main
from utils.runtime import make_config
from utils.tensor_data import load_windows


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / "inputs"
    make_inputs(directory)
    return directory


def test_attention_uses_each_query_and_preserves_checkpoint_keys(inputs):
    config = make_config(inputs / "config.json")
    layer = BertAttention(config).eval()
    hidden = torch.randn(5, 2, config.hidden_size)
    bias = torch.zeros(2, 5, 5)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    with torch.no_grad():
        attended, _ = layer.self(hidden, hidden, hidden, key_padding_mask=mask, ld_matrices=bias)
        output, _ = layer(hidden, mask, bias)
        assert torch.allclose(output, layer.output(attended, hidden))
        layer.legacy_attention_broadcast = True
        legacy, _ = layer(hidden, mask, bias)
        assert torch.allclose(legacy, layer.output(attended[0], hidden))
        assert not torch.allclose(output, legacy)
    config.legacy_attention_broadcast = True
    old_mode = BertForMLM(config)
    config.legacy_attention_broadcast = False
    new_mode = BertForMLM(config)
    new_mode.load_state_dict(old_mode.state_dict(), strict=True)
    assert new_mode.bert.bert_embeddings.word_embeddings.embedding.padding_idx == 3


def _train(inputs, output, epochs=1, resume=None):
    args = ["--input", str(inputs / "train.pt"), "--config", str(inputs / "config.json"),
            "--output-dir", str(output), "--epochs", str(epochs), "--batch-size", "2"]
    if resume is not None:
        args += ["--resume", str(resume)]
    train_main(args)


def test_training_resume_matches_uninterrupted_run(inputs, tmp_path):
    _train(inputs, tmp_path / "full", epochs=2)
    _train(inputs, tmp_path / "part")
    _train(inputs, tmp_path / "resumed", resume=tmp_path / "part/epoch_0000.pt")
    full = torch.load(tmp_path / "full/epoch_0001.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/epoch_0001.pt", weights_only=True)
    assert full["loss"] == pytest.approx(resumed["loss"], abs=1e-8)
    assert all(torch.equal(value, resumed["model_state_dict"][key])
               for key, value in full["model_state_dict"].items())


def test_inference_batching_and_preservation(inputs, tmp_path):
    _train(inputs, tmp_path / "train")
    checkpoint = tmp_path / "train/epoch_0000.pt"
    outputs = []
    source = torch.load(inputs / "input.pt", weights_only=True)["tokens"]
    for size in (1, 2, 3):
        path = tmp_path / f"prediction_{size}.pt"
        run(inputs / "input.pt", path, checkpoint, batch_size=size)
        result = torch.load(path, weights_only=True)
        observed = ~result["mask"]
        assert torch.equal(result["imputed_tokens"][observed], source[observed])
        assert torch.equal(result["mask"], source.eq(5))
        assert torch.isfinite(result["probabilities"]).all()
        assert torch.allclose(result["probabilities"].sum(-1), torch.ones_like(source).float())
        assert torch.all((result["imputed_tokens"] == 0) | (result["imputed_tokens"] == 1))
        outputs.append(result["probabilities"])
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)
    assert torch.allclose(outputs[0], outputs[2], atol=1e-6)
    with pytest.raises(FileExistsError):
        run(inputs / "input.pt", tmp_path / "prediction_1.pt", checkpoint)


@pytest.mark.parametrize("case", ["fractional_tokens", "nan_ld", "zero_position", "masked_truth"])
def test_invalid_inputs_rejected(inputs, tmp_path, case):
    data = torch.load(inputs / "train.pt", weights_only=True)
    if case == "fractional_tokens":
        data["tokens"] = data["tokens"].float() + 0.1
    elif case == "nan_ld":
        data["ld_bias"][0, 0, 0] = float("nan")
    elif case == "zero_position":
        data["positions"][0] = 0
    else:
        data["tokens"][0, 0] = 5
    path = tmp_path / "invalid.pt"
    torch.save(data, path)
    with pytest.raises(ValueError):
        load_windows(path, training=True)
