"""Regression tests for checkpoint loading, batching, and allele preservation."""

import pytest
import torch

from gtfound.cli import run
from gtfound.inference import load_gtfound, predict


def test_batching_preserves_observed_alleles(model_dir, tmp_path):
    tokens = torch.tensor([[0, 5, 1, 5], [5, 1, 5, 0], [1, 0, 5, 5]])
    data = {
        "tokens": tokens,
        "references": torch.tensor([[[0, 1, 1, 0], [1, 0, 0, 1]]] * 3),
        "positions": torch.tensor([100, 200, 300, 400]),
        "ld_bias": torch.zeros(3, 4, 4),
    }
    input_path = tmp_path / "input.pt"
    torch.save(data, input_path)
    outputs = []
    for batch_size in (1, 2, 3):
        output_path = tmp_path / f"batch_{batch_size}.pt"
        run(input_path, output_path, model_dir, "cpu", batch_size)
        result = torch.load(output_path, weights_only=True)
        probabilities = result["probabilities"]
        assert probabilities.shape == (3, 4, 2)
        assert torch.isfinite(probabilities).all()
        assert torch.allclose(probabilities.sum(-1), torch.ones(3, 4))
        assert torch.equal(result["mask"], tokens.eq(5))
        assert torch.equal(result["imputed_tokens"][tokens != 5], tokens[tokens != 5])
        outputs.append(probabilities)
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)
    assert torch.allclose(outputs[0], outputs[2], atol=1e-6)


def test_invalid_alleles_rejected(model_dir):
    model, _ = load_gtfound(model_dir)
    with pytest.raises(ValueError, match="tokens may contain only"):
        predict(model, torch.tensor([[0, 2]]), torch.zeros(1, 2, 2, dtype=torch.long),
                torch.tensor([10, 20]), torch.zeros(1, 2, 2))


def test_missing_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model files missing"):
        load_gtfound(tmp_path)
