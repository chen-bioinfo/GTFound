"""Run pretrained inference on synthetic data; this is not an accuracy benchmark."""

import argparse
from pathlib import Path

import torch

from examples.make_synthetic_input import make_input
from gtfound.cli import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quick_start"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    input_path = args.output_dir / "input.pt"
    output_path = args.output_dir / "prediction.pt"
    data = make_input(input_path)
    run(input_path, output_path, args.model_dir, args.device, batch_size=1)
    result = torch.load(output_path, map_location="cpu", weights_only=True)
    probabilities = result["probabilities"]
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(data["tokens"], dtype=torch.float32),
        atol=1e-5,
    ):
        raise RuntimeError("invalid output probabilities")
    observed = ~result["mask"]
    if not torch.equal(result["imputed_tokens"][observed], data["tokens"][observed]):
        raise RuntimeError("observed alleles changed")
    print(f"Probability shape: {tuple(probabilities.shape)}")
    print(f"Masked sites: {int(result['mask'].sum())}; observed alleles preserved")
    print("Synthetic smoke test passed. These outputs do not measure biological accuracy.")


if __name__ == "__main__":
    main()
