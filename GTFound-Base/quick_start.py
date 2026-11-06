"""Run synthetic CPU training and imputation; this is not a biological benchmark."""

import argparse
from pathlib import Path
import subprocess
import sys

import torch

from examples.make_synthetic_input import make_inputs


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quick_start"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("quick-start output directory must be absent or empty")
    inputs = output / "inputs"
    make_inputs(inputs)
    subprocess.run([sys.executable, str(ROOT / "model_train.py"),
                    "--input", str(inputs / "train.pt"), "--config", str(inputs / "config.json"),
                    "--output-dir", str(output / "checkpoints"), "--epochs", "1",
                    "--batch-size", "2", "--device", "cpu"], check=True)
    subprocess.run([sys.executable, str(ROOT / "inference.py"),
                    "--input", str(inputs / "input.pt"), "--output", str(output / "prediction.pt"),
                    "--checkpoint", str(output / "checkpoints/epoch_0000.pt"),
                    "--batch-size", "2", "--device", "cpu"], check=True)
    source = torch.load(inputs / "input.pt", weights_only=True)
    result = torch.load(output / "prediction.pt", weights_only=True)
    probabilities = result["probabilities"]
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(-1), torch.ones_like(source["tokens"], dtype=torch.float32), atol=1e-5
    ):
        raise RuntimeError("invalid probabilities")
    observed = ~result["mask"]
    if not torch.equal(source["tokens"][observed], result["imputed_tokens"][observed]):
        raise RuntimeError("observed alleles changed")
    print(f"Smoke test passed: probability shape {tuple(probabilities.shape)}; observed alleles preserved.")
    print("Synthetic training does not establish imputation accuracy.")


if __name__ == "__main__":
    main()
