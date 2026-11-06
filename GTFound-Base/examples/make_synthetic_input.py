"""Generate deterministic synthetic windows and a small smoke-test config."""

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def make_inputs(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("synthetic output directory must be empty")
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randint(0, 2, (3, 16), generator=generator)
    common = {"positions": torch.arange(100001, 100161, 10), "ld_bias": torch.zeros(3, 16, 16)}
    torch.save({"tokens": tokens, **common}, output_dir / "train.pt")
    masked = tokens.clone()
    masked[:, ::4] = 5
    torch.save({"tokens": masked, **common}, output_dir / "input.pt")
    config = json.loads((ROOT / "bert_1KGPMLM/config.json").read_text())
    config.update(hidden_size=24, intermediate_size=48, num_hidden_layers=1,
                  num_attention_heads=4, hidden_dropout_prob=0.0,
                  attention_probs_dropout_prob=0.0)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved synthetic training/inference inputs and a small config to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    make_inputs(args.output_dir)


if __name__ == "__main__":
    main()
