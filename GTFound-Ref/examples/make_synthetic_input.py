"""Create deterministic synthetic tensors for an interface smoke test."""

import argparse
from pathlib import Path

import torch


def make_input(output, length=64, reference_count=8):
    if length < 2 or reference_count < 1:
        raise ValueError("length must be >= 2 and reference_count must be positive")
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randint(0, 2, (1, length), generator=generator)
    tokens[:, ::4] = 5
    data = {
        "tokens": tokens,
        "references": torch.randint(
            0, 2, (1, reference_count, length), generator=generator
        ),
        "positions": torch.arange(100001, 100001 + 10 * length, 10),
        "ld_bias": torch.zeros(1, length, length),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/example/input.pt"))
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--reference-count", type=int, default=8)
    args = parser.parse_args()
    data = make_input(args.output, args.length, args.reference_count)
    print(f"Saved synthetic tokens {tuple(data['tokens'].shape)} to {args.output}")


if __name__ == "__main__":
    main()
