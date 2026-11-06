"""Impute aligned GTFound-Base tensor windows using a compatible checkpoint."""

import argparse
from pathlib import Path

import torch

from utils.runtime import load_checkpoint
from utils.tensor_data import load_windows, position_encoding


def run(input_path, output_path, checkpoint_path, device="cpu", batch_size=1):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    data = load_windows(input_path)
    model, config, _ = load_checkpoint(checkpoint_path, device, batch_size)
    if data["tokens"].shape[1] > config.max_position_embeddings:
        raise ValueError("input window exceeds config.max_position_embeddings")
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(data["tokens"]), batch_size):
            tokens = data["tokens"][start:start + batch_size].to(config.device)
            positions = data["positions"][start:start + batch_size].to(config.device)
            logits = model(
                input_ids=tokens, attention_mask=torch.zeros_like(tokens, dtype=torch.bool),
                position_ids=position_encoding(positions, config.hidden_size),
                ld_matrices=data["ld_bias"][start:start + batch_size].to(config.device),
            )
            batches.append(torch.softmax(logits[..., :2], dim=-1).cpu())
    probabilities = torch.cat(batches)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("inference produced nonfinite probabilities")
    mask = data["tokens"].eq(5)
    imputed = data["tokens"].clone()
    imputed[mask] = probabilities.argmax(-1)[mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"probabilities": probabilities, "mask": mask, "imputed_tokens": imputed}, output_path)
    print(f"Saved {len(imputed)} windows and {int(mask.sum())} imputed sites to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output .pt path")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    run(args.input, args.output, args.checkpoint, args.device, args.batch_size)


if __name__ == "__main__":
    main()
