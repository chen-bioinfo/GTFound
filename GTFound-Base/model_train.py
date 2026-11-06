"""Train GTFound-Base on aligned haplotype windows with explicit input paths."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BertForMLM
from utils.runtime import DEFAULT_CONFIG, load_checkpoint, make_config, serializable_config
from utils.tensor_data import load_windows, mask_training_tokens, position_encoding


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.num_workers < 0:
        raise ValueError("epochs, batch size, and learning rate must be positive; workers must be nonnegative")
    if not 0 < args.mask_rate <= 1:
        raise ValueError("mask rate must be in (0, 1]")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    torch.manual_seed(args.seed)
    data = load_windows(args.input, training=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("output directory must be empty; use a new directory for each run or resume")
    checkpoint = None
    if args.resume:
        model, config, checkpoint = load_checkpoint(args.resume, args.device, args.batch_size)
    else:
        config = make_config(args.config, args.device, args.batch_size)
        model = BertForMLM(config).to(config.device)
    if data["tokens"].shape[1] > config.max_position_embeddings:
        raise ValueError("input window exceeds config.max_position_embeddings")
    dataset = TensorDataset(data["tokens"], data["positions"], data["ld_bias"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if config.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if checkpoint.get("mask_rate", args.mask_rate) != args.mask_rate:
            raise ValueError("resume mask rate must match the saved checkpoint")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0.0
        selected_sites = 0
        for tokens, positions, ld_bias in loader:
            tokens = tokens.to(config.device)
            masked, labels = mask_training_tokens(tokens, args.mask_rate)
            optimizer.zero_grad()
            loss, _ = model(
                input_ids=masked, attention_mask=torch.zeros_like(masked, dtype=torch.bool),
                position_ids=position_encoding(positions.to(config.device), config.hidden_size),
                ld_matrices=ld_bias.to(config.device), masked_lm_labels=labels,
            )
            loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            count = int(labels.ne(3).sum())
            total_loss += float(loss.detach()) * count
            selected_sites += count
        record = {"config": serializable_config(config), "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                  "loss": total_loss / selected_sites, "mask_rate": args.mask_rate,
                  "torch_rng_state": torch.get_rng_state()}
        if config.device.type == "cuda":
            record["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        output = output_dir / f"epoch_{epoch:04d}.pt"
        torch.save(record, output)
        logging.info("Epoch %d: masked-site loss %.6f; checkpoint %s", epoch, record["loss"], output)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Training tensor dictionary with complete 0/1 tokens")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty checkpoint directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Architecture JSON for a new run")
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer from a Base checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--epochs", type=int, default=1, help="Number of additional epochs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for a new run; resume restores the optimizer")
    parser.add_argument("--mask-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
