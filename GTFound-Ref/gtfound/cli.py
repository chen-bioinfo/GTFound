"""Run GTFound on aligned haplotype windows stored in a PyTorch tensor file.

Input: a .pt dictionary with tokens [N,L], references [N,R,L], positions
[L] or [N,L], and ld_bias [N,L,L]. See README.md for allele coding.
"""

import argparse
from pathlib import Path

import torch

from .inference import load_gtfound, predict


def run(input_path, output_path, model_dir, device, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("input must be a dictionary of tensors")
    required = {"tokens", "references", "positions", "ld_bias"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"missing input tensors: {sorted(missing)}")
    if any(not isinstance(data[key], torch.Tensor) for key in required):
        raise ValueError("all four inputs must be torch.Tensor objects")

    tokens = data["tokens"]
    references = data["references"]
    positions = data["positions"]
    ld_bias = data["ld_bias"]
    if tokens.ndim != 2 or tokens.size(0) == 0:
        raise ValueError("tokens must have shape [N,L] with N > 0")
    n, length = tokens.shape
    if references.ndim != 3 or references.shape[0] != n or references.shape[2] != length or references.shape[1] == 0:
        raise ValueError("references must have shape [N,R,L] with R > 0")
    if positions.shape not in ((length,), (n, length)):
        raise ValueError("positions must have shape [L] or [N,L]")
    if ld_bias.shape != (n, length, length):
        raise ValueError("ld_bias must have shape [N,L,L]")

    model, vocab = load_gtfound(model_dir, device=device)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_positions = positions if positions.ndim == 1 else positions[start:end]
        probs = predict(
            model,
            tokens[start:end],
            references[start:end],
            batch_positions,
            ld_bias[start:end],
        )
        batches.append(probs.cpu())

    probabilities = torch.cat(batches, dim=0)
    mask = tokens.eq(vocab["[MASK]"])
    imputed_tokens = tokens.clone().to(dtype=torch.long)
    imputed_tokens[mask] = probabilities.argmax(dim=-1)[mask]
    output = {
        "probabilities": probabilities,  # [N,L,2], conditional softmax over 0/1
        "mask": mask,                      # [N,L], sites requested for imputation
        "imputed_tokens": imputed_tokens,  # [N,L], original observed tokens retained
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved {n} windows, {int(mask.sum())} masked sites to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gtfound", description="GTFound haplotype imputation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer_parser = subparsers.add_parser("infer", help="Impute aligned haplotype tensors in a .pt file")
    infer_parser.add_argument("--input", type=Path, required=True, help="Input .pt dictionary")
    infer_parser.add_argument("--output", type=Path, required=True, help="Output .pt dictionary")
    infer_parser.add_argument("--model-dir", type=Path, default=None, help="Model artifact directory; defaults to the source repository root")
    infer_parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    infer_parser.add_argument("--batch-size", type=int, default=1)
    prepare_parser = subparsers.add_parser("prepare-vcf", help="Prepare tensors and window mappings from phased VCFs")
    prepare_parser.add_argument("--target-vcf", type=Path, required=True)
    prepare_parser.add_argument("--reference-vcf", type=Path, required=True)
    prepare_parser.add_argument("--sample", required=True, help="Sample name in the target VCF")
    prepare_parser.add_argument("--chrom", required=True, help="Chromosome name, matching the VCF exactly")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--window-length", type=int, default=1024)
    prepare_parser.add_argument("--overlap", type=int, default=128)
    prepare_parser.add_argument("--reference-count", type=int, default=8)
    prepare_parser.add_argument("--windows-per-shard", type=int, default=8)
    prepare_parser.add_argument("--plink-bin", default="plink", help="PLINK 1.9 executable; defaults to plink on PATH")
    export_parser = subparsers.add_parser("export-vcf", help="Merge predicted windows into a single-sample VCF")
    export_parser.add_argument("--prepared-dir", type=Path, required=True)
    export_parser.add_argument("--predictions-dir", type=Path, required=True)
    export_parser.add_argument("--output-vcf", type=Path, required=True)
    export_parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "infer":
        run(args.input, args.output, args.model_dir, args.device, args.batch_size)
    elif args.command == "prepare-vcf":
        from .vcf_pipeline import prepare_vcf
        prepare_vcf(args.target_vcf, args.reference_vcf, args.sample, args.chrom,
                    args.output_dir, args.window_length, args.overlap,
                    args.reference_count, args.windows_per_shard, args.plink_bin)
    elif args.command == "export-vcf":
        from .vcf_pipeline import export_vcf
        export_vcf(args.prepared_dir, args.predictions_dir, args.output_vcf,
                   args.min_confidence)


if __name__ == "__main__":
    main()
