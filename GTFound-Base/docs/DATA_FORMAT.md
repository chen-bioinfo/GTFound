# Data, checkpoints, and outputs

## Aligned windows

The public scripts accept tensor dictionaries; extra keys are ignored.

- `tokens`: nonempty integer `[N,L]`. Training accepts only 0/1 truth. Inference additionally accepts 5. Floating-point tokens are rejected rather than silently truncated.
- `positions`: positive integer `[L]` or `[N,L]`, sorted within each window. Shared coordinates are expanded across rows. Sinusoidal features encode genomic positions at the configured hidden dimension.
- `ld_bias`: finite floating-point `[N,L,L]`, cast to float32. Values are added to every attention head before softmax. The loader does not calculate or normalize LD. Matrix rows and columns must match the tokens exactly.

One row is one haplotype window; diploid samples need two rows per genomic window. Unphased 0/1/2 dosage is not a haplotype input. No automatic padding or window construction is performed. Input length must not exceed the configured maximum.

Generate format examples with:

```bash
python examples/make_synthetic_input.py --output-dir outputs/synthetic
```

This creates complete training windows, masked inference windows, and a small one-layer configuration. No human data are included.

## Training checkpoints

The trainer writes `epoch_0000.pt`, `epoch_0001.pt`, and so on:

| Key | Content |
| --- | --- |
| `config` | JSON-compatible architecture and attention compatibility mode. |
| `model_state_dict` | Weights retaining historical Base parameter names. |
| `optimizer_state_dict` | Adam state for training resume. |
| `epoch` | Zero-based index of the completed epoch. |
| `loss` | Cross-entropy averaged over selected prediction sites. |
| `mask_rate` | Per-site masking probability. |
| `torch_rng_state` | CPU random state after the epoch. |
| `cuda_rng_state` | CUDA generator states when training on CUDA. |

Resume is supported at epoch boundaries. Exact GPU reproducibility can depend on kernels and software versions. Preserve the code revision, dependencies, input and LD provenance, configuration, seed, and training arguments for research results.

## Inference output

| Key | Shape | Content |
| --- | --- | --- |
| `probabilities` | `[N,L,2]` | CPU softmax over REF/ALT logits only. |
| `mask` | `[N,L]` | Original token 5 positions. |
| `imputed_tokens` | `[N,L]` | Original observed alleles and imputed masked alleles. |

An exact probability tie selects REF by PyTorch argmax. Inference does not merge overlaps, filter by confidence, export VCF metadata, or generate calibrated GP/DS. Existing output files and nonempty training output directories are rejected. All public tensor and checkpoint loading uses `weights_only=True`; use trusted files.
