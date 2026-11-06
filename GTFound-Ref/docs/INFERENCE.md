# Tensor inference and Python API

## Input tensors

Save a dictionary with `torch.save(data, "input.pt")`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `tokens` | `[N, L]` | Target haplotypes: 0 = REF, 1 = ALT, 5 = missing `[MASK]`. |
| `references` | `[N, R, L]` | Reference haplotypes, encoded as 0/1, aligned to the same variants. |
| `positions` | `[L]` or `[N, L]` | One-based GRCh38 genomic positions in the same variant order. |
| `ld_bias` | `[N, L, L]` | Floating-point LD attention bias in the same variant order. |

Use integer tensors for tokens, references, and positions, and finite float32 LD matrices. `N`, `R`, and `L` must be positive. The released VCF workflow uses `L=1024`, `R=8`, PLINK r-squared LD, and zero LD diagonal. The vocabulary is fixed: `0`, `1`, `2`, `[PAD]`, `[UNK]`, `[MASK]`, in that order. Token 2 and the padding/unknown tokens are not accepted by the imputation API. The released task predicts biallelic haplotypes, not unphased 0/1/2 genotype dosage.

Only load tensor files from trusted sources. CLI tensor loading uses `weights_only=True`.

## Command-line inference

```bash
gtfound infer \
  --input input.pt \
  --output prediction.pt \
  --model-dir . \
  --device cpu \
  --batch-size 1
```

Batch size controls how many input rows are evaluated together. The output contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `probabilities` | `[N, L, 2]` | Conditional softmax over REF and ALT logits, in that order. |
| `mask` | `[N, L]` | Boolean mask identifying original token 5. |
| `imputed_tokens` | `[N, L]` | Predictions at masked sites; observed tokens retained. |

The CLI produces an allele argmax for each masked position. VCF export instead averages overlapping-window ALT probabilities and applies its confidence threshold. At an exact 0.5/0.5 tie, tensor argmax selects REF while the VCF exporter selects ALT. Probabilities at observed sites are also returned, but those sites are not overwritten. Output files from `infer` can be overwritten; use a new output path when retaining previous results.

## Python API

```python
import torch
from gtfound.inference import load_gtfound, predict

model, vocab = load_gtfound(model_dir=".", device="cpu")
data = torch.load("input.pt", map_location="cpu", weights_only=True)
probabilities = predict(
    model,
    tokens=data["tokens"],
    references=data["references"],
    positions=data["positions"],
    ld_bias=data["ld_bias"],
)
print(probabilities.shape)  # [N, L, 2]
```

`load_gtfound` reads the configuration and vocabulary, strictly loads the safetensors state dictionary, and switches the model to evaluation mode. `predict` runs under `torch.inference_mode()` and returns a tensor on the model's device. Use `.cpu()` before serialization if CPU output is needed. Reference inputs are mandatory in this release.

## Synthetic input without a checkpoint

To inspect the tensor format before downloading weights:

```bash
python examples/make_synthetic_input.py --output outputs/example/input.pt
```

The deterministic generator uses 64 sites, 8 references, and zero LD by default. It does not contain human data and must not be used to estimate imputation accuracy. `python quick_start.py` combines this generator with pretrained inference and basic output checks.
