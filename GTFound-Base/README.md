# GTFound-Base

GTFound-Base is the reference-free implementation of GTFound for masked haplotype allele prediction. It conditions on observed target alleles, genomic position encodings, and precomputed linkage disequilibrium (LD). Unlike [GTFound-Ref](../GTFound-Ref/), it does not take reference haplotypes as model inputs. An external reference panel may still be needed to estimate LD.

The default architecture has 12 Transformer layers, 12 attention heads, hidden dimension 768, and a maximum window length of 1,024 panel sites. Each position represents a biallelic variant record, not a nucleotide. The fixed vocabulary is `0`, `1`, `2`, `[PAD]`, `[UNK]`, `[MASK]`; the public workflow accepts REF=0 and ALT=1, with missing inference alleles encoded as 5. Token 2 is not a haplotype allele in this interface.

This directory provides training from scratch, training resume, and tensor inference. It does not distribute a Base pretrained checkpoint, genotype data, or an end-to-end VCF conversion pipeline. GTFound-Ref weights are a different architecture and cannot be used as Base checkpoints.

## Installation

Use Python 3.10+ and PyTorch 2.x. Linux is recommended. CPU is supported. For GPU training, install a PyTorch build compatible with your driver and CUDA environment using the [official instructions](https://pytorch.org/get-started/locally/).

From the repository root:

```bash
cd GTFound-Base
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python model_train.py --help
python inference.py --help
```

This directory uses source-based scripts. Run them from `GTFound-Base`, or supply their absolute paths. Default architecture paths resolve relative to the script; explicit user paths resolve relative to the current working directory. No training data are downloaded automatically.

## Quick start

Run a small CPU smoke test without downloading human data or model weights:

```bash
python quick_start.py
```

It generates three synthetic 16-site windows, creates a small one-layer model, trains for one epoch, reloads the checkpoint, and imputes masked sites. It checks probability normalization and preservation of observed alleles. Results are saved under `outputs/quick_start/`, including `inputs/`, `checkpoints/epoch_0000.pt`, and `prediction.pt`.

The output directory must be absent or empty. Use `--output-dir outputs/another_run` for a new run. Short synthetic windows and zero LD test the software interface; they do not establish biological accuracy or reproduce full-scale training.

## Input format

Save a dictionary using `torch.save`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `tokens` | `[N,L]` | Integer haplotypes: 0=REF, 1=ALT. Inference additionally permits 5=`[MASK]`. |
| `positions` | `[L]` or `[N,L]` | Positive, sorted, one-based integer genomic coordinates. |
| `ld_bias` | `[N,L,L]` | Finite floating-point LD attention bias in the same site order. |

Training requires complete 0/1 truth. The trainer masks sites dynamically. There is no implicit padding, assembly conversion, window overlap merging, or allele harmonization. Normalize and phase variants beforehand; keep allele definitions, positions, and both LD axes aligned. Use one assembly and an independently selected LD panel. Normally zero the LD diagonal. Windows cannot exceed the configured maximum.

Prepared Ref shards can supply `tokens`, `positions`, and `ld_bias` for Base inference; `references` is ignored. Ref VCF export can consume the resulting predictions only when shard names, source tokens, and window mappings are preserved exactly. Masked inference shards are not valid training truth. See [data format details](docs/DATA_FORMAT.md). Only load trusted tensor and checkpoint files; public loaders use `weights_only=True`.

## Training

```bash
python model_train.py \
  --input data/train_windows.pt \
  --config bert_1KGPMLM/config.json \
  --output-dir outputs/train_run \
  --device cuda:0 --epochs 20 --batch-size 1 \
  --learning-rate 5e-5 --mask-rate 0.4 --seed 42
```

Use `--device cpu` for CPU training. The public trainer uses one device, full precision, Adam, and a constant learning rate. Each site is independently selected with the mask rate; at least one site per row is selected. Selected tokens become `[MASK]`; other labels are PAD=3 and ignored by cross-entropy. Each epoch writes a checkpoint. Output directories must be absent or empty. This recipe replaces the incomplete experiment-specific entry point; it does not claim to reproduce the original paper's training schedule.

Resume into a new output directory:

```bash
python model_train.py \
  --input data/train_windows.pt \
  --resume outputs/train_run/epoch_0019.pt \
  --output-dir outputs/resumed_run \
  --device cuda:0 --epochs 5 --batch-size 1 --mask-rate 0.4
```

`--epochs` counts additional epochs. Resume restores architecture, model, optimizer, and PyTorch random state. The saved optimizer learning rate overrides `--learning-rate`; `--config` is used only for a new run. Keep input, device setup, batch size, worker count, and mask rate unchanged when checking reproducibility. The trainer does not automatically report held-out accuracy or choose validation thresholds.

## Inference

```bash
python inference.py \
  --input data/input_windows.pt \
  --checkpoint outputs/train_run/epoch_0019.pt \
  --output outputs/prediction.pt \
  --device cuda:0 --batch-size 1
```

The output path must not exist. The dictionary contains conditional REF/ALT `probabilities [N,L,2]`, boolean `mask [N,L]`, and `imputed_tokens [N,L]`. Only original token 5 positions are filled by allele argmax; observed alleles are retained. Probabilities are not calibrated GP/DS values. Validate independently by ancestry, allele frequency, variant type, and missingness before scientific use.

## Compatibility and development

Read [the migration guide](docs/MIGRATION.md) before using historical checkpoints. New runs correct first-query attention broadcasting and use PAD index 3. Parameter names are retained, but corrected attention changes outputs. Historical numerical behavior requires the documented compatibility setting.

The `model/` modules retain historical model and auxiliary BERT heads. `utils/kgp_2.py` retains experimental dataset APIs for older callers; these are not the validated public training or VCF interfaces. New scripts use `utils/tensor_data.py`. Local experiment paths, commented-out experiments, and tracked Python bytecode have been removed.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use synthetic inputs and small CPU models. See [CONTRIBUTING.md](CONTRIBUTING.md) for release checks. All public comments, docstrings, logs, CLI help, and documentation are English.
