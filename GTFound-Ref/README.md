# GTFound

GTFound is a haplotype model for **biallelic genotype imputation**. It predicts missing REF/ALT alleles from observed target alleles, aligned reference haplotypes, genomic positions, and linkage disequilibrium (LD). The model combines LD-enhanced self-attention with site-aligned cross-attention to reference haplotypes.

This repository provides pretrained-model inference and a phased **VCF → tensors → predictions → VCF** workflow. It does not include training scripts, reference panels, or evaluation datasets.

- **Author:** Wang Qiyu
- **Affiliation:** Harbin Institute of Technology, Shenzhen (HITSZ)
- **Training data:** 1000 Genomes Project, GRCh38
- **Architecture:** 12 layers, 12 attention heads, hidden dimension 768
- **Default input:** 1,024 panel variants per window, 128 overlapping variants, 8 reference haplotypes
- **License:** [MIT](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md)
- **Model download:** [unknow1024/GTFound on Hugging Face](https://huggingface.co/unknow1024/GTFound)

Each sequence position represents a panel variant record, not a nucleotide. This release requires reference haplotypes. Outputs are intended for research and have not been established as clinically calibrated predictions.

## Contents

- [Installation and setup](#installation-and-setup)
- [Quick start](#quick-start)
- [Impute your own VCF](#impute-your-own-vcf)
- [Tensor and Python interfaces](docs/INFERENCE.md)
- [VCF processing details](docs/VCF_PIPELINE.md)
- [Repository structure](#repository-structure)
- [Development and GitHub release](CONTRIBUTING.md)

## Installation and setup

### 1. Prepare the environment

Use Python 3.10+ and PyTorch 2.x. Linux is recommended for the VCF workflow. CPU inference is supported; an NVIDIA GPU with a compatible PyTorch/CUDA installation is recommended for chromosome-scale work. Memory use depends on window length, reference count, and batch size; start with `--batch-size 1`. This repository does not prescribe a measured minimum GPU memory requirement.

From the cloned or downloaded GitHub repository:

```bash
cd GTFound
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[download]'
gtfound --help
```

For CUDA, install the appropriate PyTorch build using the [official installation guide](https://pytorch.org/get-started/locally/) before installing this package. `python -m gtfound` is equivalent to the `gtfound` command.

### 2. Download the pretrained weights

Code is distributed through GitHub; weights are distributed separately through Hugging Face. Download only the checkpoint into the repository root so the checked-in configuration and vocabulary are retained:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='unknow1024/GTFound', filename='model.safetensors', local_dir='.')"
sha256sum -c SHA256SUMS
```

The checkpoint is approximately 910 MB (867 MiB). The checksum must report `model.safetensors: OK`. A mismatch means the downloaded checkpoint does not match this release.

The default model directory is the source repository root when installed in editable mode. For a regular `pip install .`, or weights stored elsewhere, pass `--model-dir /path/to/model` containing all three files: `model.safetensors`, `config.json`, and `vocab.txt`. These model artifacts are not bundled into the Python wheel. GTFound uses native PyTorch and is not loaded through Transformers `AutoModel.from_pretrained`.

### 3. Install PLINK for VCF input

The VCF preparation command additionally requires [PLINK 1.9](https://www.cog-genomics.org/plink/1.9/). Install it separately and check:

```bash
plink --version
```

The version must identify PLINK v1.9. If the executable is outside `PATH`, pass `--plink-bin /path/to/plink` to `prepare-vcf`. Tensor inference and the quick-start demo do not require PLINK.

## Quick start

Run the supplied demo after downloading the weights:

```bash
python quick_start.py --device cpu
# For an available NVIDIA GPU:
python quick_start.py --device cuda:0
```

The demo creates a deterministic synthetic window with 64 sites and 8 reference haplotypes, runs the pretrained model, and writes `outputs/quick_start/input.pt` and `prediction.pt`. It verifies probability normalization and preservation of observed alleles, then prints the output shape and masked-site count. The short window and zero LD are for an interface smoke test; they are not a biological benchmark or the recommended settings for real imputation.

## Impute your own VCF

Process **one target sample and one chromosome per work directory**. The paths, sample name, and chromosome below are examples; replace them with your own inputs.

### 1. Check the inputs

| Input | Requirements |
| --- | --- |
| Target VCF | Diploid GT; observed heterozygous and partially missing calls must be phased, e.g. `0\|1` or `0\|.`. Absent panel sites are treated as missing. |
| Reference VCF | Sorted, biallelic panel records with complete diploid 0/1 GT in eligible reference samples; heterozygotes must be phased. The default 8 reference haplotypes require at least 4 eligible samples. |
| Both VCFs | The same GRCh38 assembly and matching `CHROM/POS/REF/ALT`. Normalize indels, split multiallelic records, check alleles, phase, and perform quality control beforehand. `chr1` and `1` are different names. |

If the target sample name also occurs in the panel, it is excluded from reference selection and LD estimation. Other duplicate individuals and relatedness require separate checks. Assembly identity is not checked automatically. Only eligible panel sites are imputed; haploid GT is not supported.

### 2. Prepare aligned tensors and LD

```bash
gtfound prepare-vcf \
  --target-vcf target.phased.vcf.gz \
  --reference-vcf reference.phased.vcf.gz \
  --sample SAMPLE_001 \
  --chrom chr1 \
  --output-dir work_chr1
```

`work_chr1` must be absent or empty. Preparation selects nearest reference haplotypes using observed target alleles and computes per-window LD with PLINK 1.9. Windows contain **1,024 panel sites**, not 1,024 base pairs. Check `manifest.json`, especially `site_count`, `unmatched_target_variants`, `skipped_*`, and `windows_without_observed_sites`, before inference.

### 3. Run inference on each shard

```bash
mkdir -p work_chr1/predictions
for input in work_chr1/input_*.pt; do
  name=$(basename "$input")
  gtfound infer \
    --input "$input" \
    --output "work_chr1/predictions/${name/input_/pred_}" \
    --model-dir . \
    --device cuda:0 \
    --batch-size 1
done
```

Use `--device cpu` for CPU inference. Prediction filenames must correspond to the prepared shards: `input_0000.pt` becomes `pred_0000.pt`.

### 4. Export the imputed VCF

```bash
gtfound export-vcf \
  --prepared-dir work_chr1 \
  --predictions-dir work_chr1/predictions \
  --output-vcf SAMPLE_001.chr1.imputed.vcf.gz
```

The output path must not already exist. The exporter preserves observed alleles, averages ALT probabilities across overlapping windows, and calls missing alleles at a threshold of 0.5. Set `--min-confidence 0.9` to leave low-confidence alleles missing; select any threshold using independent validation data.

The result is a new single-sample VCF containing eligible panel sites and `GT` only. It does not copy other INFO/FORMAT fields, original FILTER values, or off-panel variants. Probabilities are conditional REF/ALT softmax values, not calibrated GP or DS. `.vcf.gz` output uses BGZF; optionally index it with `bcftools index -t`. See the [VCF processing guide](docs/VCF_PIPELINE.md) for alignment, filtering, LD, and overlap details.

For the same workflow in one command, use the [VCF example script](examples/README.md).

## Repository structure

```text
GTFound/
├── README.md                 # Installation and usage
├── quick_start.py            # Synthetic pretrained-model demo
├── pyproject.toml            # Package metadata and CLI entry point
├── requirements.txt          # Runtime dependencies
├── config.json               # Checkpoint architecture
├── vocab.txt                 # Fixed token order
├── SHA256SUMS                # Checkpoint integrity
├── gtfound/                  # Model, inference, CLI, and VCF processing
├── docs/                     # Detailed interface and VCF documentation
├── examples/                 # VCF workflow and synthetic input generator
├── tests/                    # Inference and VCF regression tests
├── .github/workflows/        # Automated CPU checks and packaging
├── CONTRIBUTING.md           # Development and release instructions
├── LICENSE                   # MIT license
├── THIRD_PARTY_NOTICES.md
└── third_party/ESM_LICENSE
```

`model.safetensors` remains local after download and is ignored by Git. Reference panels, private VCFs, generated tensors, and output directories are also ignored. See [CONTRIBUTING.md](CONTRIBUTING.md) before publishing a release.

## Limitations and data use

Validate on held-out samples with known genotypes before scientific use. Check ancestry, allele frequency, variant type, and missingness strata, along with alignment and preservation of observed GT. The repository supplies inference code and a checkpoint, not a clinical interpretation pipeline. Users must comply with the terms of their reference panels and target datasets; neither is distributed here.
