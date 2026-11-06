# Examples

Install GTFound first, following the [main README](../README.md). Commands below are run from the repository root.

## Synthetic tensor input

```bash
python examples/make_synthetic_input.py --output outputs/example/input.pt
```

This requires PyTorch but no checkpoint or PLINK. It creates a deterministic single-row tensor dictionary with 64 sites and 8 reference haplotypes. Missing target alleles occur every fourth site. Positions are synthetic and LD is zero. No human genotypes are included.

After downloading weights, run the complete synthetic demo:

```bash
python quick_start.py --device cpu
```

These examples exercise the input/output interface; they are not imputation benchmarks. Existing demo outputs at the same paths are overwritten. Use `--output-dir` to retain earlier runs.

## Complete VCF workflow

Prepare your own phased, normalized target and reference VCFs, install PLINK 1.9, and download the checkpoint. Then run:

```bash
bash examples/run_vcf_pipeline.sh \
  target.phased.vcf.gz reference.phased.vcf.gz \
  SAMPLE_001 chr1 work_chr1 SAMPLE_001.chr1.imputed.vcf.gz \
  . cuda:0 plink
```

The positional arguments are target VCF, reference VCF, target sample, chromosome, work directory, output VCF, model directory, optional device (default `cpu`), and optional PLINK executable (default `plink`). The work directory must be absent or empty, and the output VCF must not exist. Defaults are 1,024 sites per window, 128 overlapping sites, and 8 reference haplotypes. For custom parameters or confidence filtering, run the individual [CLI commands](../README.md#impute-your-own-vcf).
