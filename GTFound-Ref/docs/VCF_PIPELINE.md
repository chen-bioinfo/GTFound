# Phased VCF preparation and export

This guide describes `gtfound prepare-vcf`, `gtfound infer`, and `gtfound export-vcf`. Each prepared directory represents one chromosome and one target sample. See the [README](../README.md) for installation and the complete workflow.

## 1. Input validation and filtering

- `--target-vcf` must contain the `--sample` name. Observed alleles condition prediction; absent panel sites are missing.
- `--reference-vcf` provides the panel variants. Eligible reference samples must have complete diploid 0/1 GT at every retained site, with phased heterozygotes. The target sample is excluded by exact sample name from retrieval, LD calculation, and reference-completeness checks. Other duplicates and relatedness require external checks.
- Match assembly and `CHROM/POS/REF/ALT` before preparation. Matching uses this complete four-field key. An eligible reference record with different alleles at a target position causes an error. Normalize, split multiallelic sites, check REF alleles, phase, and perform quality control first.
- Non-biallelic records, symbolic ALT alleles beginning with `<`, breakends, and sites with incomplete or unphased heterozygous reference calls are skipped. This is not a complete allele-normalization or QC procedure. Only eligible panel records are output; target-only sites are omitted.
- Target GT must be diploid. Observed heterozygotes and partially missing calls require phase; `0|.` assigns the known allele to the first haplotype. Unphased homozygotes and fully missing calls do not require an allele-order decision.
- `--chrom` must match the VCF contig text exactly. Retained panel sites must be sorted by position. Duplicate variant keys cause an error.
- At least `--window-length` eligible panel sites are required. Defaults are 1,024 panel records per window, not 1,024 base pairs.

## 2. VCF to aligned tensor shards

```bash
gtfound prepare-vcf \
  --target-vcf target.vcf.gz \
  --reference-vcf panel.vcf.gz \
  --sample SAMPLE_001 --chrom chr1 \
  --output-dir prepared_chr1 \
  --window-length 1024 --overlap 128 \
  --reference-count 8 --windows-per-shard 8
```

The output directory must be absent or empty. Each target GT becomes two haplotypes: REF = 0, ALT = 1, and missing = token 5. Each genomic window produces two tensor rows, one per target haplotype.

Reference selection is per window and per target haplotype. Hamming distances use observed target sites only. The nearest `--reference-count` haplotypes are selected; ties follow stable reference VCF sample/haplotype order. When no target sites are observed, the first reference haplotypes in that order are selected. These uninformative retrieval cases increment `windows_without_observed_sites`, which counts target-haplotype windows.

Windows advance by `window-length - overlap`. If needed, an additional right-aligned window covers the chromosome tail. Every window is full length; the final window may overlap more than the requested overlap.

| Output | Description |
| --- | --- |
| `manifest.json` | Sample, chromosome, retained-site count, window parameters, shard names, filtering statistics, LD method, and PLINK version. |
| `sites.tsv` | Global site index, `CHROM/POS/ID/REF/ALT`, and the two original target alleles. Missing alleles use token 5. |
| `windows.tsv` | Input shard filename, row number, target haplotype 0/1, and global start index. |
| `input_0000.pt`, etc. | Tensor dictionaries with `tokens`, `references`, `positions`, and `ld_bias`. |

The [tensor interface](INFERENCE.md) defines shapes and encoding. By default each shard contains 8 genomic windows, hence 16 target-haplotype rows. One 1,024 × 1,024 float32 LD matrix occupies approximately 4 MiB per row. Reduce `--windows-per-shard` to reduce shard buffering; this setting does not control model inference batch size.

Check `unmatched_target_variants`, `skipped_non_biallelic`, `skipped_incomplete_reference`, and `windows_without_observed_sites` in the manifest. `unmatched_target_variants` counts retained target keys not consumed by eligible panel records, including target-only variants and panel records rejected by filtering. A failed preparation can leave partial files; inspect them and use a fresh empty directory for a retry.

## 3. LD calculation with PLINK 1.9

PLINK must identify itself as v1.9. Set `--plink-bin` if it is not named `plink` on `PATH`. The pipeline:

1. Writes a temporary VCF with eligible reference samples and variants, excluding the target by sample name and assigning unique internal variant IDs. Converts it using `--vcf ... --double-id --allow-extra-chr --make-bed`.
2. Extracts each model window by internal IDs using `--bfile ... --extract ... --make-bed`. Verifies the BIM IDs and their order against the tensor site order. Variant loss or reordering stops preparation.
3. Runs `--bfile ... --allow-extra-chr --r2 square gz yes-really`. Reads the `.ld.gz` square matrix as float32, sets nonfinite values to zero, and zeros the diagonal. The two target haplotypes share the same per-window LD matrix.

LD uses all eligible panel samples, while cross-attention references are selected per target haplotype. PLINK defines the r-squared calculation; the package does not replace it with NumPy dosage correlation. Different panels, sample sets, and variant sets produce different LD. Retain the panel version, sample set, parameters, and manifest for reproducibility. Temporary PLINK files are cleaned up on success or failure; allow enough working disk space for panel conversion.

## 4. Inference and overlap-aware export

Run inference on every `input_*.pt` shard, saving corresponding `pred_*.pt` files. The exporter checks prediction shapes, masks against prepared source tensors, finite probabilities in [0, 1], and probability normalization.

```bash
gtfound export-vcf \
  --prepared-dir prepared_chr1 \
  --predictions-dir prepared_chr1/predictions \
  --output-vcf SAMPLE_001.chr1.imputed.vcf.gz \
  --min-confidence 0.9
```

For each missing allele, export averages ALT probability across all overlapping windows that cover it. It calls ALT at mean probability >= 0.5, otherwise REF. If the larger of REF and ALT probabilities is below `--min-confidence`, the allele stays missing. The default threshold is zero. Uncovered missing alleles also remain missing. Observed alleles are preserved exactly.

The output path must not exist. The result is a new panel-ordered single-sample VCF with `GT` only. It does not copy the input's INFO, other FORMAT values, FILTER, or off-panel variants, and it does not generate GP/DS. A `0|1` output combines the two target haplotypes; this does not independently validate the input phase. `.vcf.gz` output is BGZF-compressed and can be indexed with `bcftools index -t`.

## 5. Validation before scientific use

Use truth samples independent of the panel and mask known alleles according to the intended missingness pattern. Check allele orientation, variant mapping, tail coverage, observed-allele preservation, and missing-site coverage before evaluating accuracy. Stratify by ancestry, minor allele frequency, variant type, and missingness. Select confidence thresholds on independent validation data. The synthetic examples test software behavior and do not establish scientific performance.
