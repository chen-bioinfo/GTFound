#!/usr/bin/env bash
# Run one target sample and one chromosome through the complete VCF workflow.
set -euo pipefail

if [[ $# -lt 7 || $# -gt 9 ]]; then
  echo "Usage: bash $0 TARGET_VCF REFERENCE_VCF SAMPLE CHROM WORK_DIR OUTPUT_VCF MODEL_DIR [DEVICE] [PLINK_BIN]" >&2
  exit 2
fi

target_vcf=$1
reference_vcf=$2
sample=$3
chrom=$4
work_dir=$5
output_vcf=$6
model_dir=$7
device=${8:-cpu}
plink_bin=${9:-plink}

if [[ -e "$output_vcf" ]]; then
  echo "Output already exists: $output_vcf" >&2
  exit 1
fi

python -m gtfound prepare-vcf \
  --target-vcf "$target_vcf" --reference-vcf "$reference_vcf" \
  --sample "$sample" --chrom "$chrom" --output-dir "$work_dir" \
  --plink-bin "$plink_bin"

mkdir -p "$work_dir/predictions"
for input in "$work_dir"/input_*.pt; do
  name=$(basename "$input")
  python -m gtfound infer \
    --input "$input" --output "$work_dir/predictions/${name/input_/pred_}" \
    --model-dir "$model_dir" --device "$device" --batch-size 1
done

python -m gtfound export-vcf \
  --prepared-dir "$work_dir" --predictions-dir "$work_dir/predictions" \
  --output-vcf "$output_vcf"
