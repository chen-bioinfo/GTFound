"""Check overlap merging, original GT preservation, and PLINK integration."""

import csv
import json
import os
import shutil
import subprocess

import pysam
import pytest
import torch

from gtfound.cli import run
from gtfound.vcf_pipeline import export_vcf, prepare_vcf


def _write_tsv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)


@pytest.fixture
def prepared_predictions(tmp_path):
    prepared = tmp_path / "prepared"
    predictions = tmp_path / "predictions"
    prepared.mkdir()
    predictions.mkdir()
    manifest = {"site_count": 3, "window_length": 2, "sample": "TARGET",
                "chrom": "chr1", "input_files": ["input_0000.pt"]}
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_tsv(prepared / "sites.tsv",
               ["index", "chrom", "pos", "id", "ref", "alt", "target_h0", "target_h1"],
               [[0, "chr1", 10, ".", "A", "C", 0, 1],
                [1, "chr1", 20, ".", "A", "C", 5, 5],
                [2, "chr1", 30, ".", "A", "C", 5, 0]])
    _write_tsv(prepared / "windows.tsv", ["input_file", "row", "haplotype", "start_index"],
               [["input_0000.pt", 0, 0, 0], ["input_0000.pt", 1, 1, 0],
                ["input_0000.pt", 2, 0, 1], ["input_0000.pt", 3, 1, 1]])
    tokens = torch.tensor([[0, 5], [1, 5], [5, 5], [5, 0]])
    torch.save({"tokens": tokens}, prepared / "input_0000.pt")
    # At site 1, haplotype 0 averages ALT probabilities 0.2 and 0.9 to 0.55.
    alt = torch.tensor([[0.99, 0.2], [0.01, 0.05], [0.9, 0.95], [0.05, 0.99]])
    torch.save({"probabilities": torch.stack([1 - alt, alt], dim=-1),
                "mask": tokens.eq(5)}, predictions / "pred_0000.pt")
    return prepared, predictions


def test_overlap_merge_and_confidence_filter(prepared_predictions, tmp_path):
    prepared, predictions = prepared_predictions
    output = tmp_path / "result.vcf.gz"
    counts = export_vcf(prepared, predictions, output, min_confidence=0.8)
    with pysam.VariantFile(str(output)) as reader:
        calls = [record.samples["TARGET"]["GT"] for record in reader]
    assert calls == [(0, 1), (None, 0), (1, 0)]
    assert counts == {"imputed_alleles": 2, "unfilled_alleles": 1}
    with pytest.raises(FileExistsError):
        export_vcf(prepared, predictions, output)


def test_mismatched_mask_rejected(prepared_predictions, tmp_path):
    prepared, predictions = prepared_predictions
    path = predictions / "pred_0000.pt"
    result = torch.load(path, weights_only=True)
    result["mask"][0, 0] = True
    torch.save(result, path)
    with pytest.raises(ValueError, match="prediction mask does not match"):
        export_vcf(prepared, predictions, tmp_path / "invalid.vcf")


def test_plink_prepare_infer_export(model_dir, tmp_path):
    plink = shutil.which(os.environ.get("PLINK_BIN", "plink"))
    if plink is None:
        pytest.skip("PLINK 1.9 is not installed; set PLINK_BIN to enable integration")
    version = subprocess.run([plink, "--version"], check=True, capture_output=True,
                             text=True).stdout
    if "PLINK v1.9" not in version:
        pytest.skip("PLINK 1.9 is required for integration")
    header = ('##fileformat=VCFv4.2\n##contig=<ID=chr1>\n'
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    panel = tmp_path / "panel.vcf"
    target = tmp_path / "target.vcf"
    panel.write_text(header + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tR1\tR2\n"
                     + "".join(f"chr1\t{i * 10}\tv{i}\tA\tC\t.\tPASS\t.\tGT\t0|1\t1|0\n"
                               for i in range(1, 7)), encoding="utf-8")
    target.write_text(header + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTARGET\n"
                      + "".join(f"chr1\t{i * 10}\tv{i}\tA\tC\t.\tPASS\t.\tGT\t0|.\n"
                                for i in range(1, 7)), encoding="utf-8")
    prepared = tmp_path / "work"
    manifest = prepare_vcf(target, panel, "TARGET", "chr1", prepared,
                           window_length=4, overlap=1, reference_count=4,
                           windows_per_shard=1, plink_bin=plink)
    assert manifest["site_count"] == 6
    assert len(manifest["input_files"]) == 2
    predictions = tmp_path / "predictions"
    for filename in manifest["input_files"]:
        data = torch.load(prepared / filename, weights_only=True)
        assert torch.isfinite(data["ld_bias"]).all()
        assert torch.count_nonzero(data["ld_bias"].diagonal(dim1=-2, dim2=-1)) == 0
        run(prepared / filename, predictions / filename.replace("input_", "pred_"),
            model_dir, "cpu", batch_size=1)
    output = tmp_path / "result.vcf"
    counts = export_vcf(prepared, predictions, output)
    assert counts == {"imputed_alleles": 6, "unfilled_alleles": 0}
    with pysam.VariantFile(str(output)) as reader:
        records = list(reader)
    assert len(records) == 6
    assert all(record.samples["TARGET"]["GT"][0] == 0 for record in records)
    assert all(record.samples["TARGET"]["GT"][1] in (0, 1) for record in records)
