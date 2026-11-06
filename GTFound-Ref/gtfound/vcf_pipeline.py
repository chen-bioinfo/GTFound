"""Single-sample, single-chromosome phased VCF preparation and VCF export.

LD is computed per aligned panel window by PLINK 1.9 --r2 square gz yes-really,
with zero diagonal.
"""

import csv
import gzip
import json
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pysam
import torch


def _key(record):
    return (record.contig, record.pos, record.ref, record.alts[0])


def _biallelic(record):
    return (
        record.alts is not None
        and len(record.alts) == 1
        and record.ref
        and record.alts[0]
        and not record.alts[0].startswith("<")
        and "[" not in record.alts[0]
        and "]" not in record.alts[0]
    )


def _target_calls(path, sample, chrom):
    calls = {}
    by_pos = set()
    with pysam.VariantFile(str(path)) as reader:
        if sample not in reader.header.samples:
            raise ValueError(f"target sample {sample!r} not found in {path}")
        for record in reader:
            if record.contig != chrom:
                continue
            if not _biallelic(record):
                continue
            key = _key(record)
            if key in calls:
                raise ValueError(f"duplicate target variant: {key}")
            gt = record.samples[sample].get("GT")
            if gt is None or len(gt) != 2:
                raise ValueError(f"target GT must be diploid at {key}")
            a, b = gt
            if any(value not in (None, 0, 1) for value in (a, b)):
                raise ValueError(f"unsupported target GT at {key}: {gt}")
            if a != b and (a is not None or b is not None) and not record.samples[sample].phased:
                raise ValueError(f"unphased heterozygous target GT at {key}; phase before preparation")
            calls[key] = (5 if a is None else a, 5 if b is None else b)
            by_pos.add((record.contig, record.pos))
    return calls, by_pos


def _run_plink(plink_bin, *args):
    command = [plink_bin, *map(str, args)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"PLINK failed ({result.returncode}): {' '.join(command)}\n"
                           + result.stdout[-3000:])


def _plink_ld(plink_bin, panel_prefix, records, work_dir, window_length):
    ids = [item[1] for item in records]
    extract_path = work_dir / "window.extract"
    extract_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    chunk_prefix = work_dir / "window"
    try:
        _run_plink(plink_bin, "--bfile", panel_prefix, "--allow-extra-chr",
                   "--extract", extract_path, "--make-bed", "--out", chunk_prefix)
        with (work_dir / "window.bim").open(encoding="utf-8") as handle:
            extracted_ids = [line.split()[1] for line in handle if line.strip()]
        if extracted_ids != ids:
            raise ValueError("PLINK window variant IDs/order differ from model tensor; check panel variants")
        _run_plink(plink_bin, "--bfile", chunk_prefix, "--allow-extra-chr",
                   "--r2", "square", "gz", "yes-really", "--out", chunk_prefix)
        with gzip.open(work_dir / "window.ld.gz", "rt", encoding="utf-8") as handle:
            matrix = np.asarray([[float(value) for value in line.split()]
                                 for line in handle if line.strip()], dtype=np.float32)
        if matrix.shape != (window_length, window_length):
            raise ValueError(f"PLINK LD matrix shape {matrix.shape}; expected {(window_length, window_length)}")
        np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(matrix, 0.0)
        return torch.from_numpy(matrix)
    finally:
        extract_path.unlink(missing_ok=True)
        for suffix in (".bed", ".bim", ".fam", ".log", ".nosex", ".ld.gz"):
            (work_dir / f"window{suffix}").unlink(missing_ok=True)


def prepare_vcf(target_vcf, reference_vcf, sample, chrom, output_dir,
                window_length=1024, overlap=128, reference_count=8,
                windows_per_shard=8, plink_bin="plink"):
    if not (0 <= overlap < window_length) or window_length < 2:
        raise ValueError("require window_length >= 2 and 0 <= overlap < window_length")
    if reference_count < 1 or windows_per_shard < 1:
        raise ValueError("reference_count and windows_per_shard must be positive")
    resolved_plink = shutil.which(str(plink_bin))
    if resolved_plink is None:
        raise FileNotFoundError(f"PLINK 1.9 executable not found: {plink_bin}; use --plink-bin")
    version = subprocess.run([resolved_plink, "--version"], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             check=True).stdout.strip()
    if "PLINK v1.9" not in version:
        raise ValueError(f"this workflow requires PLINK 1.9; found: {version}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")

    target_calls, target_positions = _target_calls(target_vcf, sample, chrom)
    stats = {"skipped_non_biallelic": 0, "skipped_incomplete_reference": 0,
             "windows_without_observed_sites": 0}
    site_count = 0
    sites_path = output_dir / "sites.tsv"
    windows_path = output_dir / "windows.tsv"
    input_files = []
    with tempfile.TemporaryDirectory(prefix=".plink_", dir=output_dir) as temp_name:
        work_dir = Path(temp_name)
        panel_vcf = work_dir / "panel.vcf"
        panel_prefix = work_dir / "panel"
        with pysam.VariantFile(str(reference_vcf)) as reader, \
             sites_path.open("w", newline="", encoding="utf-8") as sites_handle, \
             panel_vcf.open("w", encoding="utf-8") as panel_handle:
            ref_samples = list(reader.header.samples)
            eligible_samples = [name for name in ref_samples if name != sample]
            if len(eligible_samples) < 2:
                raise ValueError("need at least two eligible reference samples after excluding the target")
            if 2 * len(eligible_samples) < reference_count:
                raise ValueError(f"only {2 * len(eligible_samples)} eligible reference haplotypes; requested {reference_count}")
            sites_writer = csv.writer(sites_handle, delimiter="\t")
            sites_writer.writerow(["index", "chrom", "pos", "id", "ref", "alt", "target_h0", "target_h1"])
            panel_handle.write("##fileformat=VCFv4.2\n")
            panel_handle.write(f"##contig=<ID={chrom}>\n")
            panel_handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
            panel_handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                               + "\t".join(eligible_samples) + "\n")
            last_pos = 0
            seen_at_pos = set()
            for record in reader:
                if record.contig != chrom:
                    continue
                if record.pos < last_pos:
                    raise ValueError(f"reference VCF is not sorted at {record.contig}:{record.pos}")
                if record.pos != last_pos:
                    seen_at_pos.clear()
                last_pos = record.pos
                if not _biallelic(record):
                    stats["skipped_non_biallelic"] += 1
                    continue
                key = _key(record)
                if key in seen_at_pos:
                    raise ValueError(f"duplicate reference variant: {key}")
                seen_at_pos.add(key)
                phased_gts = []
                for name in eligible_samples:
                    call = record.samples[name]
                    gt = call.get("GT")
                    if gt is None or len(gt) != 2 or any(a not in (0, 1) for a in gt) or (gt[0] != gt[1] and not call.phased):
                        break
                    phased_gts.append(f"{gt[0]}|{gt[1]}")
                if len(phased_gts) != len(eligible_samples):
                    stats["skipped_incomplete_reference"] += 1
                    continue
                if key not in target_calls and (record.contig, record.pos) in target_positions:
                    raise ValueError(f"target/reference allele mismatch at {key}; harmonize VCFs first")
                target_gt = target_calls.pop(key, (5, 5))
                variant_id = f"gtfound_{site_count}"
                sites_writer.writerow([site_count, record.contig, record.pos,
                                       record.id or ".", record.ref, record.alts[0],
                                       target_gt[0], target_gt[1]])
                panel_handle.write("\t".join([record.contig, str(record.pos), variant_id,
                                              record.ref, record.alts[0], ".", "PASS", ".",
                                              "GT", *phased_gts]) + "\n")
                site_count += 1

        if site_count < window_length:
            raise ValueError(f"only {site_count} eligible panel sites; need {window_length}")
        _run_plink(resolved_plink, "--vcf", panel_vcf, "--double-id",
                   "--allow-extra-chr", "--make-bed", "--out", panel_prefix)
        with (work_dir / "panel.bim").open(encoding="utf-8") as handle:
            bim_count = 0
            for line in handle:
                if not line.strip():
                    continue
                if line.split()[1] != f"gtfound_{bim_count}":
                    raise ValueError("PLINK changed the number/order of eligible panel variants")
                bim_count += 1
        if bim_count != site_count:
            raise ValueError("PLINK changed the number of eligible panel variants")
        stride = window_length - overlap
        buffer = deque(maxlen=window_length)
        shard_tokens, shard_refs, shard_positions, shard_ld = [], [], [], []
        next_start = 0
        last_start = None

        with pysam.VariantFile(str(panel_vcf)) as panel_reader, \
             sites_path.open(newline="", encoding="utf-8") as sites_handle, \
             windows_path.open("w", newline="", encoding="utf-8") as windows_handle:
            sites_reader = csv.DictReader(sites_handle, delimiter="\t")
            windows_writer = csv.writer(windows_handle, delimiter="\t")
            windows_writer.writerow(["input_file", "row", "haplotype", "start_index"])

            def flush():
                if not shard_tokens:
                    return
                filename = f"input_{len(input_files):04d}.pt"
                torch.save({
                    "tokens": torch.stack(shard_tokens),
                    "references": torch.stack(shard_refs),
                    "positions": torch.stack(shard_positions),
                    "ld_bias": torch.stack(shard_ld),
                }, output_dir / filename)
                input_files.append(filename)
                shard_tokens.clear()
                shard_refs.clear()
                shard_positions.clear()
                shard_ld.clear()

            def emit(start):
                nonlocal last_start
                records = list(buffer)
                if len(records) != window_length:
                    raise AssertionError("window buffer has wrong length")
                haps = np.stack([item[2] for item in records], axis=1)
                target = np.asarray([item[3] for item in records], dtype=np.int64).T
                pos = torch.tensor([item[0] for item in records], dtype=torch.long)
                ld = _plink_ld(resolved_plink, panel_prefix, records, work_dir, window_length)
                shard_name = f"input_{len(input_files):04d}.pt"
                for haplotype in (0, 1):
                    observed = target[haplotype] != 5
                    if observed.any():
                        distances = (haps[:, observed] != target[haplotype, observed]).sum(axis=1)
                    else:
                        distances = np.zeros(haps.shape[0], dtype=np.int32)
                        stats["windows_without_observed_sites"] += 1
                    nearest = np.argsort(distances, kind="stable")[:reference_count]
                    row = len(shard_tokens)
                    windows_writer.writerow([shard_name, row, haplotype, start])
                    shard_tokens.append(torch.from_numpy(target[haplotype].copy()))
                    shard_refs.append(torch.from_numpy(haps[nearest].astype(np.int64)))
                    shard_positions.append(pos)
                    shard_ld.append(ld)
                last_start = start
                if len(shard_tokens) >= 2 * windows_per_shard:
                    flush()

            for index, (record, row) in enumerate(zip(panel_reader, sites_reader, strict=True)):
                if record.id != f"gtfound_{index}" or int(row["index"]) != index:
                    raise ValueError("temporary panel and sites.tsv lost variant alignment")
                ref_haps = [allele for name in eligible_samples
                            for allele in record.samples[name]["GT"]]
                target_gt = (int(row["target_h0"]), int(row["target_h1"]))
                buffer.append((record.pos, record.id,
                               np.asarray(ref_haps, dtype=np.int8), target_gt))
                start = index + 1 - window_length
                if start == next_start:
                    emit(start)
                    next_start += stride
            tail_start = site_count - window_length
            if last_start != tail_start:
                emit(tail_start)
            flush()

    manifest = {
        "sample": sample, "chrom": chrom, "site_count": site_count,
        "window_length": window_length, "overlap": overlap,
        "reference_count": reference_count,
        "reference_sample_count": len(eligible_samples),
        "input_files": input_files, "unmatched_target_variants": len(target_calls),
        "ld_method": "PLINK 1.9 --r2 square gz yes-really; diagonal zero; nonfinite zero",
        "plink_version": version,
        **stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {site_count} sites, {len(input_files)} shards in {output_dir}")
    return manifest


def export_vcf(prepared_dir, predictions_dir, output_vcf, min_confidence=0.0):
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    prepared_dir = Path(prepared_dir)
    predictions_dir = Path(predictions_dir)
    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    site_count = manifest["site_count"]
    window_length = manifest["window_length"]
    score_sum = np.zeros((2, site_count), dtype=np.float64)
    coverage = np.zeros((2, site_count), dtype=np.int32)
    windows = {}
    with (prepared_dir / "windows.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            windows.setdefault(row["input_file"], []).append(row)
    for input_file in manifest["input_files"]:
        prediction_file = predictions_dir / input_file.replace("input_", "pred_", 1)
        result = torch.load(prediction_file, map_location="cpu", weights_only=True)
        probs = result["probabilities"].numpy()
        masks = result["mask"].numpy()
        rows = windows[input_file]
        if probs.shape != (len(rows), window_length, 2) or masks.shape != (len(rows), window_length):
            raise ValueError(f"prediction shape mismatch in {prediction_file}")
        source = torch.load(prepared_dir / input_file, map_location="cpu", weights_only=True)
        expected_mask = source["tokens"].eq(5).numpy()
        if not np.array_equal(masks, expected_mask):
            raise ValueError(f"prediction mask does not match {input_file}")
        if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=-1), 1, atol=1e-4):
            raise ValueError(f"invalid probabilities in {prediction_file}")
        for row in rows:
            i, hap, start = int(row["row"]), int(row["haplotype"]), int(row["start_index"])
            local = np.flatnonzero(masks[i])
            sites = start + local
            np.add.at(score_sum[hap], sites, probs[i, local, 1])
            np.add.at(coverage[hap], sites, 1)

    output_vcf = Path(output_vcf)
    if output_vcf.exists():
        raise FileExistsError(f"output VCF already exists: {output_vcf}")
    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    if output_vcf.suffix == ".gz":
        writer = pysam.BGZFile(str(output_vcf), "w")
        def write(line): writer.write(line.encode("utf-8"))
    else:
        writer = output_vcf.open("w", encoding="utf-8")
        def write(line): writer.write(line)
    counts = {"imputed_alleles": 0, "unfilled_alleles": 0}
    try:
        write("##fileformat=VCFv4.2\n")
        write("##source=GTFound\n")
        write(f"##contig=<ID={manifest['chrom']}>\n")
        write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + manifest["sample"] + "\n")
        with (prepared_dir / "sites.tsv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                j = int(row["index"])
                alleles = []
                for hap in (0, 1):
                    original = row[f"target_h{hap}"]
                    if original != "5":
                        alleles.append(original)
                        continue
                    if coverage[hap, j] == 0:
                        alleles.append(".")
                        counts["unfilled_alleles"] += 1
                        continue
                    p1 = score_sum[hap, j] / coverage[hap, j]
                    if max(p1, 1 - p1) < min_confidence:
                        alleles.append(".")
                        counts["unfilled_alleles"] += 1
                    else:
                        alleles.append("1" if p1 >= 0.5 else "0")
                        counts["imputed_alleles"] += 1
                write("\t".join([row["chrom"], row["pos"], row["id"], row["ref"], row["alt"],
                                   ".", "PASS", ".", "GT", "|".join(alleles)]) + "\n")
    finally:
        writer.close()
    print(f"Wrote {site_count} panel sites to {output_vcf}; {counts}")
    return counts
