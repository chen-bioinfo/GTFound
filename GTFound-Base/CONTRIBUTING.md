# Development and release checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python quick_start.py --output-dir outputs/release_smoke
```

Use a new output directory for each smoke test. Tests cover English source, portable paths, attention output handling, checkpoint compatibility, observed-allele preservation, input validation, and training resume. They do not replace full training reproduction or held-out imputation evaluation.

Keep comments, docstrings, documentation, logs, and CLI help in English. Preserve vocabulary order and document computation changes in the migration guide. Retain source references in rotary embedding documentation. Do not commit private genomic data, machine paths, weights, caches, or environment files.

From the parent repository, review this directory without opening a pager:

```bash
git add -A -- GTFound-Base/
git --no-pager diff --cached --stat
git --no-pager diff --cached --check
```

Stage only intended files. Ignore rules do not remove previously published files from history. Record code revision, configuration, checkpoint provenance, and environment for published benchmarks.
