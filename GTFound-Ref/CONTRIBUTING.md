# Contributing and preparing a release

## Development setup

Use Python 3.10+ and install a PyTorch build suitable for your machine:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```

The test suite uses small randomly initialized models and synthetic VCFs. It does not download weights or require a GPU. PLINK integration tests run when a PLINK 1.9 executable is available through `PLINK_BIN` or `PATH`; otherwise they are explicitly skipped. Test failure is not replaced by a skip when PLINK is present but fails.

For the pretrained smoke test, download the matching checkpoint and run:

```bash
sha256sum -c SHA256SUMS
python quick_start.py --device cpu
```

## Code and documentation

Use English for comments, docstrings, command-line help, user-facing messages, and documentation. Preserve the vocabulary order, checkpoint parameter names, and state-dictionary aliases when changing the architecture. Changes to model computation require separate scientific validation. Keep examples deterministic and identify synthetic data explicitly.

Report the command, package versions, expected behavior, actual behavior, and a minimal synthetic reproducer in bug reports. Do not attach private genotypes, patient identifiers, credentials, or restricted reference panels to public issues.

## GitHub release contents

The GitHub repository distributes source, documentation, examples, tests, configuration, and vocabulary. The pretrained checkpoint is distributed through [Hugging Face](https://huggingface.co/unknow1024/GTFound), with its expected checksum in `SHA256SUMS`. The existing local checkpoint can stay in the repository directory; `.gitignore` excludes it from normal Git staging. The `.gitattributes` LFS rule is retained for an explicitly chosen LFS distribution, but is not required for the default code-only GitHub release.

Before the first publication:

1. Run the tests and build the wheel and source distribution. Confirm neither archive contains checkpoint files, reference panels, or generated output.
2. Verify the hosted checkpoint against `SHA256SUMS` and run the pretrained demo.
3. Review the staged changes, file sizes, and Git history for private data or machine-specific paths. Ignore rules do not remove files already tracked in Git or present in history.
4. Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `third_party/ESM_LICENSE` in the release. Wheels carry the third-party notice and ESM license through setuptools license-file metadata.
5. Verify model-host links and document the code tag, dependency versions, checkpoint hash, and input-panel version used for any published result. Add a paper citation only once a real publication identifier is available.

The `.github/workflows/ci.yml` workflow runs CPU regression tests, rejects CJK text in public text files, and builds distribution archives. It does not establish biological accuracy or replace checkpoint-level validation.
