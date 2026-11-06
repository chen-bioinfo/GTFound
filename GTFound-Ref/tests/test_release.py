"""Enforce English public text and portable release sources."""

import re
from pathlib import Path


def test_public_text_has_no_cjk_or_private_paths():
    root = Path(__file__).resolve().parents[1]
    suffixes = {".py", ".md", ".toml", ".json", ".txt", ".sh", ".yml", ".yaml", ".in"}
    ignored = {".git", ".venv", "venv", "__pycache__", "build", "dist", "outputs", ".cache"}
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    failures = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ignored.intersection(relative.parts) or any(p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.is_file() and path.suffix in suffixes:
            text = path.read_text(encoding="utf-8")
            if cjk.search(text) or ("/" + "geniusland" + "/") in text:
                failures.append(str(relative))
    assert not failures, f"Nonportable or non-English public text: {failures}"
