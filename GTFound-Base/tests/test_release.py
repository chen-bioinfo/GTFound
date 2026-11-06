"""Check public source syntax, English text, and portable paths."""

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_source_is_english_and_portable():
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    suffixes = {".py", ".md", ".json", ".txt", ".ini", ".toml", ".yaml", ".yml", ".sh"}
    ignored = {"__pycache__", ".pytest_cache", ".venv", "venv", "outputs", "data", "cache"}
    for path in ROOT.rglob("*"):
        if ignored.intersection(path.relative_to(ROOT).parts):
            continue
        if not path.is_file() or path.suffix not in suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        assert not cjk.search(text), str(path.relative_to(ROOT))
        assert ("/" + "geniusland" + "/") not in text, str(path.relative_to(ROOT))
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
