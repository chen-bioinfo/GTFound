"""Configure English console and optional file logging for legacy callers."""

import logging
from pathlib import Path


def logger_init(log_file_name="gtfound-base", log_level=logging.INFO, log_dir=None):
    handlers = [logging.StreamHandler()]
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(directory / f"{log_file_name}.log", encoding="utf-8"))
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
