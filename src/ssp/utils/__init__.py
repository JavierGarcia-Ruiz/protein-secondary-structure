"""Utilities: paths, configuration, seeding and logging."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``other`` into a copy of ``base``."""
    out = dict(base)
    for k, v in other.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; a top-level ``base:`` key is merged recursively first."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_update(base, cfg)
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_logger(name: str = "ssp", log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root ``ssp`` logger (stdout + optional file) and return ``name``."""
    root = logging.getLogger("ssp")
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        root.setLevel(level)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
    return logging.getLogger(name)
