"""Tiny YAML config loader for mnist_dermnist.

Kept intentionally minimal — no schema, no deep-merging chains. Just load the
YAML, return a plain dict. Code that consumes it is responsible for `.get(...)`
defaults and validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "mnist_dermnist/configs/base.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_from_config(cfg: Dict[str, Any]):
    """Instantiate a model based on cfg['model']."""
    from mnist_dermnist.models import get_model
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name")
    if not name:
        raise ValueError("cfg['model']['name'] is required")
    return get_model(name, **{k: v for k, v in model_cfg.items() if k != "name"})
