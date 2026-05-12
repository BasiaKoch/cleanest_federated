from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any, Dict
import yaml


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = merge_configs(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        override = yaml.safe_load(f) or {}
    base_path = path.parent / 'base.yaml'
    if path.name != 'base.yaml' and base_path.exists():
        with open(base_path, 'r', encoding='utf-8') as f:
            base = yaml.safe_load(f) or {}
        cfg = merge_configs(base, override)
    else:
        cfg = override
    cfg.setdefault('_config_stem', path.stem)
    return cfg


def config_to_experiment_name(config: Dict[str, Any]) -> str:
    stem = config.get('_config_stem', 'experiment')
    seed = config.get('misc', {}).get('seed', 42)
    local_epochs = config.get('federation', {}).get('local_epochs', 5)
    return f'{stem}_E{local_epochs}_s{seed}'


def config_to_flat_dict(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    flat = {}
    for key, val in config.items():
        new_key = f'{prefix}.{key}' if prefix else key
        if isinstance(val, dict):
            flat.update(config_to_flat_dict(val, new_key))
        else:
            flat[new_key] = val
    return flat
