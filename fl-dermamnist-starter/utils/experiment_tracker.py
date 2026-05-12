from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import time
import platform
import yaml
import torch


def make_jsonable(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    return obj


class ExperimentTracker:
    def __init__(self, experiment_name: str, save_dir: str | Path, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        with open(self.save_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(make_jsonable(config), f, sort_keys=False)
        self.save_system_info()

    def save_system_info(self) -> None:
        info = {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        with open(self.save_dir / 'system_info.json', 'w', encoding='utf-8') as f:
            json.dump(make_jsonable(info), f, indent=2)

    def log_final(self, metrics: Dict[str, Any], filename: str = 'global_test_metrics.json') -> None:
        metrics = dict(metrics)
        metrics['elapsed_seconds'] = time.time() - self.start_time
        with open(self.save_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(make_jsonable(metrics), f, indent=2)
