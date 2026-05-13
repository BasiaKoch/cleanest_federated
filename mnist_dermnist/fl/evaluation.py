"""Global evaluation helpers — macro-F1, balanced accuracy, per-class F1, loss.

Same signature as the rest of the project. Used for both the round-by-round
validation pass and the single final-test pass at the best-val checkpoint.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | str, num_classes: int = 7) -> Dict:
    """Run inference and compute classification metrics.

    Returns dict with:
      - loss               : cross-entropy
      - accuracy           : raw acc
      - balanced_accuracy  : sklearn balanced_accuracy_score
      - macro_f1           : sklearn f1_score(average='macro')
      - per_class_f1       : list of length num_classes
      - n                  : total samples seen
    """
    device = torch.device(device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_total = 0
    preds_all: List[int] = []
    targets_all: List[int] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).view(-1).long()
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        n_total += y.size(0)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().numpy().tolist())
        targets_all.extend(y.cpu().numpy().tolist())

    preds_arr = np.asarray(preds_all)
    targets_arr = np.asarray(targets_all)
    labels = list(range(num_classes))

    return {
        "loss": total_loss / max(n_total, 1),
        "accuracy": float(np.mean(preds_arr == targets_arr)) if n_total else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(targets_arr, preds_arr)) if n_total else float("nan"),
        "macro_f1": float(f1_score(targets_arr, preds_arr, average="macro", labels=labels, zero_division=0)) if n_total else float("nan"),
        "per_class_f1": f1_score(targets_arr, preds_arr, average=None, labels=labels, zero_division=0).tolist() if n_total else [float("nan")] * num_classes,
        "n": int(n_total),
    }
