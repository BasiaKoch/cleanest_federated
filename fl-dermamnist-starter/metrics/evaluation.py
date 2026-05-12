from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix, roc_auc_score,
)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


def _jsonable_metrics(metrics: Dict):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def evaluate_model(model, dataloader, device, num_classes: int) -> Dict:
    model.eval()
    losses, preds, targets, probs = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device).view(-1).long()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item() * y.numel())
            p = F.softmax(logits, dim=1)
            probs.append(p.detach().cpu().numpy())
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    y_true = np.concatenate(targets) if targets else np.array([], dtype=int)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=int)
    y_prob = np.concatenate(probs) if probs else np.zeros((0, num_classes))

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_acc = []
    for c in labels:
        mask = y_true == c
        per_class_acc.append(float((y_pred[mask] == c).mean()) if mask.any() else np.nan)

    present_classes = sorted(np.unique(y_true).astype(int).tolist())
    macro_auroc = np.nan
    if len(present_classes) == num_classes:
        try:
            macro_auroc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=labels))
        except ValueError as exc:
            warnings.warn(f'AUROC failed: {exc}')
    else:
        warnings.warn(f'AUROC set to NaN because present classes are {present_classes}, expected {labels}')

    total = max(len(y_true), 1)
    metrics = {
        'loss': float(sum(losses) / total),
        'accuracy': float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        'per_class_f1': per_class_f1.astype(float).tolist(),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'per_class_recall': per_class_recall.astype(float).tolist(),
        'per_class_precision': per_class_precision.astype(float).tolist(),
        'per_class_accuracy': [float(x) if not np.isnan(x) else np.nan for x in per_class_acc],
        'worst_class_recall': float(np.nanmin(per_class_recall)),
        'worst_class_f1': float(np.nanmin(per_class_f1)),
        'confusion_matrix': cm,
        'macro_auroc': macro_auroc,
        'present_classes': present_classes,
    }
    return metrics


def plot_confusion_matrix(cm, class_names, save_path, title='', normalize=True):
    arr = np.asarray(cm, dtype=float)
    if normalize:
        arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns is not None:
        sns.heatmap(arr, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    else:
        im = ax.imshow(arr, cmap='Blues')
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title or 'Confusion Matrix')
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_per_class_metrics(metrics_dict, class_names, save_path, title=''):
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, metrics_dict['per_class_f1'], width, label='F1')
    ax.bar(x, metrics_dict['per_class_recall'], width, label='Recall')
    ax.bar(x + width, metrics_dict['per_class_precision'], width, label='Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title(title or 'Per-class Metrics')
    melanoma = [i for i, name in enumerate(class_names) if 'melanoma' in name.lower()]
    if melanoma:
        i = melanoma[0]
        ax.scatter([i], [metrics_dict['per_class_recall'][i]], marker='*', s=120, color='black', zorder=5, label='Melanoma')
    ax.legend()
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_training_curves(history_df, save_path, title=''):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if 'balanced_accuracy' in history_df:
        ax1.plot(history_df['round'], history_df['balanced_accuracy'], label='Balanced accuracy')
    if 'accuracy' in history_df:
        ax1.plot(history_df['round'], history_df['accuracy'], label='Accuracy')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    if 'loss' in history_df:
        ax2.plot(history_df['round'], history_df['loss'], linestyle='--', label='Loss')
    ax2.set_ylabel('Loss')
    ax1.set_title(title or 'Training Curves')
    ax1.legend(loc='upper left')
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


class MetricsLogger:
    def __init__(self, save_dir, class_names=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or []
        self.rows = []

    def log_round(self, round_num, metrics_dict):
        row = {'round': round_num}
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
        self.rows.append(row)

    def save(self):
        df = self.get_dataframe()
        df.to_csv(self.save_dir / 'metrics_history.csv', index=False)
        return df

    def get_dataframe(self):
        return pd.DataFrame(self.rows)

    def plot_all(self):
        df = self.get_dataframe()
        if df.empty:
            return []
        outputs = []
        outputs.append(plot_training_curves(df, self.save_dir / 'training_curves.png'))
        metric_cols = [c for c in df.columns if c.startswith('f1_') or c.startswith('recall_')]
        if metric_cols:
            metric_df = df[['round'] + metric_cols].set_index('round')
            fig, ax = plt.subplots(figsize=(8, 4))
            metric_df.plot(ax=ax)
            ax.set_xlabel('Round')
            ax.set_ylabel('Metric')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
            fig.tight_layout()
            out = self.save_dir / 'per_class_metrics_history.png'
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            outputs.append(fig)
        return outputs
