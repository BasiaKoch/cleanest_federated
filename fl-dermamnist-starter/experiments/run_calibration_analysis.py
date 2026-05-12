from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.download import load_dermamnist, get_dataset_info, get_class_distribution
from losses.logit_adjustment import post_hoc_logit_adjustment
from metrics.evaluation import evaluate_model
from models import get_model


def evaluate_with_adjustment(model, dataloader, device, num_classes, class_priors, tau):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x.to(device))
            adjusted = post_hoc_logit_adjustment(logits, class_priors.to(device), tau=tau)
            preds.append(adjusted.argmax(1).cpu())
            targets.append(y.view(-1).long())
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    labels = list(range(num_classes))
    return {
        'tau': tau,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'worst_class_f1': min(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)),
        'per_class_recall': recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', default='simple_cnn')
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = Path('results/analysis/logit_adjustment')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    train, _, test = load_dermamnist(size=args.size)
    info = get_dataset_info('dermamnist')
    model = get_model(args.model, info['input_channels'], info['num_classes'], image_size=args.size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=2)
    counts = get_class_distribution(train)
    count_arr = torch.tensor([counts.get(i, 0) for i in range(info['num_classes'])], dtype=torch.float32)
    priors = count_arr / count_arr.sum()
    rows = [evaluate_with_adjustment(model, test_loader, device, info['num_classes'], priors, tau) for tau in [0.5, 1.0, 1.5, 2.0]]
    baseline = evaluate_model(model, test_loader, device, info['num_classes'])
    pd.DataFrame(rows).to_csv(out / 'logit_adjustment_results.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    df = pd.DataFrame(rows)
    ax.plot(df['tau'], df['balanced_accuracy'], marker='o', label='Adjusted')
    ax.axhline(baseline['balanced_accuracy'], linestyle='--', color='black', label='Baseline')
    ax.set_xlabel('Tau')
    ax.set_ylabel('Balanced accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'balanced_accuracy_vs_tau.png', dpi=150)
    plt.close(fig)
    print(df[['tau', 'balanced_accuracy', 'macro_f1', 'worst_class_f1']].to_string(index=False))


if __name__ == '__main__':
    main()
