"""Centralised (non-federated) training baseline on DermaMNIST.

Trains a single DermMNISTCNN on the *pooled* training set, with the same
hyperparameters as the FL runs except for the absence of FL-specific
machinery (no rounds, no aggregation, no per-client partitioning).
The output is directly comparable to the FedAvg/FedProx test results
from the headline sweep.

Run on HPC:
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_centralised \\
        --seed 42 --num-epochs 50 --device cuda \\
        --out-dir mnist_dermnist/results/centralised
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.models import DermMNISTCNN


def set_seed(seed: int) -> None:
    # Match fl/server_loop.py:set_all_seeds so centralised runs are as
    # reproducible as the FL runs (audit P2 fix). Python's `random`
    # module is seeded too, even though centralised doesn't use it
    # directly, in case downstream sklearn or third-party utilities do.
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, num_classes: int = 7):
    """Same metric set as fl/evaluation.py, with explicit `labels` so a
    class absent from predictions still contributes 0 to macro-F1
    rather than being silently dropped (audit P2 fix)."""
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
    model.eval()
    ys, preds, total_loss, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    labels = list(range(num_classes))
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device).view(-1).long()
            logits = model(x)
            total_loss += float(crit(logits, y).item())
            ys.append(y.cpu().numpy()); preds.append(logits.argmax(1).cpu().numpy())
            n += len(y)
    y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
    return {
        "loss": total_loss / n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None,
                                 labels=labels, zero_division=0).tolist(),
        "n": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-epochs", type=int, default=50,
                    help="Roughly matches R*E / fraction-fit budget of FL runs")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path",
                    default="/home/bk489/federated_clean/cleanest_federated/dermamnist_64.npz")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/centralised")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=128, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test,  batch_size=128, shuffle=False, num_workers=0)

    model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    crit  = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state  = None
    best_epoch  = -1
    history = []
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device).view(-1).long()
            optim.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); optim.step()
            running += float(loss.item()); n_batches += 1
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / n_batches, **val_metrics})
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch  = epoch
        print(f"  epoch {epoch:>3}: train_loss={running/n_batches:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  (best={best_val_f1:.4f} @ ep{best_epoch})",
              flush=True)

    elapsed = time.time() - t0

    # Test at best-val epoch
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "regime": "centralised",
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "selected_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
        "device": args.device, "elapsed_s": elapsed,
        **test_metrics,
    }
    out_path = out_dir / f"centralised_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))

    hist_path = out_dir / f"history_centralised_seed{args.seed}.json"
    hist_path.write_text(json.dumps(history, indent=2))

    print(f"\nCentralised result @ seed={args.seed}:")
    print(f"  test_macro_f1     = {test_metrics['macro_f1']:.4f}")
    print(f"  test_balanced_acc = {test_metrics['balanced_accuracy']:.4f}")
    print(f"  test_accuracy     = {test_metrics['accuracy']:.4f}")
    print(f"  selected at epoch = {best_epoch} of {args.num_epochs}")
    print(f"  elapsed           = {elapsed/60:.1f} min")
    print(f"\nWrote {out_path}")
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
