"""Local-only (no-federation) baseline for the small-hospital case study.

Trains a single ``DermMNISTCNN`` on a *single client's* local data only,
matched to the federation's cumulative SGD-step budget so the comparison
isolates the effect of data availability rather than compute. Evaluates
at the best-validation checkpoint on the *global* validation set, then
reports test-set metrics on the *global* test set --- the same model-
selection and evaluation protocol used by the federated runs.

Why this exists
----------------
The 2-client 90/10 case study (Section sec:small-hospital) asks whether
FedProx specifically helps small institutions with rare-class data. To
quantify the "value of federation" for each client, we need a baseline
that answers: *what would each client achieve if it trained alone on
its own data?* This script produces that baseline. It is the standard
local-only comparison used by Sheller et al. (2018, 2020), Roth et al.
(2020), Pati et al. (2022), and the FL surveys (Kairouz et al. 2021).

Key design choices
------------------
1.  Compute budget is matched to the federation's cumulative SGD steps.
    A federation with ``R`` rounds and ``E`` local epochs per round
    performs ``R*E`` epochs of local SGD on each participating client.
    To keep the comparison clean, we run ``R*E = 3000`` epochs of
    local-only training. Differences between local-only and federated
    therefore cannot be attributed to one regime having more compute.

2.  Validation and test sets are the *global* DermaMNIST validation/test
    splits, same as the federated runs. Selecting the model at best-
    validation macro-F1 (rather than train loss or final epoch) matches
    the federated protocol exactly.

3.  No data augmentation, no learning-rate schedule, no early stopping
    other than best-validation checkpointing --- all to keep this code
    path bit-comparable to the federated local trainer.

Usage
-----
PYTHONPATH=. python -m fl_dermamnist.experiments.run_local_only \\
    --seed 42 --partition two_client_90_10_rare_stress \\
    --client-id 0 --num-epochs 3000 --device cuda \\
    --out-dir fl_dermamnist/results/small_hospital_local_only

Output: ``test_at_best_local_only_c<cid>_E<num_epochs>_s<seed>.json``
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from fl_dermamnist.data.load import load_dermmnist
from fl_dermamnist.data.partition import (
    balanced_paired_7_clients,
    balanced_specialist_7_clients,
    dirichlet_7_clients,
    iid_7_clients,
    specialist_7_clients,
    two_client_90_10_rare_stress,
)
from fl_dermamnist.fl.runtime_provenance import collect_runtime_provenance, utc_now_iso
from fl_dermamnist.models import DermMNISTCNN


PARTITIONERS = {
    "balanced_paired_7_clients": balanced_paired_7_clients,
    "balanced_specialist_7_clients": balanced_specialist_7_clients,
    "specialist_7_clients": specialist_7_clients,
    "iid_7_clients": iid_7_clients,
    "two_client_90_10_rare_stress": two_client_90_10_rare_stress,
    "dirichlet_alpha01_7_clients": lambda y, seed=42: dirichlet_7_clients(y, seed=seed, alpha=0.1),
}


def set_seed(seed: int) -> None:
    """Match ``run_centralised.set_seed`` for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, num_classes: int = 7) -> dict:
    """Identical metric set to ``run_centralised.evaluate``."""
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
    run_started_at = utc_now_iso()
    ap = argparse.ArgumentParser(
        description="Local-only baseline: single-client training, "
                    "no federation, matched compute budget.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partition", choices=list(PARTITIONERS),
                    default="two_client_90_10_rare_stress")
    ap.add_argument("--client-id", type=int, required=True,
                    help="Which client's local data to train on")
    ap.add_argument("--num-epochs", type=int, default=3000,
                    help="Total local-SGD epochs. Default matches the "
                         "federation's cumulative R*E budget (150*20).")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="Run validation every N epochs. 20 matches the "
                         "federated round granularity (one eval per R*E "
                         "block), so the validation cadence is identical "
                         "to the federation's 150-eval timeline.")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path",
                    default=str(Path(__file__).resolve().parents[2] / "dermamnist_64.npz"))
    ap.add_argument("--out-dir",
                    default="fl_dermamnist/results/small_hospital_local_only")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Data --------------------------------------------------------------
    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    # Run the partitioner to get this client's indices into the train set.
    # Partitioners take labels and return (clients, spec_df).
    labels = np.array([int(y) for _, y in train])
    partition_fn = PARTITIONERS[args.partition]
    clients, _ = partition_fn(labels, seed=args.seed)
    if args.client_id < 0 or args.client_id >= len(clients):
        raise ValueError(
            f"client-id {args.client_id} out of range for partition "
            f"{args.partition} with {len(clients)} clients")
    client_indices = clients[args.client_id]
    client_train = Subset(train, client_indices)
    n_client = len(client_indices)

    # Per-class count for this client, for provenance.
    per_class = np.bincount(labels[client_indices], minlength=7).tolist()

    train_loader = DataLoader(client_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val,  batch_size=128, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

    print(f"Local-only training: partition={args.partition} client_id={args.client_id}")
    print(f"  client n_train={n_client}, per-class counts={per_class}")

    # --- Model / optimiser -------------------------------------------------
    model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    crit  = nn.CrossEntropyLoss()

    # --- Training loop with best-val checkpointing ------------------------
    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
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

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            val_metrics = evaluate(model, val_loader, device)
            history.append({
                "epoch": epoch,
                "train_loss": running / max(n_batches, 1),
                **val_metrics,
            })
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state  = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
                best_epoch = epoch
            print(f"  epoch {epoch:>4}: train_loss={running/max(n_batches,1):.4f}  "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
                  f"(best={best_val_f1:.4f} @ ep{best_epoch})",
                  flush=True)

    elapsed = time.time() - t0

    # --- Test at best-val checkpoint --------------------------------------
    if best_state is None:
        raise RuntimeError("No best state recorded; check --eval-every and --num-epochs.")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "regime": "local_only",
        "partition": args.partition,
        "client_id": args.client_id,
        "client_n_train": n_client,
        "client_per_class_counts": per_class,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "selected_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "lr": args.lr, "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "device": args.device, "elapsed_s": elapsed,
        **test_metrics,
        **collect_runtime_provenance(run_started_at),
    }
    tag = f"test_at_best_local_only_c{args.client_id}_E{args.num_epochs}_s{args.seed}"
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))

    hist_path = out_dir / f"history_local_only_c{args.client_id}_E{args.num_epochs}_s{args.seed}.json"
    hist_path.write_text(json.dumps(history, indent=2))

    print(f"\nLocal-only result @ partition={args.partition} client={args.client_id} seed={args.seed}:")
    print(f"  test_macro_f1     = {test_metrics['macro_f1']:.4f}")
    print(f"  test_balanced_acc = {test_metrics['balanced_accuracy']:.4f}")
    print(f"  test_accuracy     = {test_metrics['accuracy']:.4f}")
    print(f"  per-class F1      = {[f'{x:.3f}' for x in test_metrics['per_class_f1']]}")
    print(f"  selected at epoch = {best_epoch} of {args.num_epochs}")
    print(f"  elapsed           = {elapsed/60:.1f} min")
    print(f"\nWrote {out_path}")
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
