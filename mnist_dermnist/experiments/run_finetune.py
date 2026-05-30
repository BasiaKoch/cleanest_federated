"""Per-client local fine-tuning baseline for the small-hospital case study.

Loads a federated checkpoint (FedAvg or FedProx at its best-validation
round) and adapts it to a single client's local data by a short burst of
local SGD. Evaluates the adapted model on the *global* test set so the
adapted-vs-federated comparison is directly comparable to every other
result in the dissertation.

This implements the canonical local-adaptation baseline from the
personalised-FL literature: Yu et al. (2022) "Salvaging Federated
Learning by Local Adaptation"; Collins et al. (2021) FedRep §5;
Marfoq et al. (2021); see also the discussion in Kairouz et al. (2021)
§5. The protocol is intentionally minimal --- a small number of local
epochs at a reduced learning rate --- so that the comparison isolates
*the value of starting from the federated initialisation*, rather than
the value of any sophisticated personalisation algorithm. A larger gap
would indicate that the federated global model leaves room for
personalisation; a smaller gap would indicate that the federated
algorithm has already done most of the per-client adaptation work.

Why this matters for the FedProx-vs-FedAvg comparison
-----------------------------------------------------
If FedProx + local FT ≈ FedProx, the proximal term has \\emph{already}
captured the per-client adaptation that local FT would otherwise add ---
a non-trivial mechanism claim that to the author's knowledge has not
been shown cleanly on a class-disjoint medical classification task.
If FedProx + local FT \\gg FedProx, FedProx is one ingredient but not
sufficient on its own for under-represented clients, motivating
personalised-FL methods (Collins 2021, Marfoq 2021) as a next step.

Usage
-----
PYTHONPATH=. python -m mnist_dermnist.experiments.run_finetune \\
    --checkpoint mnist_dermnist/results/two_client_90_10_rare_stress/\\
                  best_state_fedavg_mu0.0_E20_s42.pt \\
    --partition two_client_90_10_rare_stress \\
    --client-id 1 --num-epochs 5 --lr 0.001 \\
    --seed 42 --device cuda \\
    --out-dir mnist_dermnist/results/small_hospital_finetune
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

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.data.partition import (
    balanced_paired_7_clients,
    balanced_specialist_7_clients,
    dirichlet_7_clients,
    iid_7_clients,
    specialist_7_clients,
    two_client_90_10_rare_stress,
)
from mnist_dermnist.fl.runtime_provenance import collect_runtime_provenance, utc_now_iso
from mnist_dermnist.models import DermMNISTCNN


PARTITIONERS = {
    "balanced_paired_7_clients": balanced_paired_7_clients,
    "balanced_specialist_7_clients": balanced_specialist_7_clients,
    "specialist_7_clients": specialist_7_clients,
    "iid_7_clients": iid_7_clients,
    "two_client_90_10_rare_stress": two_client_90_10_rare_stress,
    "dirichlet_alpha01_7_clients": lambda y, seed=42: dirichlet_7_clients(y, seed=seed, alpha=0.1),
}


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, num_classes: int = 7) -> dict:
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
        description="Per-client local fine-tuning from a federated checkpoint "
                    "(Yu et al. 2022 protocol).")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a state_dict saved by run_one_flower.py "
                         "with --save-best-checkpoint, e.g. "
                         "'best_state_fedavg_mu0.0_E20_s42.pt'.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partition", choices=list(PARTITIONERS),
                    default="two_client_90_10_rare_stress")
    ap.add_argument("--client-id", type=int, required=True)
    ap.add_argument("--num-epochs", type=int, default=5,
                    help="Local fine-tuning epochs. Default 5 follows "
                         "the Yu et al. (2022) §4 protocol.")
    ap.add_argument("--lr", type=float, default=0.001,
                    help="Reduced learning rate for fine-tuning. Default "
                         "0.001 (10x smaller than the main-training lr).")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path",
                    default=str(Path(__file__).resolve().parents[2] / "dermamnist_64.npz"))
    ap.add_argument("--out-dir",
                    default="mnist_dermnist/results/small_hospital_finetune")
    ap.add_argument("--tag", default=None,
                    help="Optional explicit filename stem. If absent, derived "
                         "from the checkpoint filename + client id.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Data --------------------------------------------------------------
    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
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
    per_class = np.bincount(labels[client_indices], minlength=7).tolist()

    train_loader = DataLoader(client_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val,  batch_size=128, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

    # --- Model: load checkpoint -------------------------------------------
    model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    # Allow both raw state_dicts and wrapped {"state_dict": ...} containers.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  partition={args.partition} client_id={args.client_id} "
          f"n_train={n_client} per_class={per_class}")

    # --- Baseline: evaluate the un-finetuned global model first -----------
    # This gives us the federated (pre-FT) baseline using the exact same
    # eval pipeline as the FT result, so they are bit-comparable.
    pre_ft_metrics = evaluate(model, test_loader, device)
    print(f"  pre-FT (federated checkpoint): test_macro_f1={pre_ft_metrics['macro_f1']:.4f}")

    # --- Local fine-tuning loop -------------------------------------------
    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    crit  = nn.CrossEntropyLoss()

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
        history.append({
            "ft_epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            **val_metrics,
        })
        print(f"  ft_epoch {epoch}: train_loss={running/max(n_batches,1):.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}",
              flush=True)
    elapsed = time.time() - t0

    # --- Post-FT test ------------------------------------------------------
    post_ft_metrics = evaluate(model, test_loader, device)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem
    if args.tag:
        stem = args.tag
    else:
        stem = f"finetune_{ckpt_name.replace('best_state_', '')}_c{args.client_id}"
    out_path = out_dir / f"test_at_best_{stem}.json"
    result = {
        "regime": "federated_plus_local_finetune",
        "starting_checkpoint": str(args.checkpoint),
        "starting_checkpoint_name": ckpt_name,
        "partition": args.partition,
        "client_id": args.client_id,
        "client_n_train": n_client,
        "client_per_class_counts": per_class,
        "seed": args.seed,
        "ft_num_epochs": args.num_epochs,
        "ft_lr": args.lr,
        "ft_momentum": args.momentum,
        "ft_weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "device": args.device, "elapsed_s": elapsed,
        # Pre / post fine-tuning metrics on the same global test set:
        "pre_ft_macro_f1": pre_ft_metrics["macro_f1"],
        "pre_ft_per_class_f1": pre_ft_metrics["per_class_f1"],
        "pre_ft_accuracy": pre_ft_metrics["accuracy"],
        "pre_ft_balanced_accuracy": pre_ft_metrics["balanced_accuracy"],
        # Post-FT (these become the main reported numbers).
        **post_ft_metrics,
        "personalisation_gap_macro_f1": (
            post_ft_metrics["macro_f1"] - pre_ft_metrics["macro_f1"]),
        **collect_runtime_provenance(run_started_at),
    }
    out_path.write_text(json.dumps(result, indent=2))

    hist_path = out_dir / f"history_{stem}.json"
    hist_path.write_text(json.dumps(history, indent=2))

    gap = result["personalisation_gap_macro_f1"]
    print(f"\nFine-tuning result:")
    print(f"  pre-FT  test_macro_f1 = {pre_ft_metrics['macro_f1']:.4f}")
    print(f"  post-FT test_macro_f1 = {post_ft_metrics['macro_f1']:.4f}")
    print(f"  personalisation gap   = {gap:+.4f}")
    print(f"  per-class F1 (post-FT)= {[f'{x:.3f}' for x in post_ft_metrics['per_class_f1']]}")
    print(f"  elapsed               = {elapsed:.1f}s")
    print(f"\nWrote {out_path}")
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
