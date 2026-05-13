"""FedAvg server (Flower simulation entry point).

Mirrors scripts/seg_fedavg/server.py from the reference repo but uses
fl.simulation.start_simulation (single-machine) instead of TCP networking,
and PyTorch + classification instead of TensorFlow + segmentation.

Usage:
    python -m new_clean_fed.scripts.fedavg.server
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import List

import flwr as fl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Subset

from new_clean_fed.configs import fedavg_config as cfg
from new_clean_fed.src.fedavg.client import FedAvgClient, numpy_to_state_dict
from new_clean_fed.src.fedavg.model import CNNClassifier
from new_clean_fed.src.fedavg.prepare_data import (
    load_dermamnist, dirichlet_partition, print_distribution,
)


def evaluate_global(parameters: List[np.ndarray], val_loader: DataLoader, device: str) -> dict:
    model = CNNClassifier(in_channels=3, num_classes=7, dropout=cfg.DROPOUT).to(device)
    numpy_to_state_dict(model, parameters)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y).item()) * y.size(0)
            total += y.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_targets))),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(all_targets, all_preds)),
        "per_class_f1": f1_score(all_targets, all_preds, average=None, zero_division=0).tolist(),
    }


def main():
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print(f"=== FedAvg server ===")
    print(f"  Clients: {cfg.NUM_CLIENTS} | Rounds: {cfg.NUM_ROUNDS} | E: {cfg.NUM_LOCAL_EPOCHS}")
    print(f"  Fraction-fit: {cfg.FRACTION_FIT} | Partition: {cfg.PARTITION_STRATEGY}(α={cfg.DIRICHLET_ALPHA})")
    print(f"  Device: {cfg.DEVICE}")

    train_ds, val_ds, test_ds = load_dermamnist(cfg.NPZ_PATH)
    print(f"  Sizes — train: {len(train_ds)} val: {len(val_ds)} test: {len(test_ds)}")

    partitions = dirichlet_partition(
        train_ds, num_clients=cfg.NUM_CLIENTS, alpha=cfg.DIRICHLET_ALPHA,
        seed=cfg.SEED, min_samples_per_client=cfg.MIN_SAMPLES_PER_CLIENT,
    )
    print()
    print_distribution(train_ds, partitions)
    print()

    # Per-client loaders (deterministic per-client generator → reproducible mini-batch order)
    def make_client_fn():
        train_loaders, val_loaders = [], []
        for cid, idxs in enumerate(partitions):
            sub = Subset(train_ds, idxs)
            gen = torch.Generator().manual_seed(cfg.SEED * 10000 + cid)
            train_loaders.append(DataLoader(sub, batch_size=cfg.BATCH_SIZE, shuffle=True, generator=gen))
            # Per-client validation is degenerate under strong non-IID; use central val
            val_loaders.append(DataLoader(val_ds, batch_size=128, shuffle=False))

        def client_fn(cid: str):
            cid_int = int(cid)
            return FedAvgClient(
                cid=cid_int,
                train_loader=train_loaders[cid_int],
                val_loader=val_loaders[cid_int],
                device=cfg.DEVICE,
                num_local_epochs=cfg.NUM_LOCAL_EPOCHS,
                lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY,
                dropout=cfg.DROPOUT,
            ).to_client()
        return client_fn

    central_val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    def fit_metrics_agg(metrics_list):
        total = sum(n for n, _ in metrics_list) or 1
        return {"train_loss_weighted": sum(n * m.get("train_loss", 0.0) for n, m in metrics_list) / total}

    def evaluate_fn(server_round, params, _config):
        m = evaluate_global(params, central_val_loader, cfg.DEVICE)
        return m["loss"], {k: v for k, v in m.items() if k != "per_class_f1"}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.FRACTION_FIT,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, math.ceil(cfg.NUM_CLIENTS * cfg.FRACTION_FIT)),
        min_evaluate_clients=0,
        min_available_clients=cfg.NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(),
        num_clients=cfg.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=cfg.NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if cfg.DEVICE == "cpu" else 0.1},
    )

    out_dir = Path(cfg.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump per-round metrics
    rows = {}
    for k, vals in history.metrics_centralized.items():
        for rnd, v in vals:
            rows.setdefault(int(rnd), {"round": int(rnd)})[k] = float(v)
    for rnd, v in history.losses_centralized:
        rows.setdefault(int(rnd), {"round": int(rnd)})["loss"] = float(v)
    if hasattr(history, "metrics_distributed_fit"):
        for k, vals in history.metrics_distributed_fit.items():
            for rnd, v in vals:
                rows.setdefault(int(rnd), {"round": int(rnd)})[k] = float(v)
    df = pd.DataFrame([rows[k] for k in sorted(rows)])
    df.to_csv(out_dir / "metrics_history.csv", index=False)

    # Final test using the last aggregated params (Flower stores final in history)
    if hasattr(history, "parameters") and history.parameters is not None:
        final_params = fl.common.parameters_to_ndarrays(history.parameters)
    else:
        # Pull from final round's centralized eval — re-evaluating with current best is sufficient
        final_params = None

    # In the simulation API the final aggregated params aren't directly exposed.
    # The centralized eval at round N is the best proxy; save the per-round metrics.
    print()
    print("Final round metrics (centralized validation):")
    if not df.empty:
        print(df.tail(1).to_string(index=False))
    print()
    print(f"Wrote {out_dir / 'metrics_history.csv'}")


if __name__ == "__main__":
    main()
