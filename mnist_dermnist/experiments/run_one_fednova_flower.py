"""Flower-framework FedNova entry point.

Mirrors `run_one_flower.py` exactly, except that the server strategy is
`PairedFedNovaStrategy` (Wang et al., 2020, NeurIPS) instead of FedAvg.
Use this script to obtain a third comparator alongside FedAvg and FedProx
for the system-heterogeneity section, where heterogeneous local-step
counts create objective inconsistency that FedNova's normalised
aggregation is specifically designed to address.

The local client is `FlClientFedNova` (in `mnist_dermnist/fl_flower/`),
which is identical to the standard `FlClient` except that it reports
`tau` (the local SGD step count) via the fit-metrics dict so the
strategy can compute the momentum-aware normaliser — the L1 norm of
the cumulative-momentum series (Wang et al. 2020, §3.3):

    a_i = sum_{j=1..tau_i} (1 - m^j) / (1 - m)
        = (tau_i*(1 - m) - m*(1 - m^{tau_i})) / (1 - m)^2

which reduces to a_i = tau_i at m = 0 (vanilla SGD). NB: an earlier
draft of this comment used `a_i = (1 - m^{tau_i}) / (1 - m)`, the last
element of the cumulative series; that is NOT the L1 norm and gives
the wrong normalisation under momentum > 0. See
`mnist_dermnist/fl_flower/strategy_fednova.py:fednova_normaliser` for
the implementation.

CLI is identical to `run_one_flower.py`. Algorithm is hard-coded to
"fednova" (no proximal term; same local objective as FedAvg, but with
normalised aggregation rule). Setting `--algorithm fednova` is the only
external user-facing change.

Usage:
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_fednova_flower \\
        --seed 42 --num-rounds 150 --local-epochs 20 \\
        --partition balanced_paired_7_clients \\
        --device cuda \\
        --system-het-mode random_stragglers --straggler-fraction 0.5 \\
        --out-dir mnist_dermnist/results/system_het_fednova
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.data.partition import (
    balanced_paired_7_clients,
    balanced_specialist_7_clients,
    dirichlet_7_clients,
    iid_7_clients,
    medical_skew_7_clients,
    quantity_skew_improved,
    simple_pathological_3_clients,
    specialist_7_clients,
    two_client_50_50_label_skew_only,
    two_client_50_50_stratified_iid,
    two_client_70_30_rare_enriched,
    two_client_86_14_quantity_only_stratified,
    two_client_90_10_rare_stress,
)
from mnist_dermnist.fl.evaluation import evaluate
from mnist_dermnist.fl.runtime_provenance import collect_runtime_provenance, utc_now_iso
from mnist_dermnist.fl.system_het import SystemHetConfig, build_epoch_schedule
from mnist_dermnist.fl_flower.client import state_dict_to_numpy, numpy_to_state_dict
from mnist_dermnist.fl_flower.client_fednova import FlClientFedNova
from mnist_dermnist.fl_flower.strategy_fednova import PairedFedNovaStrategy
from mnist_dermnist.models import DermMNISTCNN


def _dir_a01(labels, seed=42):
    return dirichlet_7_clients(labels, seed=seed, alpha=0.1)


def _dir_a05(labels, seed=42):
    return dirichlet_7_clients(labels, seed=seed, alpha=0.5)


PARTITIONERS = {
    "medical_skew_7_clients": medical_skew_7_clients,
    "simple_pathological_3_clients": simple_pathological_3_clients,
    "balanced_specialist_7_clients": balanced_specialist_7_clients,
    "balanced_paired_7_clients": balanced_paired_7_clients,
    "specialist_7_clients": specialist_7_clients,
    "quantity_skew_improved": quantity_skew_improved,
    "iid_7_clients": iid_7_clients,
    "dirichlet_alpha01_7_clients": _dir_a01,
    "dirichlet_alpha05_7_clients": _dir_a05,
    "two_client_90_10_rare_stress": two_client_90_10_rare_stress,
    # Heterogeneity-ladder partitions (Levels 0-3; Level 4 above).
    "two_client_50_50_stratified_iid": two_client_50_50_stratified_iid,
    "two_client_86_14_quantity_only_stratified": two_client_86_14_quantity_only_stratified,
    "two_client_50_50_label_skew_only": two_client_50_50_label_skew_only,
    "two_client_70_30_rare_enriched": two_client_70_30_rare_enriched,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Flower-framework FedNova run (paired-fair with FedAvg/FedProx).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-epochs", type=int, default=20)
    ap.add_argument("--num-rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-per-client", type=str, default=None,
                    help="Per-client learning-rate override (e.g. "
                         "'0:0.01,1:0.005'). Tests Li 2020 §5.2 protocol "
                         "extension from asymmetric-E to asymmetric-LR. "
                         "NOTE: FedNova's normaliser derivation (Wang 2020) "
                         "assumes uniform LR — asymmetric LR is OUTSIDE the "
                         "regime FedNova proves correction for.")
    ap.add_argument("--momentum", type=float, default=0.9,
                    help="Local SGD momentum. Passed to FedNova normaliser.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--partition", choices=list(PARTITIONERS),
                    default="balanced_paired_7_clients")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path",
                    default=str(Path(__file__).resolve().parents[2] / "dermamnist_64.npz"))
    ap.add_argument("--out-dir", default="mnist_dermnist/results/system_het_fednova")
    ap.add_argument("--fraction-fit", type=float, default=1.0,
                    help="Fraction of clients sampled per round (C).")
    ap.add_argument("--system-het-mode",
                    choices=["uniform", "fixed_stragglers", "random_stragglers"],
                    default="uniform")
    ap.add_argument("--straggler-epochs", type=int, default=5)
    ap.add_argument("--fixed-straggler-ids", default=None)
    ap.add_argument("--straggler-fraction", type=float, default=0.5)
    ap.add_argument("--log-update-norms", action="store_true",
                    help="Capture per-(round, client) L2 update norm and "
                         "write client_update_norms_*.csv alongside the JSON.")
    # --- Mechanism probes for random-τ FedNova failure (thesis §5.5). ---
    ap.add_argument("--tau-clip-min", type=int, default=0,
                    help="Lower clamp on tau_i used in the FedNova normaliser "
                         "denominator. 0 disables (default). Recommended "
                         "diagnostic value: E_max // 2 (e.g. 10 for E=20). "
                         "Partial parameter deltas still enter aggregation "
                         "unchanged; only the 1/tau scaling is bounded.")
    ap.add_argument("--server-momentum", type=float, default=0.0,
                    help="Heavy-ball server-side momentum coefficient applied "
                         "to the FedNova pseudo-gradient (Hsu et al. 2019, "
                         "FedAvgM). 0.0 disables (default). Recommended: 0.9.")
    return ap


def main():
    run_started_at = utc_now_iso()
    args = build_parser().parse_args()
    seed = int(args.seed)

    # --- Reproducibility (incl. random.seed for paired-seed parity with the
    #     pure-PyTorch reference loop) ---
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_str = args.device
    device = torch.device(device_str)

    # --- Data + partition ---
    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
    partitioner = PARTITIONERS[args.partition]
    client_indices, _ = partitioner(train.labels, seed=seed)
    num_clients = len(client_indices)

    # --- System-het schedule ---
    fixed_ids = None
    if args.fixed_straggler_ids:
        fixed_ids = [int(x) for x in args.fixed_straggler_ids.split(",")]
    sh_cfg = SystemHetConfig(
        mode=args.system_het_mode,
        E_max=args.local_epochs,
        E_straggler=args.straggler_epochs,
        fixed_straggler_ids=fixed_ids,
        random_straggler_fraction=args.straggler_fraction,
    )
    epoch_schedule = build_epoch_schedule(
        sh_cfg, num_clients=num_clients,
        num_rounds=args.num_rounds, seed=seed,
    )

    # Parse --lr-per-client into {cid: lr} dict (asymmetric-LR experiment).
    lr_per_client_map: "dict[int, float] | None" = None
    if args.lr_per_client:
        lr_per_client_map = {}
        for spec in args.lr_per_client.split(","):
            cid_s, lr_s = spec.split(":")
            lr_per_client_map[int(cid_s.strip())] = float(lr_s.strip())
        if any(v <= 0 for v in lr_per_client_map.values()):
            raise ValueError("--lr-per-client values must be strictly positive.")

    # --- Global model init ---
    torch.manual_seed(seed)
    global_model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    initial_params = state_dict_to_numpy(global_model)

    best_val_macro_f1 = {"value": -1.0, "round": -1,
                         "params": [a.copy() for a in initial_params]}
    history_rows: List[Dict] = []
    # Populated post-simulation from history_obj.metrics_distributed_fit
    # so the FedNova CSV has the same `train_loss` column as the pure-
    # PyTorch path (audit fix B3). PairedFedNovaStrategy.aggregate_fit
    # already returns size-weighted train_loss in its metrics_agg dict;
    # we just need to wire it back to the per-round rows.
    train_loss_by_round: Dict[int, float] = {}

    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config):
        eval_model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
        numpy_to_state_dict(eval_model, parameters)
        metrics = evaluate(eval_model, val_loader, device, num_classes=7)

        # Skip round 0 to match the pure-PyTorch reference loop's
        # rounds-1..R convention; see run_one_flower.py for rationale.
        if int(server_round) >= 1:
            row = {
                "seed": seed,
                "algorithm": "fednova",
                "mu": 0.0,
                "local_epochs": args.local_epochs,
                "round": server_round,
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
            for c, f1c in enumerate(metrics["per_class_f1"]):
                row[f"val_f1_class_{c}"] = float(f1c)
            history_rows.append(row)

            if metrics["macro_f1"] > best_val_macro_f1["value"]:
                best_val_macro_f1["value"] = float(metrics["macro_f1"])
                best_val_macro_f1["round"] = int(server_round)
                best_val_macro_f1["params"] = [a.copy() for a in parameters]

        return metrics["loss"], {
            "val_macro_f1": metrics["macro_f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_accuracy": metrics["accuracy"],
        }

    def on_fit_config_fn(server_round: int) -> Dict:
        return {"round": int(server_round)}

    # Per-(round, client) update-norm sink. The strategy appends one row
    # per participating client per round whenever this list is not None.
    update_norm_rows: List[Dict] = []
    n_fit = max(1, int(round(args.fraction_fit * num_clients)))
    strategy = PairedFedNovaStrategy(
        client_momentum=float(args.momentum),       # ← MF2: momentum-aware normaliser
        tau_clip_min=int(args.tau_clip_min),         # 0 = off (byte-identical)
        server_momentum=float(args.server_momentum), # 0.0 = off (byte-identical)
        fraction_fit=float(args.fraction_fit),       # ← MF4: respect partial participation
        fraction_evaluate=0.0,
        min_fit_clients=n_fit,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        accept_failures=False,
        update_norm_rows=(update_norm_rows if args.log_update_norms else None),
    )

    def client_fn(context_or_cid) -> fl.client.Client:
        if hasattr(context_or_cid, "node_config"):
            cid_int = int(context_or_cid.node_config.get("partition-id", 0))
        elif hasattr(context_or_cid, "cid"):
            cid_int = int(context_or_cid.cid)
        else:
            cid_int = int(context_or_cid)
        # Per-client lr lookup (asymmetric-LR experiment).
        client_lr = (
            lr_per_client_map.get(cid_int, args.lr)
            if lr_per_client_map is not None
            else args.lr
        )
        return FlClientFedNova(
            cid=cid_int,
            train_dataset=train,
            indices=client_indices[cid_int],
            model_builder=lambda: DermMNISTCNN(num_classes=7, dropout=0.2),
            seed=seed,
            lr=client_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device_str,
            epoch_schedule=epoch_schedule,
        ).to_client()

    print(f"\n=== Flower runtime: FEDNOVA seed={seed} ===")
    print(f"  partition={args.partition}  rounds={args.num_rounds}  E_max={args.local_epochs}  C={args.fraction_fit}")
    print(f"  client sizes: {[len(c) for c in client_indices]}")
    print(f"  client_momentum={args.momentum}  (used for FedNova momentum-aware normaliser)")
    print(f"  system_het: mode={args.system_het_mode}")
    print(f"  device={device_str}\n")

    client_resources = ({"num_cpus": 1, "num_gpus": 1.0 / num_clients}
                        if device_str == "cuda" else {"num_cpus": 1, "num_gpus": 0.0})

    # Use Flower's default ray.init() — RAY_TMPDIR from the SLURM template
    # is sufficient for the historical working configuration (see
    # submit_two_client_90_10_rare_stress.sh and the existing results that
    # were generated this way).

    t0 = time.time()
    history_obj = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    elapsed = time.time() - t0

    # Back-fill train_loss from PairedFedNovaStrategy's aggregated fit metrics.
    if hasattr(history_obj, "metrics_distributed_fit"):
        for r, v in history_obj.metrics_distributed_fit.get("train_loss", []):
            train_loss_by_round[int(r)] = float(v)
    for row in history_rows:
        row["train_loss"] = train_loss_by_round.get(int(row["round"]), float("nan"))

    test_model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    numpy_to_state_dict(test_model, best_val_macro_f1["params"])
    test_metrics = evaluate(
        test_model, test_loader, device, num_classes=7,
        return_predictions=True,
    )
    test_metrics["selected_round"] = best_val_macro_f1["round"]
    test_metrics["best_val_macro_f1"] = best_val_macro_f1["value"]
    _preds = test_metrics.pop("predictions", None)
    _targets = test_metrics.pop("targets", None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sh_tag = "" if args.system_het_mode == "uniform" else f"_sh-{args.system_het_mode}"
    c_tag = "" if abs(args.fraction_fit - 1.0) < 1e-9 else f"_C{args.fraction_fit}"
    # Per-client-lr tag — prevents filename collision when --lr-per-client
    # varies the LR per client. Without this tag, asymmetric-LR FedNova
    # runs overwrite symmetric-LR runs within the same seed.
    if lr_per_client_map is not None:
        lrc_tag = "_lrPC-" + "-".join(
            f"c{cid}lr{lr_per_client_map[cid]}"
            for cid in sorted(lr_per_client_map)
        )
    else:
        lrc_tag = ""
    # Probe-variant tags. Both default to empty string so baseline filenames
    # stay byte-identical to pre-probe runs.
    tc_tag = f"_tauclip{args.tau_clip_min}" if args.tau_clip_min > 0 else ""
    sm_tag = (f"_smom{args.server_momentum:g}"
              if args.server_momentum > 0.0 else "")
    stem = (f"fednova_mu0.0_E{args.local_epochs}"
            f"{sh_tag}{c_tag}{lrc_tag}{tc_tag}{sm_tag}_s{seed}")

    import pandas as pd
    pd.DataFrame(history_rows).to_csv(out_dir / f"history_{stem}.csv", index=False)
    # Per-(round, client) update norms if --log-update-norms was set.
    if args.log_update_norms and update_norm_rows:
        pd.DataFrame(update_norm_rows).to_csv(
            out_dir / f"client_update_norms_{stem}.csv", index=False)
    predictions_file = None
    if _preds is not None and _targets is not None:
        predictions_file = f"test_predictions_{stem}.npz"
        np.savez_compressed(
            out_dir / predictions_file,
            predictions=np.asarray(_preds, dtype=np.int64),
            targets=np.asarray(_targets, dtype=np.int64),
            selected_round=int(test_metrics["selected_round"]),
            seed=int(seed),
            algorithm="fednova",
        )
    with open(out_dir / f"test_at_best_{stem}.json", "w") as f:
        json.dump({
            **test_metrics,
            "predictions_file": predictions_file,
            "seed": seed, "algorithm": "fednova", "mu": 0.0,
            "lr_per_client": (
                {str(k): float(v) for k, v in lr_per_client_map.items()}
                if lr_per_client_map is not None else None
            ),
            "local_epochs": args.local_epochs, "num_rounds": args.num_rounds,
            "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "device": device_str,
            "fraction_fit": float(args.fraction_fit),
            # Provenance fields (mixing-risk mitigation per audit):
            "partition": args.partition,
            "image_size": int(args.image_size),
            "npz_path": str(args.npz_path),
            "framework": "flower-simulation",
            "framework_version": fl.__version__,
            "runner_script": "run_one_fednova_flower.py",
            "loss_type": "cross_entropy",
            "client_momentum_for_fednova_normaliser": args.momentum,
            "fednova_normaliser_formula": "(tau*(1-m) - m*(1-m**tau)) / (1-m)**2  [L1 norm of cumulative momentum series; Wang 2020 §3.3]",
            # Mechanism-probe settings (default 0 = baseline FedNova).
            "tau_clip_min": int(args.tau_clip_min),
            "server_momentum": float(args.server_momentum),
            "system_het": sh_cfg.to_dict(),
            "elapsed_s": elapsed,            # legacy field name
            "wall_clock_seconds": elapsed,   # canonical (audit fix)
            # Runtime provenance (CP3.2 / Rank-4 patch — git SHA,
            # hostname, torch/python versions, timestamps, etc.)
            **collect_runtime_provenance(run_started_at),
        }, f, indent=2)

    print(f"\nTest @ best-val (round {test_metrics['selected_round']}, val_macro_f1={test_metrics['best_val_macro_f1']:.4f}):")
    print(f"  test_macro_f1   = {test_metrics['macro_f1']:.4f}")
    print(f"  test_balanced_a = {test_metrics['balanced_accuracy']:.4f}")
    print(f"  test_accuracy   = {test_metrics['accuracy']:.4f}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
