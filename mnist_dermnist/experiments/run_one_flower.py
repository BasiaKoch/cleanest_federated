"""Flower-framework entry point — paired-fair FedAvg / FedProx on DermaMNIST.

Mirrors the CLI of `run_one.py` exactly, but routes through Flower's
`flwr.simulation.start_simulation` runtime. The produced results are
equivalent (within floating-point noise) to those from the pure-PyTorch
runner — both implement the same FedAvg and FedProx mathematics; they
differ only in the orchestration framework around the per-round
broadcast / local-train / aggregate cycle.

Usage:
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \\
        --algorithm fedprox --mu 0.01 --seed 42 \\
        --local-epochs 20 --num-rounds 150 \\
        --partition balanced_paired_7_clients \\
        --device cuda \\
        --out-dir mnist_dermnist/results/headline_flower

When `--system-het-mode` is set to a non-uniform value, the server's
configure_fit() function reads from the per-(round, client) schedule
generated in `mnist_dermnist.fl.system_het`, identical to the pure-PyTorch
path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from mnist_dermnist.fl_flower.client import FlClient, state_dict_to_numpy, numpy_to_state_dict
from mnist_dermnist.fl_flower.strategy_straggler_dropping import StragglerDroppingFedAvg
from mnist_dermnist.models import DermMNISTCNN, get_model, resolve_variant


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
        description="Flower-framework FedAvg/FedProx run (paired-fair).")
    ap.add_argument("--algorithm", choices=["fedavg", "fedprox"], required=True)
    ap.add_argument("--mu", type=float, default=0.0)
    # Per-client μ override. When set, takes precedence over --mu and
    # establishes a client-cid → μ mapping. Format: "0:0.01,1:0.0" sets
    # μ = 0.01 on client 0 and μ = 0 on client 1. Used to test
    # asymmetric-proximal-regularisation hypotheses (HAPI-FedProx
    # [Springer 2024, DOI:10.1007/978-3-032-11733-5_17]; Yao et al.
    # 2024 NeurIPS [arXiv:2410.08934] — "Effect of Personalization in
    # FedProx" — proves the optimal μ depends on per-client
    # heterogeneity). Any client cid not listed in the mapping inherits
    # the global --mu value. Recorded in the output JSON metadata.
    ap.add_argument("--mu-per-client", type=str, default=None,
                    help="Per-client μ override (e.g. '0:0.01,1:0.0'). "
                         "Overrides --mu for the specified client cids. "
                         "Outside Li 2020 Thm 4's uniform-μ regime; "
                         "covered by Yao et al. 2024 per-client minimax "
                         "framework (arXiv:2410.08934).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-epochs", type=int, default=20)
    ap.add_argument("--num-rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--partition", choices=list(PARTITIONERS),
                    default="balanced_paired_7_clients")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path",
                    default=str(Path(__file__).resolve().parents[2] / "dermamnist_64.npz"))
    ap.add_argument("--out-dir", default="mnist_dermnist/results/headline_flower")
    ap.add_argument("--fraction-fit", type=float, default=1.0,
                    help="Fraction of clients sampled per round (C). "
                         "Default 1.0 = full participation (cross-silo). "
                         "Values < 1 enable partial-participation sensitivity.")
    ap.add_argument("--system-het-mode",
                    choices=["uniform", "fixed_stragglers",
                             "random_stragglers", "permanent_stragglers"],
                    default="uniform")
    ap.add_argument("--straggler-epochs", type=int, default=5)
    ap.add_argument("--fixed-straggler-ids", default=None)
    ap.add_argument("--straggler-fraction", type=float, default=0.5)
    ap.add_argument("--permanent-epoch-choices", default=None,
                    help="For --system-het-mode permanent_stragglers: "
                         "comma-separated discrete set of E_i values from "
                         "which each client's permanent local-epoch budget "
                         "is drawn once at experiment start. Default "
                         "'2,5,10,15,20' (Wang et al. 2020 FedNova-style).")
    # Loss-side imbalance baselines (audit HV2). 'ce' = headline. Set to
    # 'class_weighted_ce' or 'focal' for the imbalance-aware comparators.
    ap.add_argument("--loss-type",
                    choices=["ce", "class_weighted_ce", "focal"],
                    default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    # Architecture-normalization variant (architecture ablation).
    # 'gn' (default) = DermMNISTCNN with GroupNorm — the headline.
    # 'bn' = DermMNISTCNN_BN with BatchNorm — the FL-unfriendly ablation
    #         that probes the BN-running-stats × non-IID interaction
    #         (cf. Li et al. 2021 FedBN). Results saved with an
    #         '_arch-bn' filename tag so they cannot be mixed with the
    #         GN headline.
    ap.add_argument("--model-variant",
                    choices=["gn", "bn"],
                    default="gn",
                    help="Architecture-normalization variant: 'gn' = "
                         "GroupNorm (headline, default), 'bn' = BatchNorm "
                         "(architecture ablation; see runpod_arch_ablation_bn.sh).")
    # Asymmetric straggler-handling protocol (Li et al. 2020 §5.2).
    # When set, the server's FedAvg aggregation DROPS client updates
    # whose reported local_epochs < E_max. This is the FedAvg side of
    # the canonical FedProx evaluation: FedAvg discards γ-inexact
    # updates while FedProx (which is unaffected by this flag) includes
    # them via the proximal anchor's stability guarantee. Run FedAvg
    # WITH this flag and FedProx WITHOUT it to reproduce Li 2020's
    # 22% advantage on synthetic data (or its analogue on DermaMNIST).
    ap.add_argument("--drop-stragglers", action="store_true",
                    help="Drop straggler clients (local_epochs < E_max) "
                         "from FedAvg aggregation. Implements Li et al. "
                         "2020 §5.2 asymmetric protocol. Default off "
                         "(both algorithms see identical client subsets).")
    # Mechanism diagnostic — writes client_update_norms_*.csv beside the JSON.
    ap.add_argument("--log-update-norms", action="store_true",
                    help="Capture per-(round, client) L2 update norm "
                         "||w_k^{t+1} - w^t||_2 and write to a sibling CSV. "
                         "Off by default; clients always compute and return "
                         "the value in fit metrics, but the runner only "
                         "materialises the CSV when this flag is set.")
    # Save model weights at the best-val checkpoint — needed by
    # downstream local-adaptation / personalised-FL baselines
    # (Yu et al. 2022, Collins et al. 2021). Off by default so the
    # headline outputs are unchanged; turn on for the small-hospital
    # case study's fine-tuning baselines.
    ap.add_argument("--save-best-checkpoint", action="store_true",
                    help="Write the global model state_dict at the "
                         "best-validation checkpoint to a sibling .pt "
                         "file ('best_state_<stem>.pt'). Used as the "
                         "starting point for local-adaptation fine-tuning "
                         "in run_finetune.py.")
    return ap


def main():
    run_started_at = utc_now_iso()
    args = build_parser().parse_args()
    mu = 0.0 if args.algorithm == "fedavg" else float(args.mu)
    seed = int(args.seed)

    # Parse --mu-per-client into a {cid: mu} dict. Empty/None → no override.
    mu_per_client_map: "dict[int, float] | None" = None
    if args.mu_per_client:
        if args.algorithm != "fedprox":
            raise ValueError("--mu-per-client requires --algorithm fedprox "
                             "(per-client proximal coefficients only meaningful "
                             "when the proximal term is active).")
        mu_per_client_map = {}
        for spec in args.mu_per_client.split(","):
            cid_s, mu_s = spec.split(":")
            mu_per_client_map[int(cid_s.strip())] = float(mu_s.strip())
        if any(v < 0 for v in mu_per_client_map.values()):
            raise ValueError("--mu-per-client values must be non-negative.")

    # --- Reproducibility ---
    # NOTE: Python's `random` module is seeded explicitly here because
    # Flower's ClientManager.sample() internally calls random.sample(...)
    # for partial-participation client selection. Without this seed, two
    # runs with --fraction-fit < 1.0 (and otherwise identical
    # configurations) will select different client subsets per round.
    # The pure-PyTorch reference loop in fl/server_loop.py uses a
    # separate np.random.default_rng(seed + 9_000_001) for client
    # sampling; the two runtimes therefore sample DIFFERENT subsets even
    # with the same seed under partial participation. This is acceptable
    # for the present thesis (only C=1.0 results are claimed in the
    # headline; the partial-participation sensitivity is descriptive)
    # but must be acknowledged in any comparison.
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

    # --- System-heterogeneity schedule ---
    fixed_ids = None
    if args.fixed_straggler_ids:
        fixed_ids = [int(x) for x in args.fixed_straggler_ids.split(",")]
    permanent_choices = None
    if args.permanent_epoch_choices:
        permanent_choices = [int(x) for x in args.permanent_epoch_choices.split(",")]
    sh_cfg = SystemHetConfig(
        mode=args.system_het_mode,
        E_max=args.local_epochs,
        E_straggler=args.straggler_epochs,
        fixed_straggler_ids=fixed_ids,
        random_straggler_fraction=args.straggler_fraction,
        permanent_epoch_choices=permanent_choices,
    )
    epoch_schedule = build_epoch_schedule(
        sh_cfg, num_clients=num_clients,
        num_rounds=args.num_rounds, seed=seed,
    )

    # --- Global model: initialise ONCE, deterministically, before clients ---
    # Architecture variant is gated via --model-variant; default 'gn' is the
    # headline DermMNISTCNN (GroupNorm). 'bn' selects DermMNISTCNN_BN for the
    # architecture ablation (results go to a different output directory and
    # filename tag, see _arch_tag below).
    model_registry_key = resolve_variant(args.model_variant)
    torch.manual_seed(seed)  # ensure init is paired
    global_model = get_model(model_registry_key, num_classes=7, dropout=0.2).to(device)
    initial_params = state_dict_to_numpy(global_model)

    # --- Tracking state for best-val checkpoint ---
    best_val_macro_f1 = {"value": -1.0, "round": -1,
                         "params": [arr.copy() for arr in initial_params]}
    history_rows: List[Dict] = []
    # Populated by fit_metrics_aggregation_fn (below), keyed by 1-based round.
    # Read out post-simulation to back-fill the train_loss column in
    # history_rows so the Flower CSV matches the pure-PyTorch CSV schema.
    train_loss_by_round: Dict[int, float] = {}
    # Per-(round, client) update norms when --log-update-norms is set.
    # The fit-metrics aggregation callback is invoked once per round in
    # the order the rounds execute, so we track the round via a counter
    # rather than relying on Flower exposing it to the callback.
    update_norm_rows: List[Dict] = []
    _round_counter = {"r": 0}

    # --- Centralised evaluation function called every round by the server ---
    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config):
        # Reconstruct model and evaluate on val
        eval_model = get_model(model_registry_key, num_classes=7, dropout=0.2).to(device)
        numpy_to_state_dict(eval_model, parameters)
        metrics = evaluate(eval_model, val_loader, device, num_classes=7)

        # Flower invokes evaluate_fn once at server_round=0 with the
        # untrained initial parameters AND after every fit round 1..R.
        # The pure-PyTorch reference loop (fl/server_loop.py) only
        # evaluates after rounds 1..R, so we skip writing the round-0
        # row here to keep history CSVs cross-runtime comparable.
        # The best-val tracker is also skipped for round 0; untrained
        # params would not beat any trained checkpoint in practice.
        if int(server_round) >= 1:
            row = {
                "seed": seed,
                "algorithm": args.algorithm,
                "mu": mu,
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
                best_val_macro_f1["params"] = [arr.copy() for arr in parameters]

        return metrics["loss"], {
            "val_macro_f1": metrics["macro_f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_accuracy": metrics["accuracy"],
        }

    # --- Per-round configure_fit: pass per-client local_epochs ---
    def on_fit_config_fn(server_round: int) -> Dict:
        # Note: server_round is 1-based in Flower
        return {"round": server_round}

    # --- Aggregate per-client train_loss so it lands in the history CSV.
    # FedAvg's default strategy DROPS client fit-metrics unless we provide
    # an aggregator. Without this, the Flower history would lack train_loss
    # entirely (the pure-PyTorch CSV has it; the cross-runtime mismatch was
    # the original B3 from the audit).
    def fit_metrics_aggregation_fn(metrics):
        # Flower invokes this callback exactly once per round, inside
        # FedAvg.aggregate_fit, in round order. We use that ordering to
        # tag each client's update_norm with the round it belongs to.
        _round_counter["r"] += 1
        r = _round_counter["r"]
        if args.log_update_norms:
            for n, m in metrics:
                if "update_norm" in m and "cid" in m:
                    update_norm_rows.append({
                        "round": int(r),
                        "client_id": int(m["cid"]),
                        "update_norm": float(m["update_norm"]),
                        "n_samples": int(n),
                        "local_epochs": int(m.get("local_epochs", -1)),
                    })
        total_n = sum(int(n) for n, _ in metrics)
        if total_n <= 0:
            return {}
        return {
            "train_loss": float(
                sum(float(m.get("train_loss", 0.0)) * int(n) for n, m in metrics)
                / total_n
            ),
        }

    # --- Strategy: standard FedAvg aggregation (FedProx uses same aggregation;
    #     the proximal term is applied client-side in fit()).
    #
    # If --drop-stragglers is set AND we are running an algorithm where
    # straggler dropping is meaningful (i.e., fedavg), wrap aggregation
    # in StragglerDroppingFedAvg. The semantically-correct application
    # of Li 2020 §5.2 is:
    #     algorithm=fedavg  + --drop-stragglers    → drops stragglers
    #     algorithm=fedprox + (no flag)            → keeps stragglers (γ-inexact)
    # This is the canonical asymmetric-comparison protocol.
    #
    # If --drop-stragglers is set with algorithm=fedprox, we honor it
    # (drops stragglers from FedProx aggregation too). This is rare but
    # allows the symmetric-dropping comparison if a user wants it.
    n_fit = max(1, int(round(args.fraction_fit * num_clients)))
    strategy_kwargs = dict(
        fraction_fit=float(args.fraction_fit),
        fraction_evaluate=0.0,  # we don't run per-client eval
        min_fit_clients=n_fit,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        accept_failures=False,
    )
    if args.drop_stragglers:
        strategy = StragglerDroppingFedAvg(
            E_max=int(args.local_epochs),
            **strategy_kwargs,
        )
    else:
        strategy = fl.server.strategy.FedAvg(**strategy_kwargs)

    # --- Client factory passes per-(round, cid) local-epoch override
    #     through a custom config dict per-client. Flower's NumPyClient API
    #     reads the config in fit(). We inject the schedule by composing
    #     a per-client-cid wrapper around on_fit_config_fn. The simplest
    #     approach is for each client to read its own row from the schedule
    #     at fit-time, given the round number from config["round"].
    # Flower 1.29 prefers a Context-based client_fn; we accept both shapes
    # so this runner is portable across 1.7–1.29.
    # The full per-(round, client) epoch schedule is captured in the closure
    # and passed to each client at construction; no fragile config-string
    # round-tripping required.
    # Build the local-training loss once and share across all clients
    # (the inverse-frequency weights depend only on the global training
    # labels, which all clients agree on by design).
    def build_criterion():
        if args.loss_type == "ce":
            return torch.nn.CrossEntropyLoss()
        from mnist_dermnist.fl.class_imbalance import (
            make_class_weighted_ce, make_focal_loss,
        )
        if args.loss_type == "class_weighted_ce":
            return make_class_weighted_ce(train.labels, num_classes=7,
                                          device=device_str)
        return make_focal_loss(gamma=args.focal_gamma, labels=train.labels,
                               num_classes=7, device=device_str)

    def client_fn(context_or_cid) -> fl.client.Client:
        if hasattr(context_or_cid, "node_config"):
            cid_int = int(context_or_cid.node_config.get("partition-id", 0))
        elif hasattr(context_or_cid, "cid"):
            cid_int = int(context_or_cid.cid)
        else:
            cid_int = int(context_or_cid)
        # Per-client μ: lookup overrides --mu when --mu-per-client is set.
        # Clients not listed in the map fall back to the global μ.
        client_mu = (
            mu_per_client_map.get(cid_int, mu)
            if mu_per_client_map is not None
            else mu
        )
        return FlClient(
            cid=cid_int,
            train_dataset=train,
            indices=client_indices[cid_int],
            model_builder=lambda: get_model(model_registry_key, num_classes=7, dropout=0.2),
            seed=seed,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            proximal_mu=client_mu,
            device=device_str,
            epoch_schedule=epoch_schedule,
            criterion=build_criterion(),
        ).to_client()

    # --- Launch Flower simulation ---
    print(f"\n=== Flower runtime: {args.algorithm.upper()} (μ={mu}) seed={seed} ===")
    print(f"  partition={args.partition}  rounds={args.num_rounds}  E_max={args.local_epochs}  C={args.fraction_fit}")
    print(f"  client sizes: {[len(c) for c in client_indices]}")
    print(f"  system_het: mode={args.system_het_mode}")
    print(f"  device={device_str}\n")

    client_resources = ({"num_cpus": 1, "num_gpus": 1.0 / num_clients}
                        if device_str == "cuda" else {"num_cpus": 1, "num_gpus": 0.0})

    # --- ray_init_args: defensive configuration for shared HPC compute nodes.
    #
    # On Cambridge's ampere queue (and similar SLURM clusters) multiple SLURM
    # jobs can co-locate on the same physical compute node. Ray's default
    # ray.init() tries to bind GCS to port 6379 (Redis-style); when a sibling
    # job already holds that port the GCS server fails silently (gcs_server.out
    # is empty) and the user-facing error is the unhelpful
    #     RuntimeError: Failed to start GCS.
    # The previous RAY_TMPDIR-isolation patch in the SLURM templates fixed
    # session-directory collisions but not the port collision. Here we:
    #   (a) compute a deterministic per-job GCS port in the ephemeral range
    #       [10000, 60000) keyed off SLURM_JOB_ID — so each job gets a unique
    #       port even on a shared node;
    #   (b) disable the Ray dashboard — it grabs additional ports (8265 by
    #       default) and is unused in batch-mode HPC runs anyway;
    #   (c) pass _temp_dir through explicitly (env-var alone is not honoured
    #       reliably across all Ray versions);
    #   (d) match num_cpus to the SLURM allocation so Ray doesn't oversubscribe
    #       the node detecting 64-core total when we only have 4.
    import os as _os
    ray_init_args: Dict = {
        # Dashboard binds port 8265 by default; on shared compute nodes a
        # sibling job will already hold it. Disabling it removes one of the
        # most common GCS-startup failure modes.
        "include_dashboard": False,
        # Allow re-init across the SLURM template's retry loop without
        # raising on the second attempt.
        "ignore_reinit_error": True,
        # Reduce SLURM log spam.
        "log_to_driver": False,
    }
    # Pass _temp_dir explicitly. The RAY_TMPDIR env var IS honoured by Ray,
    # but passing it directly is the recommended approach in Ray 2.x and
    # eliminates ambiguity if a future Ray version changes behaviour.
    if "RAY_TMPDIR" in _os.environ:
        ray_init_args["_temp_dir"] = _os.environ["RAY_TMPDIR"]
    # Match Ray's resource view to the SLURM allocation. Without this, Ray
    # detects the FULL compute node (e.g. 64 cores) and over-subscribes
    # against sibling jobs on the same node, which can starve GCS startup.
    _slurm_cpus = _os.environ.get("SLURM_CPUS_PER_TASK")
    if _slurm_cpus and _slurm_cpus.isdigit():
        ray_init_args["num_cpus"] = int(_slurm_cpus)
    if device_str == "cuda":
        ray_init_args["num_gpus"] = 1
    # Unique GCS port per SLURM job. Ray's default GCS port is 6379 (Redis-
    # style); on shared HPC compute nodes this is frequently held by another
    # user's stale Ray instance or a system service, and the GCS server dies
    # silently (gcs_server.out is empty) when bind fails. Deriving a
    # deterministic port in the high ephemeral range [30000, 60000) from
    # SLURM_JOB_ID gives each job a unique GCS bind point — eliminating the
    # collision regardless of what other processes hold lower ports.
    _slurm_jid = _os.environ.get("SLURM_JOB_ID")
    if _slurm_jid and _slurm_jid.isdigit():
        ray_init_args["_redis_password"] = f"flwr_{_slurm_jid}"  # namespace lock
        gcs_port = 30000 + (int(_slurm_jid) % 30000)
        ray_init_args["port"] = gcs_port
    print(f"  ray_init_args: {ray_init_args}")

    t0 = time.time()
    history_obj = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )
    elapsed = time.time() - t0

    # --- Back-fill train_loss into history_rows from the aggregated fit metrics.
    # Flower stores aggregated fit metrics in `metrics_distributed_fit`:
    # a dict {metric_name -> List[Tuple[round, value]]}.
    if hasattr(history_obj, "metrics_distributed_fit"):
        for r, v in history_obj.metrics_distributed_fit.get("train_loss", []):
            train_loss_by_round[int(r)] = float(v)
    # When --drop-stragglers is set, the strategy also emits per-round
    # n_kept and n_dropped via aggregate_fit's returned metrics. Persist
    # these so the history CSV records how many clients were filtered
    # at each round — necessary for thesis-defensibility of the
    # asymmetric protocol claim (cf. Li 2020 §5.2).
    n_kept_by_round: Dict[int, int] = {}
    n_dropped_by_round: Dict[int, int] = {}
    if args.drop_stragglers and hasattr(history_obj, "metrics_distributed_fit"):
        for r, v in history_obj.metrics_distributed_fit.get("n_kept", []):
            n_kept_by_round[int(r)] = int(v)
        for r, v in history_obj.metrics_distributed_fit.get("n_dropped", []):
            n_dropped_by_round[int(r)] = int(v)
    for row in history_rows:
        row["train_loss"] = train_loss_by_round.get(int(row["round"]), float("nan"))
        if args.drop_stragglers:
            row["n_kept"]    = n_kept_by_round.get(int(row["round"]), -1)
            row["n_dropped"] = n_dropped_by_round.get(int(row["round"]), -1)

    # --- Final test at best-val checkpoint ---
    # `return_predictions=True` keeps per-sample argmax + target arrays
    # for downstream confusion-matrix analysis (audit P2 fix). They're
    # serialised to a sibling .npz, NOT inlined into the JSON.
    test_model = get_model(model_registry_key, num_classes=7, dropout=0.2).to(device)
    numpy_to_state_dict(test_model, best_val_macro_f1["params"])
    test_metrics = evaluate(
        test_model, test_loader, device, num_classes=7,
        return_predictions=True,
    )
    test_metrics["selected_round"] = best_val_macro_f1["round"]
    test_metrics["best_val_macro_f1"] = best_val_macro_f1["value"]
    # Extract predictions before constructing the JSON payload.
    _preds = test_metrics.pop("predictions", None)
    _targets = test_metrics.pop("targets", None)

    # --- Write outputs (mirror save_run_outputs from pure-PyTorch path) ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sh_tag = "" if args.system_het_mode == "uniform" else f"_sh-{args.system_het_mode}"
    # Encode fraction_fit only when it differs from full participation, so files
    # produced under C=1.0 retain their existing names and don't break the
    # already-collected headline data.
    c_tag = "" if abs(args.fraction_fit - 1.0) < 1e-9 else f"_C{args.fraction_fit}"
    # Architecture-variant tag — only emitted for non-default ('bn') so that
    # the existing 'gn' headline filenames are unchanged for back-compat.
    arch_tag = "" if args.model_variant == "gn" else f"_arch-{args.model_variant}"
    # Asymmetric-straggler-protocol tag — only when stragglers are dropped.
    drop_tag = "_drop" if args.drop_stragglers else ""
    # Per-client-μ tag — only when asymmetric μ is in effect. Encodes the
    # mapping in the filename so it cannot be silently mixed with
    # symmetric-μ runs in downstream analysis. Format: "_muPC-c0m0.01-c1m0.0".
    if mu_per_client_map is not None:
        muc_tag = "_muPC-" + "-".join(
            f"c{cid}m{mu_per_client_map[cid]}"
            for cid in sorted(mu_per_client_map)
        )
    else:
        muc_tag = ""
    stem = (f"{args.algorithm}_mu{mu}_E{args.local_epochs}"
            f"{sh_tag}{c_tag}{arch_tag}{drop_tag}{muc_tag}_s{seed}")

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
            algorithm=args.algorithm,
        )

    # Optional best-val state_dict for downstream local-adaptation
    # baselines (run_finetune.py).
    checkpoint_file = None
    if args.save_best_checkpoint:
        checkpoint_file = f"best_state_{stem}.pt"
        torch.save(
            test_model.state_dict(),
            out_dir / checkpoint_file,
        )
    with open(out_dir / f"test_at_best_{stem}.json", "w") as f:
        json.dump({
            **test_metrics,
            "predictions_file": predictions_file,
            "checkpoint_file": checkpoint_file,
            "seed": seed, "algorithm": args.algorithm, "mu": mu,
            "mu_per_client": (
                {str(k): float(v) for k, v in mu_per_client_map.items()}
                if mu_per_client_map is not None else None
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
            "runner_script": "run_one_flower.py",
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
            # Architecture-variant provenance (architecture ablation).
            # 'gn' = headline DermMNISTCNN; 'bn' = DermMNISTCNN_BN ablation.
            "model_variant": args.model_variant,
            "model_name": model_registry_key,
            "model_normalization": "GroupNorm" if args.model_variant == "gn" else "BatchNorm2d",
            # Straggler-policy provenance (Li 2020 §5.2 protocol).
            "drop_stragglers": bool(args.drop_stragglers),
            "straggler_policy": "drop_below_E_max" if args.drop_stragglers else "include_all",
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
