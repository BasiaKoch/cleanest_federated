"""Paired-fair FL server loop for FedAvg vs FedProx.

DESIGN INVARIANTS (per spec):
  - For a given `seed`, FedAvg and FedProx(μ>0) MUST see:
      • identical initial global model weights
      • identical client partition
      • identical client sampling schedule per round
      • identical per-client DataLoader generator state (round, cid)
      • identical optimizer settings (lr, momentum, wd)
    Differences come ONLY from the gated proximal-term branch.

  - global_weights_frozen is snapshotted ONCE per round (before any client
    runs local training) and passed BY VALUE (detached clones) to clients.
    It is not aliased to the live global model.

  - Validation is on every round. Test set is touched ONCE, at the
    best-val-macro-F1 checkpoint.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from mnist_dermnist.fl.aggregation import weighted_average_state_dicts
from mnist_dermnist.fl.evaluation import evaluate
from mnist_dermnist.fl.local_train import freeze_global_weights, local_train
from mnist_dermnist.fl.system_het import (
    SystemHetConfig,
    build_epoch_schedule,
    summarise_schedule,
)


# ---------------------------------------------------------------------------
# Seed plumbing
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Reset Python, NumPy, PyTorch (CPU+CUDA) RNGs and enable deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms — slightly slower but correct
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader_generator_seed(base_seed: int, round_num: int, client_id: int) -> int:
    """Per (seed, round, cid) DataLoader generator seed — paired across algorithms."""
    return int(base_seed) * 10_000 + int(round_num) * 100 + int(client_id)


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

@dataclass
class FLConfig:
    seed: int
    algorithm: str                  # 'fedavg' or 'fedprox'
    mu: float                       # 0 for fedavg; >0 for fedprox
    num_rounds: int
    local_epochs: int               # baseline E (used when system_het.mode='uniform')
    fraction_fit: float             # e.g. 1.0 for full participation
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    num_classes: int = 7
    device: str = "cpu"
    # System-heterogeneity scenario. When mode='uniform' (default), every client
    # performs local_epochs every round, matching the headline statistical-
    # heterogeneity experiments. Other modes simulate stragglers; see
    # mnist_dermnist.fl.system_het.
    system_het: SystemHetConfig = field(default_factory=SystemHetConfig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fl(
    cfg: FLConfig,
    model_builder: Callable[[], nn.Module],
    train_dataset: Dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    client_indices: Sequence[Sequence[int]],
) -> Dict:
    """Run one FL experiment. Returns a dict with `history` (per-round) and
    `test_metrics` (computed only at the best-val checkpoint).

    Args:
        cfg: FLConfig — every algorithmic setting comes from here.
        model_builder: zero-arg callable that returns a fresh model. The SAME
                       seed before constructing the global model AND the same
                       builder must be used across paired FedAvg/FedProx runs.
        train_dataset: full PyTorch dataset; client_indices indexes into it.
        val_loader, test_loader: global loaders, untouched by clients.
        client_indices: List[List[int]] partition (one list per client).
    """
    set_all_seeds(cfg.seed)
    device = torch.device(cfg.device)

    # 1) GLOBAL MODEL INITIALIZATION — single source of randomness, IDENTICAL
    # for any pair of (FedAvg, FedProx) runs that share cfg.seed.
    global_model = model_builder().to(device)

    num_clients = len(client_indices)
    if num_clients == 0:
        raise ValueError("client_indices is empty")

    # 2) CLIENT SAMPLING SCHEDULE — paired across algorithms via a dedicated RNG.
    # Use a derived RNG so it doesn't interfere with PyTorch RNG that drives
    # parameter init/dropout.
    sampling_rng = np.random.default_rng(seed=cfg.seed + 9_000_001)
    sampled_per_round: List[List[int]] = []
    n_sample = max(1, int(round(cfg.fraction_fit * num_clients)))
    for _ in range(cfg.num_rounds):
        chosen = sorted(sampling_rng.choice(num_clients, size=n_sample, replace=False).tolist())
        sampled_per_round.append(chosen)

    # 2b) PER-(round, client) LOCAL-EPOCH SCHEDULE for system heterogeneity.
    # Seeded with cfg.seed so paired FedAvg/FedProx runs see an identical
    # schedule; within-pair Δ therefore reflects only the algorithm difference.
    # When system_het.mode == "uniform", every entry equals cfg.local_epochs
    # and behaviour is bit-identical to the pre-system-het code path.
    epoch_schedule = build_epoch_schedule(
        cfg.system_het,
        num_clients=num_clients,
        num_rounds=cfg.num_rounds,
        seed=cfg.seed,
    )

    # State trackers
    history_rows: List[Dict] = []
    best_val_macro_f1 = -1.0
    best_state_dict: Dict[str, torch.Tensor] | None = None
    best_round: int = -1

    # ------------------------------------------------------------------ rounds
    for r in range(1, cfg.num_rounds + 1):
        # 3) FREEZE GLOBAL — snapshot global params for the proximal term and
        # also as the "starting point" each sampled client loads.
        global_weights_frozen = freeze_global_weights(global_model)   # ONCE per round
        global_state = copy.deepcopy(global_model.state_dict())

        # Sampled clients for this round
        sampled = sampled_per_round[r - 1]

        # 4) LOCAL TRAINING — paired loaders per (seed, round, cid).
        local_state_dicts: List[Dict[str, torch.Tensor]] = []
        local_weights_for_agg: List[int] = []
        round_train_losses: List[float] = []
        round_train_sizes: List[int] = []

        for cid in sampled:
            # Fresh client model from global state
            client_model = model_builder().to(device)
            client_model.load_state_dict(global_state)

            # Deterministic per (seed, round, cid) DataLoader generator
            gen = torch.Generator().manual_seed(
                dataloader_generator_seed(cfg.seed, r, cid)
            )
            client_subset = Subset(train_dataset, list(client_indices[cid]))
            client_loader = DataLoader(
                client_subset,
                batch_size=min(cfg.batch_size, max(1, len(client_subset))),
                shuffle=True,
                generator=gen,
                num_workers=0,         # deterministic — single process
                drop_last=False,
            )

            # Per-round deterministic RNG state for any in-fit randomness
            # (e.g., Dropout). Same seed → same masks for both algorithms.
            torch.manual_seed(dataloader_generator_seed(cfg.seed, r, cid))

            # Per-(round, cid) local epochs from the system-het schedule.
            # When system_het.mode='uniform' this equals cfg.local_epochs.
            client_local_epochs = int(epoch_schedule[r - 1, cid])

            stats = local_train(
                client_model,
                client_loader,
                num_epochs=client_local_epochs,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                # GATED — μ=0 path is identical to plain FedAvg by construction.
                proximal_mu=float(cfg.mu),
                global_weights_frozen=(global_weights_frozen if cfg.mu > 0 else None),
                device=device,
            )

            local_state_dicts.append({k: v.detach().clone() for k, v in client_model.state_dict().items()})
            local_weights_for_agg.append(len(client_indices[cid]))
            round_train_losses.append(stats["train_loss"])
            round_train_sizes.append(len(client_indices[cid]))

        # 5) AGGREGATION — weighted by client dataset size
        new_global_state = weighted_average_state_dicts(local_state_dicts, local_weights_for_agg)
        global_model.load_state_dict(new_global_state)

        # 6) VALIDATION — every round
        val = evaluate(global_model, val_loader, device, num_classes=cfg.num_classes)

        # Weighted train loss (size-weighted across sampled clients)
        denom = sum(round_train_sizes) or 1
        weighted_train_loss = sum(s * l for s, l in zip(round_train_sizes, round_train_losses)) / denom

        row: Dict = {
            "seed": cfg.seed,
            "algorithm": cfg.algorithm,
            "mu": cfg.mu,
            "local_epochs": cfg.local_epochs,
            "round": r,
            "n_sampled": len(sampled),
            "train_loss": weighted_train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        for c, f1c in enumerate(val["per_class_f1"]):
            row[f"val_f1_class_{c}"] = float(f1c)
        history_rows.append(row)

        # 7) BEST-VAL CHECKPOINTING — never tune on test
        if val["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val["macro_f1"]
            best_state_dict = copy.deepcopy(global_model.state_dict())
            best_round = r

    # 8) FINAL TEST AT BEST-VAL CHECKPOINT (once)
    test_metrics: Dict | None = None
    if best_state_dict is not None:
        test_model = model_builder().to(device)
        test_model.load_state_dict(best_state_dict)
        test_metrics = evaluate(test_model, test_loader, device, num_classes=cfg.num_classes)
        test_metrics["selected_round"] = best_round
        test_metrics["best_val_macro_f1"] = best_val_macro_f1

    history_df = pd.DataFrame(history_rows)
    return {
        "config": asdict(cfg),
        "history": history_df,
        "test_metrics": test_metrics,
        "best_state_dict": best_state_dict,
        "system_het_schedule_summary": summarise_schedule(epoch_schedule),
    }


# ---------------------------------------------------------------------------
# Convenience: dump per-round CSV + best-test JSON
# ---------------------------------------------------------------------------

def save_run_outputs(
    result: Dict,
    out_dir: str | "Path",
    extra_metadata: Dict | None = None,
) -> None:
    """Write history CSV and the (single) best-val test_metrics JSON.

    Parameters
    ----------
    result : dict
        Output of `run_fl()`; contains `config`, `history`, `test_metrics`.
    out_dir : str or Path
        Directory to write into.
    extra_metadata : dict, optional
        Provenance fields that aren't carried inside FLConfig but that
        the caller (CLI runner) knows. Merged into the test_at_best JSON
        so each output file is fully self-documenting. Typical keys:
        partition, image_size, npz_path, framework, framework_version,
        loss_type. Keys override anything coming from `cfg` only when
        they are missing; existing cfg keys are preserved otherwise.
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    # If a non-trivial system-heterogeneity scenario is in use, tag the
    # filename so system-het runs are distinguishable from the headline
    # statistical-heterogeneity runs even when they share the same out-dir.
    sh = cfg.get("system_het", {}) if isinstance(cfg, dict) else {}
    sh_mode = sh.get("mode", "uniform") if isinstance(sh, dict) else "uniform"
    sh_tag = "" if sh_mode == "uniform" else f"_sh-{sh_mode}"
    stem = f"{cfg['algorithm']}_mu{cfg['mu']}_E{cfg['local_epochs']}{sh_tag}_s{cfg['seed']}"

    # Per-round history
    result["history"].to_csv(out / f"history_{stem}.csv", index=False)
    # Best-val test metrics (single row) — include full provenance
    if result["test_metrics"] is not None:
        payload = {
            **result["test_metrics"],
            **cfg,
            "system_het_schedule_summary":
                result.get("system_het_schedule_summary"),
        }
        # Provenance fields not carried by FLConfig — partition, image
        # size, dataset path, framework, loss choice. Set by the caller.
        if extra_metadata:
            for k, v in extra_metadata.items():
                payload.setdefault(k, v)
        with open(out / f"test_at_best_{stem}.json", "w") as f:
            json.dump(payload, f, indent=2)
