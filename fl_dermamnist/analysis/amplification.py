"""Offline FedNova amplification diagnostic (Stage 1 of the random-τ thesis plan).

Reconstructs, per (round, client), the FedNova per‑client contribution to the
global step and the amplification factor relative to FedAvg, from the scalars
already logged by `run_one_fednova_flower.py` (`client_update_norms_*.csv`:
round, client_id, update_norm, n_samples, local_epochs, tau). Where the newer
`aggregation_client_diag_*.csv` / `aggregation_round_diag_*.csv` are present
(runs made after the Stage‑0 instrumentation), those DIRECTLY‑LOGGED values are
used instead and the `source` column is set to "logged".

Definitions (mirroring `fl_flower/strategy_fednova.py`):
    a_i              = fednova_normaliser(tau_i, m)          [reconstructed from tau]
    p_i              = n_i / sum_j n_j
    a_eff            = sum_i p_i * a_i
    contribution_norm = a_eff * p_i * ||delta_i|| / a_i      [client's share of the step]
    amp_vs_fedavg     = a_eff / a_i                          [weight inflation vs FedAvg]

IMPORTANT — what is measured vs reconstructed:
  * raw_update_norm (||delta_i||), tau, n_samples  -> DIRECTLY LOGGED (client side).
  * a_i, a_eff, contribution_norm, amp_vs_fedavg   -> RECONSTRUCTED offline from tau
    + the per‑seed client momentum read from the run JSON (unless an
    aggregation_*_diag CSV is present, in which case they are LOGGED).
  * global_update_norm (||g_t||) is NOT reconstructible from per‑client norms
    (it needs the actual vectors); it is reported ONLY when logged. Otherwise NaN.

Usage:
    python -m fl_dermamnist.analysis.amplification \\
        --dir fl_dermamnist/results/system_het_random_fednova \\
        --out fl_dermamnist/results/system_het_random_fednova/analysis/amplification \\
        --collapse-threshold 0.20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Use the canonical normaliser when importable; otherwise fall back to an
# identical local copy (kept in sync; verified by tests/test_fednova_aggregation
# style checks). The fallback avoids a hard dependency on flwr for analysis.
try:  # pragma: no cover - exercised implicitly
    from fl_dermamnist.fl_flower.strategy_fednova import fednova_normaliser
except Exception:  # pragma: no cover
    def fednova_normaliser(tau: float, momentum: float) -> float:
        tau = float(tau)
        m = float(momentum)
        if m <= 0.0:
            return tau
        if abs(1.0 - m) < 1e-12:
            return tau * (tau + 1.0) / 2.0
        return (tau * (1.0 - m) - m * (1.0 - m ** tau)) / ((1.0 - m) ** 2)


def _seed_from_stem(stem: str) -> Optional[int]:
    m = re.search(r"_s(\d+)$", stem)
    return int(m.group(1)) if m else None


def _read_momentum(results_dir: str, stem: str, default: float = 0.9) -> tuple[float, bool]:
    """Return (momentum, was_read). Reads the run JSON; default if missing."""
    p = os.path.join(results_dir, f"test_at_best_{stem}.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            if "momentum" in d:
                return float(d["momentum"]), True
        except Exception:
            pass
    return float(default), False


def _detect_collapse_round(history_csv: str, threshold: float) -> Optional[int]:
    """First round where val macro-F1 falls below `threshold` AFTER having been
    at/above it. None = never reached `threshold` (never trained) OR never fell
    below it (healthy); the best/final macro-F1 columns disambiguate. Mirrors
    run_one_fednova_flower.py:_detect_collapse_round."""
    if not os.path.exists(history_csv):
        return None
    df = pd.read_csv(history_csv)
    if "val_macro_f1" not in df or "round" not in df or df.empty:
        return None
    df = df.sort_values("round")
    seen_healthy = False
    for _, r in df.iterrows():
        vmf = float(r["val_macro_f1"])
        if vmf >= threshold:
            seen_healthy = True
        elif seen_healthy:
            return int(r["round"])
    return None


def _best_final_macro(results_dir: str, stem: str) -> Dict[str, Optional[float]]:
    out = {"best_macro_f1": None, "final_macro_f1": None}
    pb = os.path.join(results_dir, f"test_at_best_{stem}.json")
    if os.path.exists(pb):
        try:
            out["best_macro_f1"] = float(json.load(open(pb)).get("macro_f1"))
        except Exception:
            pass
    pf = os.path.join(results_dir, f"test_at_final_{stem}.json")
    if os.path.exists(pf):
        try:
            out["final_macro_f1"] = float(json.load(open(pf)).get("final_macro_f1"))
        except Exception:
            pass
    return out


def _per_round_from_logged(agg_round_csv: str, agg_client_csv: str) -> Optional[pd.DataFrame]:
    """Use directly-logged aggregation diagnostics if available."""
    if not (os.path.exists(agg_round_csv) and os.path.exists(agg_client_csv)):
        return None
    rnd = pd.read_csv(agg_round_csv)
    cli = pd.read_csv(agg_client_csv)
    max_amp = cli.groupby("round")["amp_vs_fedavg"].max().rename("max_amp")
    out = rnd.merge(max_amp, on="round", how="left")
    out["source"] = "logged"
    return out


def _per_round_reconstructed(client_norms_csv: str, momentum: float) -> pd.DataFrame:
    """Reconstruct per-round amplification stats from client_update_norms_*.csv."""
    df = pd.read_csv(client_norms_csv)
    rows = []
    for rd, g in df.groupby("round"):
        n_tot = float(g["n_samples"].sum())
        taus = g["tau"].astype(float).to_numpy()
        norms = g["update_norm"].astype(float).to_numpy()
        ns = g["n_samples"].astype(float).to_numpy()
        cids = g["client_id"].astype(int).to_numpy()
        a = np.array([fednova_normaliser(t, momentum) for t in taus])
        p = ns / n_tot
        a_eff = float(np.sum(p * a))
        u = np.where(a > 0, p * norms / a, 0.0)  # a_eff-independent contribution share part
        u_tot = float(np.sum(u))
        amp = np.where(a > 0, a_eff / a, np.nan)
        strag_idx = int(np.argmin(taus))
        dom_idx = int(np.argmax(u))
        rows.append({
            "round": int(rd),
            "a_eff": a_eff,
            "mean_tau": float(np.sum(taus * p)),
            "global_update_norm": float("nan"),  # not reconstructible offline
            "straggler_share": (float(u[strag_idx] / u_tot) if u_tot > 0 else float("nan")),
            "dominating_cid": int(cids[dom_idx]) if u_tot > 0 else -1,
            "max_contribution_share": (float(np.max(u) / u_tot) if u_tot > 0 else float("nan")),
            "max_amp": float(np.nanmax(amp)) if amp.size else float("nan"),
            "server_lr": float("nan"),
            "source": "reconstructed",
        })
    return pd.DataFrame(rows).sort_values("round")


def main():
    ap = argparse.ArgumentParser(description="Offline FedNova amplification diagnostic.")
    ap.add_argument("--dir", required=True, help="results dir with client_update_norms_*.csv")
    ap.add_argument("--out", required=True, help="output dir for analysis CSVs/plots")
    ap.add_argument("--collapse-threshold", type=float, default=0.20)
    ap.add_argument("--early-rounds", type=int, default=10,
                    help="window (rounds 1..N) summarised as 'early' amplification")
    args = ap.parse_args()

    results_dir = args.dir
    norm_files = sorted(glob.glob(os.path.join(results_dir, "client_update_norms_*.csv")))
    if not norm_files:
        print(f"[amplification] No client_update_norms_*.csv found in {results_dir!r}.")
        print("  Files present:")
        for f in sorted(glob.glob(os.path.join(results_dir, "*")))[:50]:
            print("   -", os.path.basename(f))
        print("  Not guessing. Point --dir at a FedNova run directory.")
        return

    os.makedirs(args.out, exist_ok=True)
    by_round_all: List[pd.DataFrame] = []
    seed_summ: List[Dict] = []

    for nf in norm_files:
        stem = os.path.basename(nf)[len("client_update_norms_"):-len(".csv")]
        seed = _seed_from_stem(stem)
        momentum, m_read = _read_momentum(results_dir, stem)
        agg_round = os.path.join(results_dir, f"aggregation_round_diag_{stem}.csv")
        agg_client = os.path.join(results_dir, f"aggregation_client_diag_{stem}.csv")
        pr = _per_round_from_logged(agg_round, agg_client)
        if pr is None:
            pr = _per_round_reconstructed(nf, momentum)
        pr.insert(0, "seed", seed)
        source = str(pr["source"].iloc[0]) if "source" in pr and len(pr) else "reconstructed"
        by_round_all.append(pr)

        hist = os.path.join(results_dir, f"history_{stem}.csv")
        collapse_round = _detect_collapse_round(hist, args.collapse_threshold)
        bf = _best_final_macro(results_dir, stem)
        early = pr[pr["round"] <= args.early_rounds]
        seed_summ.append({
            "seed": seed,
            "stem": stem,
            "momentum_used": momentum,
            "momentum_from_json": m_read,
            "amp_source": source,
            "a_eff_mean": float(pr["a_eff"].mean()),
            "max_amp_overall": float(pr["max_amp"].max()),
            "max_amp_early": float(early["max_amp"].max()) if len(early) else float("nan"),
            "straggler_share_mean": float(pr["straggler_share"].mean(skipna=True)),
            "straggler_share_early_mean": float(early["straggler_share"].mean(skipna=True)) if len(early) else float("nan"),
            "max_contribution_share_mean": float(pr["max_contribution_share"].mean(skipna=True)),
            "collapse_round": collapse_round,
            "best_macro_f1": bf["best_macro_f1"],
            "final_macro_f1": bf["final_macro_f1"],
        })

    by_round = pd.concat(by_round_all, ignore_index=True)
    summary = pd.DataFrame(seed_summ).sort_values("seed")
    onset = summary[["seed", "collapse_round", "max_amp_early", "straggler_share_early_mean",
                     "best_macro_f1", "final_macro_f1", "amp_source"]].copy()

    by_round_path = os.path.join(args.out, "amplification_by_round.csv")
    summary_path = os.path.join(args.out, "amplification_by_seed_summary.csv")
    onset_path = os.path.join(args.out, "collapse_onset_vs_amplification.csv")
    by_round.to_csv(by_round_path, index=False)
    summary.to_csv(summary_path, index=False)
    onset.to_csv(onset_path, index=False)

    n_logged = (summary["amp_source"] == "logged").sum()
    n_recon = (summary["amp_source"] == "reconstructed").sum()
    print(f"[amplification] {len(summary)} seeds  "
          f"({n_logged} logged, {n_recon} reconstructed-from-tau).")
    print(f"  wrote: {by_round_path}")
    print(f"  wrote: {summary_path}")
    print(f"  wrote: {onset_path}")

    # Optional plots (skipped silently if matplotlib is unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable — skipping PNGs)")
        return

    # 1) collapse onset vs early amplification
    fig, ax = plt.subplots(figsize=(6, 4.5))
    od = onset.copy()
    od["collapse_round_plot"] = od["collapse_round"].fillna(-5)  # -5 = "no collapse"
    sc = ax.scatter(od["max_amp_early"], od["collapse_round_plot"],
                    c=od["best_macro_f1"].fillna(0.0), cmap="viridis", s=60)
    for _, r in od.iterrows():
        ax.annotate(str(int(r["seed"])) if pd.notna(r["seed"]) else "?",
                    (r["max_amp_early"], r["collapse_round_plot"]), fontsize=7)
    ax.set_xlabel("early-round max amplification (a_eff / a_i)")
    ax.set_ylabel("collapse round  (−5 = no collapse)")
    ax.set_title("Collapse onset vs early amplification")
    fig.colorbar(sc, label="best macro-F1")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "collapse_onset_vs_amplification.png"), dpi=130)
    plt.close(fig)

    # 2) dominant-client share by round (one line per seed)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sd, g in by_round.groupby("seed"):
        ax.plot(g["round"], g["max_contribution_share"], label=str(sd), alpha=0.7, lw=1)
    ax.set_xlabel("round"); ax.set_ylabel("max single-client contribution share")
    ax.set_title("Dominant-client share by round")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dominant_client_share_by_round.png"), dpi=130)
    plt.close(fig)

    # 3) amplification trajectories by seed
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sd, g in by_round.groupby("seed"):
        ax.plot(g["round"], g["max_amp"], label=str(sd), alpha=0.7, lw=1)
    ax.set_xlabel("round"); ax.set_ylabel("max amplification (a_eff / a_i)")
    ax.set_title("Amplification trajectories by seed")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "amplification_trajectories_by_seed.png"), dpi=130)
    plt.close(fig)
    print(f"  wrote 3 PNGs to {args.out}")


if __name__ == "__main__":
    main()
