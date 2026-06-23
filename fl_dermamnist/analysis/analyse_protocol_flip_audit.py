"""Experiment E1 — Protocol-flip audit.

Re-analyses every (experiment × condition × seed) cell under three
reporting protocols and quantifies how often the FedProx-vs-FedAvg
ranking flips:

  A. best-val      : val_macro_f1 at the best-validation round
                     (the current default protocol)
  B. final-round   : val_macro_f1 at the last round of training
  C. last-K-mean   : mean val_macro_f1 over the final K=10 rounds

We compare WITHIN the validation split (not test) for protocol
sensitivity since (a) test is only available at the best-val
checkpoint, (b) the question is about reporting protocol, not
generalisation, and (c) staying on val isolates the question cleanly.

Sources of the gap:
  - "Not All FL Algorithms Are Created Equal" (Charles et al.,
    arXiv:2403.17287, 2024) §3 explicitly identifies that training
    instability and reporting protocols are under-studied:
    "performance stability across clients and training instability
    are under-reported in the FL benchmarking literature."
  - FLamby (arXiv:2210.04620, NeurIPS 2022) acknowledges that
    best-val is a common alternative but does not quantify
    protocol sensitivity.

Output:
  - protocol_flip_summary.csv    : full long-format table
  - protocol_flip_pivot.csv       : compact pivot (condition × protocol)
  - F_protocol_flip_rates.{pdf,png}: 2-panel visualisation
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS = REPO_ROOT / "fl_dermamnist/results"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

K = 10  # last-K window for protocol C


# ----------------------------------------------------------------
# Experiment registry: which directories to scan, and how to define
# the "comparison pair" (a fedavg-like config vs a fedprox-like config)
# within each experiment.
# ----------------------------------------------------------------
EXPERIMENTS = [
    # (label, directory, baseline_filter, treatment_filter)
    # filter is a callable on (algorithm, mu, filename_stem) → bool

    dict(
        label="perfect_storm_L4",
        root=RESULTS / "fedprox_perfect_storm_L4",
        recursive=False,
        baseline=lambda algo, mu, stem: algo == "fedavg" and "_drop" in stem,
        treatment=lambda algo, mu, stem: algo == "fedprox" and abs(mu - 1.0) < 1e-9 and "_drop" not in stem,
        baseline_label="FedAvg + drop",
        treatment_label="FedProx μ=1.0 + γ-inexact",
    ),
    dict(
        label="li2020_asymmetric_L4",
        root=RESULTS / "li2020_asymmetric_L4",
        recursive=False,
        baseline=lambda algo, mu, stem: algo == "fedavg" and "sh-fixed_stragglers_drop" in stem,
        treatment=lambda algo, mu, stem: algo == "fedprox" and "sh-fixed_stragglers" in stem and "_drop" not in stem,
        baseline_label="FedAvg + drop + straggler",
        treatment_label="FedProx + γ-inexact + straggler",
    ),
    dict(
        label="node_pinned_L4",
        root=RESULTS / "node_pinned_L4",
        recursive=False,
        baseline=lambda algo, mu, stem: algo == "fedavg",
        treatment=lambda algo, mu, stem: algo == "fedprox",
        baseline_label="FedAvg (symmetric)",
        treatment_label="FedProx μ=0.01 (symmetric)",
    ),
    dict(
        label="extended_rounds_L3",
        root=RESULTS / "extended_rounds_L3",
        recursive=False,
        baseline=lambda algo, mu, stem: algo == "fedavg",
        treatment=lambda algo, mu, stem: algo == "fedprox",
        baseline_label="FedAvg @ 250r",
        treatment_label="FedProx μ=0.01 @ 250r",
    ),
    dict(
        label="asymmetric_mu_L4_vs_FedAvg",
        root=RESULTS / "asymmetric_mu_L4",
        recursive=False,
        baseline=lambda algo, mu, stem: algo == "fedavg",
        treatment=lambda algo, mu, stem: algo == "fedprox" and "muPC-c0m0.01-c1m0.0" in stem,
        baseline_label="FedAvg",
        treatment_label="FedProx anchor-large",
    ),
    dict(
        label="two_client_90_10_rare_stress",
        root=RESULTS / "two_client_90_10_rare_stress",
        recursive=False,
        baseline=lambda algo, mu, stem: algo == "fedavg",
        treatment=lambda algo, mu, stem: algo == "fedprox",
        baseline_label="FedAvg (baseline)",
        treatment_label="FedProx μ=0.01 (baseline)",
    ),
]


# Filename stems look like e.g.
#   history_fedprox_mu0.01_E20_sh-fixed_stragglers_drop_s42.csv
# We need to parse: algorithm, mu, seed, and keep the full stem for
# experiment-specific filters.
STEM_RE = re.compile(
    r"history_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+(?P<tail>.*)_s(?P<seed>\d+)\.csv"
)


def _parse_filename(path: Path) -> dict | None:
    m = STEM_RE.match(path.name)
    if m is None:
        return None
    return dict(
        algorithm=m.group("algo"),
        mu=float(m.group("mu")),
        seed=int(m.group("seed")),
        tail=m.group("tail"),
        stem=path.name.replace("history_", "").replace(".csv", ""),
    )


def _compute_three_protocols(df: pd.DataFrame) -> dict:
    """Compute val_macro_f1 under three reporting protocols."""
    if "val_macro_f1" not in df.columns or len(df) == 0:
        return dict(best_val=np.nan, final_round=np.nan, last_K_mean=np.nan, n_rounds=0)
    s = df["val_macro_f1"].dropna()
    n_rounds = len(s)
    return dict(
        best_val=float(s.max()),
        final_round=float(s.iloc[-1]),
        last_K_mean=float(s.iloc[-K:].mean()),
        n_rounds=n_rounds,
    )


# ----------------------------------------------------------------
# 1. Build the per-run table
# ----------------------------------------------------------------
rows = []
for exp in EXPERIMENTS:
    if not exp["root"].exists():
        print(f"  skip (no dir): {exp['label']}")
        continue
    files = list(exp["root"].glob("history_*.csv"))
    for f in files:
        meta = _parse_filename(f)
        if meta is None:
            continue
        df = pd.read_csv(f)
        protos = _compute_three_protocols(df)
        is_baseline = exp["baseline"](meta["algorithm"], meta["mu"], meta["stem"])
        is_treatment = exp["treatment"](meta["algorithm"], meta["mu"], meta["stem"])
        if not (is_baseline or is_treatment):
            continue
        rows.append(dict(
            experiment=exp["label"],
            condition_role=("baseline" if is_baseline else "treatment"),
            algorithm=meta["algorithm"],
            mu=meta["mu"],
            seed=meta["seed"],
            file=f.name,
            **protos,
        ))
df_runs = pd.DataFrame(rows)
df_runs.to_csv(OUT_DIR / "protocol_flip_runs.csv", index=False)
print(f"Wrote {OUT_DIR/'protocol_flip_runs.csv'}  ({len(df_runs)} runs across {df_runs['experiment'].nunique()} experiments)")
print()

# ----------------------------------------------------------------
# 2. Compute per-(experiment, seed) Δ under each protocol and flag flips
# ----------------------------------------------------------------
flip_rows = []
for (exp, seed), grp in df_runs.groupby(["experiment", "seed"]):
    bs = grp[grp["condition_role"] == "baseline"]
    ts = grp[grp["condition_role"] == "treatment"]
    if len(bs) != 1 or len(ts) != 1:
        # Skip seeds where we don't have exactly one baseline and one treatment
        continue
    b = bs.iloc[0]
    t = ts.iloc[0]
    delta_A = t["best_val"] - b["best_val"]
    delta_B = t["final_round"] - b["final_round"]
    delta_C = t["last_K_mean"] - b["last_K_mean"]
    signs = (np.sign(delta_A), np.sign(delta_B), np.sign(delta_C))
    n_distinct = len(set(s for s in signs if s != 0))
    flip = n_distinct > 1
    flip_rows.append(dict(
        experiment=exp, seed=seed,
        baseline_label=next(e["baseline_label"] for e in EXPERIMENTS if e["label"] == exp),
        treatment_label=next(e["treatment_label"] for e in EXPERIMENTS if e["label"] == exp),
        delta_best_val=delta_A,
        delta_final_round=delta_B,
        delta_last_K_mean=delta_C,
        sign_best_val=int(signs[0]),
        sign_final_round=int(signs[1]),
        sign_last_K_mean=int(signs[2]),
        flips_between_protocols=flip,
        max_abs_protocol_diff=max(abs(delta_A - delta_B), abs(delta_A - delta_C), abs(delta_B - delta_C)),
    ))
df_flips = pd.DataFrame(flip_rows)
df_flips.to_csv(OUT_DIR / "protocol_flip_summary.csv", index=False)
print(f"Wrote {OUT_DIR/'protocol_flip_summary.csv'}  ({len(df_flips)} (experiment, seed) pairs)")
print()
print(df_flips[["experiment", "seed", "delta_best_val", "delta_final_round",
                "delta_last_K_mean", "flips_between_protocols"]].to_string(index=False))

# ----------------------------------------------------------------
# 3. Summary statistics
# ----------------------------------------------------------------
print()
print("=" * 80)
print("HEADLINE: protocol-flip rate")
print("=" * 80)
total = len(df_flips)
n_flips = int(df_flips["flips_between_protocols"].sum())
flip_rate = n_flips / total if total else 0.0
print(f"  Across {total} (experiment, seed) pairs:")
print(f"  {n_flips} cells flip sign between at least two protocols ({flip_rate*100:.1f}%)")
print()
print("Per-experiment breakdown:")
for exp, grp in df_flips.groupby("experiment"):
    n = len(grp)
    nf = int(grp["flips_between_protocols"].sum())
    rate = nf / n if n else 0
    avg_diff = float(grp["max_abs_protocol_diff"].mean())
    print(f"  {exp:<35} : {nf}/{n} flips ({rate*100:>4.0f}%), avg max-protocol-Δ = {avg_diff:.4f}")

# ----------------------------------------------------------------
# 4. Compact pivot — Δ per protocol per experiment (means across seeds)
# ----------------------------------------------------------------
pivot = df_flips.groupby("experiment").agg(
    n_seeds=("seed", "count"),
    mean_delta_best_val=("delta_best_val", "mean"),
    mean_delta_final_round=("delta_final_round", "mean"),
    mean_delta_last_K_mean=("delta_last_K_mean", "mean"),
    sd_delta_best_val=("delta_best_val", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
    flip_rate=("flips_between_protocols", "mean"),
).reset_index()
pivot.to_csv(OUT_DIR / "protocol_flip_pivot.csv", index=False)
print()
print("Pivot (Δ_treatment_minus_baseline averaged over seeds):")
print(pivot.to_string(index=False))

# ----------------------------------------------------------------
# 5. Figure: 2-panel visualisation
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.30})

# Panel A: per-experiment Δ under three protocols (grouped bars)
experiments_in_order = pivot["experiment"].tolist()
x = np.arange(len(experiments_in_order))
w = 0.27
colors = {"best_val": "#3D5A80", "final_round": "#C03A2B", "last_K_mean": "#C9A227"}

for i, proto in enumerate(["best_val", "final_round", "last_K_mean"]):
    means = pivot[f"mean_delta_{proto}"].values
    bars = axA.bar(
        x + (i - 1) * w, means, w,
        color=colors[proto], edgecolor="white", linewidth=0.6,
        label={"best_val": "best-val (default)",
               "final_round": "final-round",
               "last_K_mean": f"last-{K}-mean"}[proto],
    )
axA.axhline(0, color="#444", linewidth=0.8)
axA.set_xticks(x)
axA.set_xticklabels(experiments_in_order, rotation=22, ha="right", fontsize=8)
axA.set_ylabel(r"Δ macro-F1 (treatment − baseline)")
axA.set_title("(a) Per-experiment treatment-baseline Δ under three reporting protocols",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.legend(loc="upper right", frameon=False, fontsize=9)

# Panel B: flip-rate bar chart
flip_rates = pivot["flip_rate"].values
axB.bar(x, flip_rates * 100,
        color=["#1f6f3f" if v == 0 else "#b04040" for v in flip_rates],
        edgecolor="white", linewidth=0.6)
for i, v in enumerate(flip_rates):
    n_seeds = int(pivot.iloc[i]["n_seeds"])
    axB.text(i, v * 100 + 1.5, f"{int(v*n_seeds)}/{n_seeds}",
             ha="center", va="bottom", fontsize=8.5)
axB.axhline(0, color="#444", linewidth=0.8)
axB.set_xticks(x)
axB.set_xticklabels(experiments_in_order, rotation=22, ha="right", fontsize=8)
axB.set_ylabel("Flip rate (% of seeds with cross-protocol sign change)")
axB.set_title("(b) Protocol-flip incidence by experiment",
              loc="left", fontweight="bold", fontsize=11)
axB.set_ylim(0, max(105, max(flip_rates) * 100 + 15))
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Reporting-protocol sensitivity audit "
             "— Charles et al. 2024 (arXiv:2403.17287) framing",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_protocol_flip_rates.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_protocol_flip_rates.pdf'}")
print()
print("Done.")
