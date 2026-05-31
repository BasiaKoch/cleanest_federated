"""Experiment 11 — FedProx perfect-storm L4 analysis.

Reads 9 runs at
  mnist_dermnist/results/fedprox_perfect_storm_L4/
       test_at_best_fedavg_mu0.0_E20_sh-random_stragglers_drop_s{42,123,456}.json
       test_at_best_fedprox_mu{1.0,0.01}_E20_sh-random_stragglers_s{42,123,456}.json

and produces a 3-condition × 3-seed comparison plus a literature-grounded
verdict. Compares against the thesis-baseline L4 data (from
node_pinned_L4/ if present) for the "configuration gap" diagnostic.

Outputs:
  - perfect_storm_L4_summary.csv
  - F_fedprox_perfect_storm_L4.{pdf,png} — 2-panel
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
IN_DIR = REPO_ROOT / "mnist_dermnist/results/fedprox_perfect_storm_L4"
BASELINE_DIR = REPO_ROOT / "mnist_dermnist/results/node_pinned_L4"
OUT_DIR = IN_DIR / "analysis"
OUT_FIG = REPO_ROOT / "mnist_dermnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)

# (label, filename stem prefix, ordering index)
CONDITIONS = [
    ("1. FedAvg + drop (Li §5.2 FA arm)",     "fedavg_mu0.0_E20_sh-random_stragglers_drop",   0),
    ("2. FedProx μ=1.0 + γ-inexact ⭐",        "fedprox_mu1.0_E20_sh-random_stragglers",       1),
    ("3. FedProx μ=0.01 + γ-inexact (μ ablation)", "fedprox_mu0.01_E20_sh-random_stragglers", 2),
]


def _read(d: Path, seed: int, stem: str) -> dict | None:
    f = d / f"test_at_best_{stem}_s{seed}.json"
    if not f.exists():
        return None
    return json.load(open(f))


rows = []
for label, stem, idx in CONDITIONS:
    for seed in SEEDS:
        x = _read(IN_DIR, seed, stem)
        if x is None:
            print(f"  WARN: missing s{seed}/{stem}")
            continue
        pc = x.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            condition=label, cond_idx=idx, seed=seed,
            hostname=x.get("hostname", "?"),
            selected_round=x.get("selected_round"),
            macro_f1=x.get("macro_f1"),
            balanced_accuracy=x.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "perfect_storm_L4_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'perfect_storm_L4_summary.csv'}  ({len(df)} runs)")
print()
print(df[["condition", "seed", "macro_f1", "rare_avg_f1", "f1_vascular"]].to_string(index=False))

# --- 3-seed mean ± SD ---
print()
print("=" * 92)
print("Perfect-storm conditions: 3-seed mean ± SD")
print("=" * 92)
summary = []
for label, _, idx in CONDITIONS:
    sub = df[df["condition"] == label]
    if not len(sub):
        continue
    n = len(sub)
    def m_sd(col):
        m = float(sub[col].mean())
        s = float(sub[col].std(ddof=1)) if n > 1 else 0.0
        return m, s
    mm, ms = m_sd("macro_f1")
    rm, rs = m_sd("rare_avg_f1")
    vm, vs = m_sd("f1_vascular")  # f1_vascular comes from the **{} expansion above (CLASS_NAMES[6]=vascular)
    summary.append(dict(condition=label, cond_idx=idx, n=n,
                        macro_mean=mm, macro_sd=ms,
                        rare_mean=rm, rare_sd=rs,
                        vasc_mean=vm, vasc_sd=vs))
    print(f"  {label:<48}  macro={mm:.4f}±{ms:.4f}  rare={rm:.4f}±{rs:.4f}  vasc={vm:.4f}±{vs:.4f}")

# --- Decision rule (Li 2020 §5.3.2 prediction check) ---
print()
print("=" * 92)
print("Decision rule (Li 2020 §5.3.2 prediction)")
print("=" * 92)
sumdf = pd.DataFrame(summary)


def by_idx(i):
    sub = sumdf[sumdf["cond_idx"] == i]
    return sub.iloc[0] if len(sub) else None


fa = by_idx(0)
fp1 = by_idx(1)
fp001 = by_idx(2)

if fa is not None and fp1 is not None:
    d_macro = fp1["macro_mean"] - fa["macro_mean"]
    d_rare  = fp1["rare_mean"]  - fa["rare_mean"]
    pooled_sd = np.sqrt(0.5 * (fp1["macro_sd"] ** 2 + fa["macro_sd"] ** 2))
    print()
    print(f"  HEADLINE GAP (Li 2020 §5.3.2 prediction, condition 2 minus condition 1):")
    print(f"    Δ macro-F1  = {d_macro:+.4f}  (pooled SD = {pooled_sd:.4f})")
    print(f"    Δ rare-F1   = {d_rare:+.4f}")
    if d_macro > 0.10 and d_macro > 2 * pooled_sd:
        print(f"    → ✅ FedProx CLEARLY wins ({d_macro:+.4f} > 0.10 macro-F1 + 2·SD).")
        print(f"       Comparable to NIID-Bench Table III FMNIST #C=1 result (+17.7 pp).")
    elif d_macro > 0.05 and d_macro > 2 * pooled_sd:
        print(f"    → FedProx wins meaningfully ({d_macro:+.4f}; below the 22% headline but above noise).")
    elif d_macro > 0:
        print(f"    → FedProx ahead but small / within seed noise.")
    else:
        print(f"    → Perfect-storm setup did NOT produce a FedProx win — investigate.")

if fp1 is not None and fp001 is not None:
    d_mu = fp1["macro_mean"] - fp001["macro_mean"]
    print()
    print(f"  μ-SENSITIVITY (condition 2 minus condition 3, both FedProx in perfect-storm regime):")
    print(f"    Δ macro-F1  = {d_mu:+.4f}  (μ=1.0 minus μ=0.01)")
    if d_mu > 0.03:
        print(f"    → μ choice matters in this regime. NIID-Bench §V.B confirmed:")
        print(f"       small μ = small FedProx advantage; μ=1.0 unlocks the win.")
    elif abs(d_mu) <= 0.02:
        print(f"    → μ choice does NOT matter in this regime (FedProx win is from γ-inexact handling alone).")
    else:
        print(f"    → Weak μ sensitivity (within 1-2 SD).")

# --- Cross-reference to thesis-baseline L4 (node_pinned_L4) ---
baseline_rows = []
if BASELINE_DIR.exists():
    print()
    print("=" * 92)
    print("Cross-reference: thesis-baseline L4 (bs=32, mom=0.9, no stragglers)")
    print("=" * 92)
    for algo, mu in [("fedavg", "0.0"), ("fedprox", "0.01")]:
        macros = []
        for seed in SEEDS:
            x = _read(BASELINE_DIR, seed, f"{algo}_mu{mu}_E20")
            if x is not None:
                macros.append(x.get("macro_f1"))
        if macros:
            print(f"  baseline {algo} μ={mu}:  macro-F1 = {np.mean(macros):.4f} ± {np.std(macros, ddof=1) if len(macros) > 1 else 0:.4f}  (n={len(macros)})")
            baseline_rows.append(dict(algo=algo, mu=mu, mean=np.mean(macros),
                                       sd=np.std(macros, ddof=1) if len(macros) > 1 else 0))

    if len(baseline_rows) == 2 and fa is not None and fp1 is not None:
        b_fa = next(r for r in baseline_rows if r["algo"] == "fedavg")
        b_fp = next(r for r in baseline_rows if r["algo"] == "fedprox")
        b_gap = b_fp["mean"] - b_fa["mean"]
        s_gap = fp1["macro_mean"] - fa["macro_mean"]
        cfg_amplification = s_gap - b_gap
        print()
        print(f"  CONFIGURATION-CHANGE EFFECT (gap amplification):")
        print(f"    Thesis-baseline (FedProx − FedAvg)          = {b_gap:+.4f}")
        print(f"    Perfect-storm  (FedProx μ=1 − FedAvg+drop)  = {s_gap:+.4f}")
        print(f"    Amplification from config change             = {cfg_amplification:+.4f}")
        print(f"    → switching to literature-canonical hyperparameters and protocol")
        print(f"      amplifies the FedProx advantage by {cfg_amplification:+.4f} macro-F1.")

# --- Figure: 2-panel (macro-F1 dot plot + per-class bars) ---
COND_COLORS = ["#C03A2B", "#3D5A80", "#7FBF94"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.22})

# Panel A: macro-F1 dot plot
present = sorted(set(df["cond_idx"]))
for j, idx in enumerate(present):
    sub = df[df["cond_idx"] == idx]
    if not len(sub):
        continue
    label = sub["condition"].iloc[0]
    color = COND_COLORS[idx]
    xs = np.full(len(sub), j) + np.random.RandomState(idx).uniform(-0.08, 0.08, size=len(sub))
    axA.scatter(xs, sub["macro_f1"], c=color, s=85, edgecolor="white", linewidth=0.8, zorder=3)
    mean = float(sub["macro_f1"].mean())
    sd   = float(sub["macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
    axA.errorbar([j], [mean], yerr=[sd], fmt="_", color=color,
                 markersize=30, linewidth=2, capsize=10, zorder=2)
    axA.text(j, mean + sd + 0.008, f"{mean:.3f}\n±{sd:.3f}", ha="center",
             va="bottom", fontsize=8, color=color, fontweight="bold")
# Annotate the headline gap (cond 2 − cond 1).
if 0 in present and 1 in present:
    m0 = df[df["cond_idx"] == 0]["macro_f1"].mean()
    m1 = df[df["cond_idx"] == 1]["macro_f1"].mean()
    axA.annotate("", xy=(1, m1 - 0.01), xytext=(0, m0 + 0.01),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    axA.text(0.5, (m0 + m1) / 2 + 0.04, f"Headline gap\nΔ = {m1 - m0:+.3f}",
             ha="center", va="bottom", fontsize=9, color="#444", fontweight="bold")
axA.set_xticks(range(len(present)))
axA.set_xticklabels([CONDITIONS[i][0] for i in present], rotation=12, ha="right", fontsize=8)
axA.set_ylabel("Test macro-F1")
axA.set_title("(a) Macro-F1 in the literature-canonical regime (3 seeds)",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: per-class F1
x = np.arange(7)
group_w = 0.85
bar_w = group_w / max(len(present), 1)
for j, idx in enumerate(present):
    sub = df[df["cond_idx"] == idx]
    if not len(sub):
        continue
    label = sub["condition"].iloc[0]
    color = COND_COLORS[idx]
    means = [sub[f"f1_{CLASS_NAMES[c]}"].mean() for c in range(7)]
    sds   = [sub[f"f1_{CLASS_NAMES[c]}"].std(ddof=1) if len(sub) > 1 else 0.0 for c in range(7)]
    offset = (j - (len(present) - 1) / 2) * bar_w
    axB.bar(x + offset, means, bar_w, yerr=sds,
            color=color, edgecolor="white", linewidth=0.4, capsize=2,
            error_kw=dict(linewidth=0.8, ecolor="#333"),
            label=label)
for c in RARE_IDX:
    axB.axvspan(c - 0.5, c + 0.5, color="#C9A227", alpha=0.10)
axB.set_xticks(x); axB.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8.5)
axB.set_ylabel("Per-class test F1 (mean ± SD, 3 seeds)")
axB.set_title("(b) Per-class F1 — drop-stragglers blinds FedAvg to rare classes",
              loc="left", fontweight="bold", fontsize=11)
axB.legend(loc="upper left", frameon=False, fontsize=7)
axB.set_ylim(0, 1.0)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("FedProx perfect-storm L4 — literature-canonical replication "
             "(μ=1.0, bs=10, mom=0, 90% stragglers, Li 2020 §5.2 protocol)",
             fontsize=11, fontweight="bold", y=1.03)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_fedprox_perfect_storm_L4.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_fedprox_perfect_storm_L4.pdf'}")
print()
print("Done.")
