"""Experiment 10 — Li 2020 §5.2 asymmetric protocol analysis ("FedProx wins").

Reads 12 runs at
  mnist_dermnist/results/li2020_asymmetric_L4/
       test_at_best_{fedavg,fedprox}_..._s{42,123,456}.json

and decomposes the 4 conditions × 3 seeds into the Li 2020 §5.2
mechanism reading:
  - Condition 1: FedAvg baseline (no stragglers)
  - Condition 2: FedAvg + drop-stragglers (FedAvg arm of Li 2020 §5.2)
  - Condition 3: FedProx + γ-inexact (FedProx arm of Li 2020 §5.2)
  - Condition 4: FedProx + drop-stragglers (control)

Key reported quantities:
  - Headline gap (3 − 2)   — the Li 2020 §5.2 protocol effect
  - Algorithm gap (3 − 4)  — γ-inexact handling effect, isolating algorithm
  - Protocol-only effect (1 − 2 on FedAvg, 1 − 3 on FedProx)
  - Per-class F1 — especially rare classes that FedAvg+drop destroys

Outputs:
  - li2020_asymmetric_L4_summary.csv
  - F_li2020_asymmetric_L4.{pdf,png} — 2-panel (macro-F1 + per-class)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
IN_DIR = REPO_ROOT / "mnist_dermnist/results/li2020_asymmetric_L4"
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
# Stems mirror run_one_flower.py's tag conventions:
#   sh_tag = '' for uniform; '_sh-fixed_stragglers' for fixed-stragglers mode.
#   drop_tag = '_drop' when --drop-stragglers is set, else ''.
CONDITIONS = [
    ("1. FedAvg baseline (no straggler)",        "fedavg_mu0.0_E20",                              0),
    ("2. FedAvg + drop-stragglers (Li §5.2 FA)", "fedavg_mu0.0_E20_sh-fixed_stragglers_drop",     1),
    ("3. FedProx + γ-inexact (Li §5.2 FP) ⭐",    "fedprox_mu0.01_E20_sh-fixed_stragglers",        2),
    ("4. FedProx + drop control",                "fedprox_mu0.01_E20_sh-fixed_stragglers_drop",   3),
]


def _read(seed: int, stem: str) -> dict | None:
    f = IN_DIR / f"test_at_best_{stem}_s{seed}.json"
    if not f.exists():
        return None
    return json.load(open(f))


rows = []
for label, stem, idx in CONDITIONS:
    for seed in SEEDS:
        x = _read(seed, stem)
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
df.to_csv(OUT_DIR / "li2020_asymmetric_L4_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'li2020_asymmetric_L4_summary.csv'}  ({len(df)} runs)")
print()
print(df[["condition", "seed", "selected_round", "macro_f1",
         "rare_avg_f1", "f1_vascular", "f1_melanoma", "f1_dermato"]].to_string(index=False))

# --- 3-seed mean ± SD per condition ---
print()
print("=" * 100)
print("3-seed mean ± SD per condition (L4, two_client_90_10_rare_stress)")
print("=" * 100)
summary = []
for label, _, idx in CONDITIONS:
    sub = df[df["condition"] == label]
    if not len(sub):
        continue
    n = len(sub)

    def m_sd(col):
        mean = float(sub[col].mean())
        sd = float(sub[col].std(ddof=1)) if n > 1 else 0.0
        return mean, sd

    macro_m, macro_s = m_sd("macro_f1")
    rare_m,  rare_s  = m_sd("rare_avg_f1")
    vasc_m,  vasc_s  = m_sd("f1_vascular")
    summary.append(dict(condition=label, cond_idx=idx, n=n,
                        macro_mean=macro_m, macro_sd=macro_s,
                        rare_mean=rare_m,   rare_sd=rare_s,
                        vasc_mean=vasc_m,   vasc_sd=vasc_s))
    print(f"  {label:<48}  n={n}  macro={macro_m:.4f}±{macro_s:.4f}  "
          f"rare-avg={rare_m:.4f}±{rare_s:.4f}  vasc={vasc_m:.4f}±{vasc_s:.4f}")

# --- Headline mechanism reading ---
print()
print("=" * 100)
print("Mechanism reading (Li 2020 §5.2 protocol)")
print("=" * 100)
sumdf = pd.DataFrame(summary)


def by_idx(i):
    sub = sumdf[sumdf["cond_idx"] == i]
    return sub.iloc[0] if len(sub) else None


b = by_idx(0)  # 1. FedAvg baseline
fa_drop = by_idx(1)  # 2. FedAvg + drop
fp_inexact = by_idx(2)  # 3. FedProx + γ-inexact
fp_drop = by_idx(3)  # 4. FedProx + drop control

if fp_inexact is not None and fa_drop is not None:
    d_macro = fp_inexact["macro_mean"] - fa_drop["macro_mean"]
    d_rare  = fp_inexact["rare_mean"]  - fa_drop["rare_mean"]
    d_vasc  = fp_inexact["vasc_mean"]  - fa_drop["vasc_mean"]
    pooled_sd = np.sqrt(0.5 * (fp_inexact["macro_sd"] ** 2 + fa_drop["macro_sd"] ** 2))
    print()
    print(f"  HEADLINE GAP (Li 2020 §5.2 result, condition 3 minus condition 2):")
    print(f"    Δ macro-F1     = {d_macro:+.4f}  (pooled SD = {pooled_sd:.4f})")
    print(f"    Δ rare-avg-F1  = {d_rare:+.4f}")
    print(f"    Δ vascular-F1  = {d_vasc:+.4f}")
    if d_macro > 0.05 and d_macro > 2 * pooled_sd:
        print(f"    → ✅ FedProx CLEARLY beats FedAvg under the asymmetric protocol.")
    elif d_macro > 0:
        print(f"    → FedProx ahead but small / within noise. Try higher E or μ=0.1.")
    else:
        print(f"    → FedProx not ahead under this protocol — re-examine setup.")

if fp_inexact is not None and fp_drop is not None:
    d_proto = fp_inexact["macro_mean"] - fp_drop["macro_mean"]
    print()
    print(f"  ALGORITHM-ISOLATION (condition 3 minus condition 4, both FedProx):")
    print(f"    Δ macro-F1     = {d_proto:+.4f}")
    print(f"    → The γ-inexact mechanism contributes {d_proto:+.4f} of macro-F1 to FedProx.")

if b is not None and fa_drop is not None:
    d_protocol_fa = b["macro_mean"] - fa_drop["macro_mean"]
    print()
    print(f"  COST OF DROP-STRAGGLERS ON FedAvg (condition 1 minus condition 2):")
    print(f"    Δ macro-F1     = {d_protocol_fa:+.4f}  (positive = drop hurts FedAvg)")
    print(f"    rare-class collapse: {b['rare_mean']:.4f} → {fa_drop['rare_mean']:.4f}")

# --- Figure: 2-panel ---
COND_COLORS = ["#7FBF94", "#C03A2B", "#3D5A80", "#C9A227"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.22})

# Panel A: macro-F1 dot plot per condition
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
    axA.text(j, mean + sd + 0.005, f"{mean:.3f}\n±{sd:.3f}", ha="center",
             va="bottom", fontsize=8, color=color, fontweight="bold")
axA.set_xticks(range(len(present)))
axA.set_xticklabels([CONDITIONS[i][0] for i in present], rotation=12, ha="right", fontsize=8)
axA.set_ylabel("Test macro-F1")
axA.set_title("(a) Macro-F1 across the 4 conditions (3 seeds; Li 2020 §5.2)",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Annotate the headline gap (condition 3 − condition 2).
if 1 in present and 2 in present:
    s2 = df[df["cond_idx"] == 1]["macro_f1"].mean()
    s3 = df[df["cond_idx"] == 2]["macro_f1"].mean()
    axA.annotate("", xy=(2, s3 - 0.01), xytext=(1, s2 + 0.01),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    axA.text(1.5, (s2 + s3) / 2 + 0.04, f"Headline gap\nΔ = {s3 - s2:+.3f}",
             ha="center", va="bottom", fontsize=9, color="#444", fontweight="bold")

# Panel B: per-class F1, grouped bars
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
axB.set_title("(b) Per-class F1 — drop-stragglers makes FedAvg blind to rare classes (shaded)",
              loc="left", fontweight="bold", fontsize=11)
axB.legend(loc="upper left", frameon=False, fontsize=7)
axB.set_ylim(0, 1.0)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Li et al. 2020 §5.2 asymmetric-protocol replication on L4 — the canonical 'FedProx wins' setting",
             fontsize=11, fontweight="bold", y=1.03)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_li2020_asymmetric_L4.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_li2020_asymmetric_L4.pdf'}")
print()
print("Done.")
