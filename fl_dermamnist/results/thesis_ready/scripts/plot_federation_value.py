"""Federation-value figure for the small-hospital case study.

Compares five configurations on the 2-client 90/10 partition, seed 42:
    1. Client 0 alone (local-only, no federation)
    2. Client 1 alone (local-only, no federation)
    3. FedAvg federated
    4. FedProx (mu = 0.01) federated
    5. Centralised pooled training

Two panels:
    (a) Macro-F1 per configuration, with the centralised reference
        drawn as a horizontal dashed line. Arrows annotate each client's
        federation lift (config -> federated mean).
    (b) Per-class test F1, grouped bars across all 5 configurations,
        with vertical bands highlighting the rare classes that live
        only on Client 1.

Output: F_federation_value.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
OUT = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# --- Load all five configurations -----------------------------------
PATHS = {
    "Client 0 alone":  REPO_ROOT / "fl_dermamnist/results/small_hospital_local_only/test_at_best_local_only_c0_E3000_s42.json",
    "Client 1 alone":  REPO_ROOT / "fl_dermamnist/results/small_hospital_local_only/test_at_best_local_only_c1_E3000_s42.json",
    "FedAvg":          REPO_ROOT / "fl_dermamnist/results/two_client_90_10_rare_stress/test_at_best_fedavg_mu0.0_E20_s42.json",
    r"FedProx ($\mu=0.01$)": REPO_ROOT / "fl_dermamnist/results/two_client_90_10_rare_stress/test_at_best_fedprox_mu0.01_E20_s42.json",
    "Centralised":     REPO_ROOT / "fl_dermamnist/results/centralised/centralised_seed42.json",
}

CONFIGS = list(PATHS.keys())
data = {name: json.load(open(p)) for name, p in PATHS.items()}

macro_f1 = [data[c]["macro_f1"] for c in CONFIGS]
per_class = np.array([data[c]["per_class_f1"] for c in CONFIGS])

CLASS_NAMES = [
    "Actinic\nkeratoses",
    "Basal cell\ncarcinoma",
    "Benign\nkeratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic\nnevi",
    "Vascular\nlesions",
]
RARE_CLASS_IDX = (3, 4, 6)   # held only by Client 1

# Color scheme matching the rest of the thesis.
COLORS = {
    "Client 0 alone":              "#C9A227",
    "Client 1 alone":              "#8a6d12",
    "FedAvg":                       "#7FBF94",
    r"FedProx ($\mu=0.01$)":        "#3D5A80",
    "Centralised":                  "#A37FBF",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


# --- Build the two-panel figure -------------------------------------
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(15, 5.2),
    gridspec_kw={"width_ratios": [1.0, 2.2], "wspace": 0.22},
)

# === Panel (a): Macro-F1 per configuration =========================
xA = np.arange(len(CONFIGS))
barsA = axA.bar(
    xA, macro_f1,
    color=[COLORS[c] for c in CONFIGS],
    edgecolor="white", linewidth=0.6,
)
for i, v in enumerate(macro_f1):
    axA.text(xA[i], v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)

# Reference line at centralised macro-F1.
cent_mf = macro_f1[-1]
axA.axhline(cent_mf, color=COLORS["Centralised"], linestyle="--",
            linewidth=1.0, alpha=0.55, zorder=0)

# Federation-lift arrows for both clients.
fedavg_mf = macro_f1[2]
fedprox_mf = macro_f1[3]
for client_idx, label, target_mf, color in [
    (0, "Client 0 lift\n(FedProx − alone)", fedprox_mf, COLORS["FedProx ($\\mu=0.01$)"]),
    (1, "Client 1 lift\n(FedProx − alone)", fedprox_mf, COLORS["FedProx ($\\mu=0.01$)"]),
]:
    delta = target_mf - macro_f1[client_idx]
    # Vertical arrow + text annotation between local-only bar and FedProx bar.
    axA.annotate(
        "", xy=(xA[client_idx], target_mf - 0.005),
        xytext=(xA[client_idx], macro_f1[client_idx] + 0.005),
        arrowprops=dict(arrowstyle="->", color=color, alpha=0.55, lw=1.5),
    )
    axA.text(xA[client_idx] - 0.32,
             (macro_f1[client_idx] + target_mf) / 2,
             f"+{delta:.3f}", color=color,
             ha="right", va="center", fontsize=8.5, fontweight="bold")

axA.set_xticks(xA)
axA.set_xticklabels(CONFIGS, rotation=25, ha="right", fontsize=9)
axA.set_ylabel("Test macro-F1 (seed 42)")
axA.set_title("(a) Macro-F1 across configurations",
              loc="left", fontweight="bold", fontsize=10)
axA.set_ylim(0, 0.65)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# === Panel (b): Per-class F1, grouped bars ========================
n_configs = len(CONFIGS)
n_classes = 7
xB = np.arange(n_classes)
bar_w = 0.16

# Yellow band behind the rare classes (held only by Client 1).
for c in RARE_CLASS_IDX:
    axB.axvspan(xB[c] - 0.5, xB[c] + 0.5,
                color="#C9A227", alpha=0.08, zorder=0)

for i, name in enumerate(CONFIGS):
    offset = (i - (n_configs - 1) / 2) * bar_w
    axB.bar(xB + offset, per_class[i],
            width=bar_w, label=name,
            color=COLORS[name], edgecolor="white", linewidth=0.4)

# Mark rare classes with stars.
tick_labels = [
    name + (r" $\bigstar$" if c in RARE_CLASS_IDX else "")
    for c, name in enumerate(CLASS_NAMES)
]
axB.set_xticks(xB)
axB.set_xticklabels(tick_labels, fontsize=9)
axB.set_ylabel("Per-class test F1 (seed 42)")
axB.set_title("(b) Per-class F1 — $\\bigstar$ = held only by Client 1",
              loc="left", fontweight="bold", fontsize=10)
axB.set_ylim(0, 1.0)
axB.set_xlim(-0.5, n_classes - 0.5)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2,
           handlelength=1.4, columnspacing=0.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = OUT / f"F_federation_value.{ext}"
    fig.savefig(out)
    print(f"Wrote: {out}")
plt.close(fig)
