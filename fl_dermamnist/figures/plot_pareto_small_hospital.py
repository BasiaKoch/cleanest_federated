"""Pareto scatter - Client 0 specialty F1 vs Client 1 specialty F1.

Compresses the entire small-hospital federation story into a single
2D plot. Each training regime becomes one point; the axes show mean
test F1 over each client's specialty classes. Federated points sit on
the Pareto trade-off frontier between the two local-only corners;
centralised dominates everything.

This is the visualisation the federated-medical-imaging literature
should be using (Sheller 2018, Roth 2020, Pati 2022 show per-client
bars but not the cross-client trade-off). Distinguishes "FedProx is
strictly better than FedAvg on the small client's classes at
near-equal cost on the large client's classes" --- a fairness
claim that the global macro-F1 number cannot show.

Output: F_pareto_small_hospital.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
OUT = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Specialty class indices.
CLASSES_C0 = (0, 1, 2, 5)
CLASSES_C1 = (3, 4, 6)

# Test JSONs to read.
PATHS = {
    "Client 0 alone":              "fl_dermamnist/results/small_hospital_local_only/test_at_best_local_only_c0_E3000_s42.json",
    "Client 1 alone":              "fl_dermamnist/results/small_hospital_local_only/test_at_best_local_only_c1_E3000_s42.json",
    "FedAvg":                      "fl_dermamnist/results/two_client_90_10_rare_stress/test_at_best_fedavg_mu0.0_E20_s42.json",
    r"FedProx ($\mu = 0.01$)":     "fl_dermamnist/results/two_client_90_10_rare_stress/test_at_best_fedprox_mu0.01_E20_s42.json",
    "Centralised":                 "fl_dermamnist/results/centralised/centralised_seed42.json",
}

COLORS = {
    "Client 0 alone":              "#C9A227",
    "Client 1 alone":              "#8a6d12",
    "FedAvg":                       "#7FBF94",
    r"FedProx ($\mu = 0.01$)":      "#3D5A80",
    "Centralised":                  "#A37FBF",
}

# Annotation offsets (x_offset, y_offset, horizontal_alignment).
ANNOTATION_OFFSETS = {
    "Client 0 alone":              (-0.020,  0.025, "right"),
    "Client 1 alone":              ( 0.020,  0.000, "left"),
    "FedAvg":                       (-0.020,  0.000, "right"),
    r"FedProx ($\mu = 0.01$)":      ( 0.020,  0.000, "left"),
    "Centralised":                  ( 0.020,  0.000, "left"),
}


def specialty_means(per_class_f1):
    pc = np.asarray(per_class_f1)
    return float(pc[list(CLASSES_C0)].mean()), float(pc[list(CLASSES_C1)].mean())


points = {}
for name, path in PATHS.items():
    full = REPO_ROOT / path
    d = json.load(open(full))
    x, y = specialty_means(d["per_class_f1"])
    points[name] = {"x": x, "y": y, "macro_f1": d["macro_f1"]}
    print(f"  {name:30s}  C0={x:.3f}  C1={y:.3f}  macro_f1={d['macro_f1']:.3f}")


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# --- Background shading: dominated region of each local-only point ---
c0_alone = points["Client 0 alone"]
c1_alone = points["Client 1 alone"]

# Lightly shade the region where Client 0 alone is Pareto-dominant
# (x <= c0_alone.x AND y <= 0): the "Client 0 already does this alone" zone.
ax.fill_between(
    [0, c0_alone["x"]], [0, 0], [c0_alone["y"], c0_alone["y"]],
    color=COLORS["Client 0 alone"], alpha=0.07, zorder=0,
)
# And the analogous Client-1-dominant zone.
ax.fill_between(
    [0, c1_alone["x"]], [0, 0], [c1_alone["y"], c1_alone["y"]],
    color=COLORS["Client 1 alone"], alpha=0.07, zorder=0,
)

# Diagonal "equal-specialty" reference line.
ax.plot([0, 0.75], [0, 0.75], color="#888", linestyle=":",
        linewidth=0.8, alpha=0.7, zorder=0,
        label="equal-specialty line (y = x)")


# --- Connecting arrows: each client's federation trajectory ---
for src_key, dst_key, color in [
    ("Client 0 alone", r"FedProx ($\mu = 0.01$)", COLORS["FedProx ($\\mu = 0.01$)"]),
    ("Client 1 alone", r"FedProx ($\mu = 0.01$)", COLORS["FedProx ($\\mu = 0.01$)"]),
]:
    src = points[src_key]; dst = points[dst_key]
    ax.annotate("",
                xy=(dst["x"], dst["y"]),
                xytext=(src["x"], src["y"]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.35,
                                lw=1.2, linestyle="--"),
                zorder=1)


# --- Plot each regime as a point ---
for name, p in points.items():
    ax.scatter(p["x"], p["y"], color=COLORS[name], s=180, zorder=5,
               edgecolor="white", linewidth=1.8)
    dx, dy, ha = ANNOTATION_OFFSETS[name]
    # Two-line label: name on top, macro-F1 underneath
    ax.annotate(f"{name}\n(macro-F1 = {p['macro_f1']:.3f})",
                (p["x"] + dx, p["y"] + dy),
                ha=ha, va="center", fontsize=9.5, fontweight="bold",
                color=COLORS[name])


# --- Quadrant labels (faint, in corners) ---
ax.text(0.55, 0.55,
        "Pareto-optimal\n(both clients\nwell-served)",
        ha="center", va="top", fontsize=9, alpha=0.35, style="italic",
        transform=ax.transData)

ax.set_xlabel("Mean test F1 over Client 0's classes\n"
              "(actinic, basal, benign keratosis, melanocytic nevi)",
              fontsize=10)
ax.set_ylabel("Mean test F1 over Client 1's classes\n"
              "(dermatofibroma, melanoma, vascular lesions)",
              fontsize=10)
ax.set_xlim(-0.02, 0.75)
ax.set_ylim(-0.02, 0.6)
ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title(
    "The federation Pareto frontier on the 2-client 90/10 partition (seed 42)",
    loc="left", fontweight="bold", fontsize=11.5, pad=10,
)
# Tiny legend only for the diagonal reference line + arrows.
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="#888", linestyle=":", linewidth=0.8,
           label="equal-specialty line (y = x)"),
    Line2D([0], [0], color=COLORS["FedProx ($\\mu = 0.01$)"],
           linestyle="--", linewidth=1.2, alpha=0.5,
           label="federation lift (local → FedProx)"),
]
ax.legend(handles=legend_handles, loc="upper right",
          frameon=False, fontsize=9)


fig.tight_layout()
for ext in ("pdf", "png"):
    out = OUT / f"F_pareto_small_hospital.{ext}"
    fig.savefig(out)
    print(f"Wrote: {out}")
plt.close(fig)
