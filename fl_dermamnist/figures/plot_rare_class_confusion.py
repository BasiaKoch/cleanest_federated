"""Where do Client 1's rare classes get classified, under FedAvg vs FedProx?

For each of the three rare classes (dermatofibroma, melanoma, vascular)
held only by Client 1, the model's predictions on test inputs of that
true class are decomposed into the 7 possible predicted classes. Shows
the clinically most interesting failure mode: melanoma being
misclassified as the dominant melanocytic nevi class under FedAvg, and
how FedProx reduces that error.

Reads the test_predictions_*.npz files saved by run_one_flower.py and
reconstructs the relevant confusion-matrix rows.

Output: F_rare_class_confusion.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
FED_DIR   = REPO_ROOT / "fl_dermamnist/results/two_client_90_10_rare_stress"
OUT       = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "actinic\nkeratoses",
    "basal cell\ncarcinoma",
    "benign\nkeratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic\nnevi",
    "vascular\nlesions",
]
RARE_CLASS_IDX = (3, 4, 6)
RARE_CLASS_LABELS = {3: "Dermatofibroma", 4: "Melanoma", 6: "Vascular lesions"}

# Categorical palette - one colour per *predicted* class, with the
# correct (rare) classes at higher saturation so a "correct prediction"
# segment stands out within each stack.
PRED_COLORS = [
    "#88B299",  # actinic
    "#A8C09F",  # basal
    "#C9D6C0",  # benign keratosis (common-class palette)
    "#E07B39",  # dermatofibroma  (rare, distinct)
    "#C03A2B",  # melanoma         (rare, distinct, clinically critical)
    "#7E9CD0",  # nevi             (the dominant common class)
    "#5E3A85",  # vascular         (rare)
]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_predictions(algo: str, mu: float):
    f = FED_DIR / f"test_predictions_{algo}_mu{mu}_E20_s42.npz"
    d = np.load(f)
    return d["targets"].astype(int), d["predictions"].astype(int)


# Build (rare_class, algo) -> normalised distribution of predictions.
algos = [
    ("fedavg",  "0.0",  "FedAvg"),
    ("fedprox", "0.01", r"FedProx ($\mu = 0.01$)"),
]
rows = []   # each row: (label, distribution: shape (7,))
for true_class in RARE_CLASS_IDX:
    for algo_key, mu, algo_label in algos:
        y_true, y_pred = load_predictions(algo_key, mu)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
        # Distribution of predictions among test samples with true == true_class.
        row = cm[true_class].astype(float)
        if row.sum() > 0:
            row /= row.sum()
        rows.append({
            "true_class": true_class,
            "true_class_label": RARE_CLASS_LABELS[true_class],
            "algorithm": algo_label,
            "dist": row,
            "n_samples": int(cm[true_class].sum()),
            "n_correct": int(cm[true_class, true_class]),
        })


# --- Figure: horizontal stacked bars ---
n_rows = len(rows)
fig, ax = plt.subplots(1, 1, figsize=(13, 5.6))

# Group the 6 bars in 3 pairs (one pair per rare class), with gaps.
bar_y = np.empty(n_rows, dtype=float)
y = 0.0
for i in range(n_rows):
    if i > 0 and rows[i]["true_class"] != rows[i - 1]["true_class"]:
        y += 0.7  # extra gap between rare-class groups
    bar_y[i] = y
    y += 1.0
bar_y = bar_y[::-1]   # flip so first row appears at top

bar_h = 0.65
left_acc = np.zeros(n_rows)
for c in range(7):
    seg = np.array([r["dist"][c] for r in rows])
    label = f"predicted: {CLASS_NAMES[c].replace(chr(10), ' ')}"
    if c in RARE_CLASS_IDX:
        label += " (rare)"
    ax.barh(bar_y, seg, height=bar_h, left=left_acc,
            color=PRED_COLORS[c], edgecolor="white", linewidth=0.6,
            label=label)
    # Annotate segments that are >= 5% with the percentage, inside the bar.
    for i, v in enumerate(seg):
        if v >= 0.05:
            ax.text(left_acc[i] + v / 2, bar_y[i],
                    f"{100*v:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if c in (3, 4, 5, 6) else "#333",
                    fontweight="bold")
    left_acc += seg

# Y-axis labels: "{algorithm}\n(n_correct/n_total)".
y_labels = [
    f"{r['algorithm']}\n({r['n_correct']}/{r['n_samples']} correct)"
    for r in rows
]
ax.set_yticks(bar_y)
ax.set_yticklabels(y_labels, fontsize=9.5)

# Add the rare-class group titles on the right margin.
for true_class in RARE_CLASS_IDX:
    idx = [i for i, r in enumerate(rows) if r["true_class"] == true_class]
    y_mid = bar_y[idx].mean()
    ax.text(1.02, y_mid, f"True: {RARE_CLASS_LABELS[true_class]}",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color="#444", transform=ax.get_yaxis_transform())

ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of test samples predicted as ... (per true class)")
ax.set_title(
    "Where Client 1's rare-class test inputs get classified, under FedAvg vs FedProx",
    loc="left", fontweight="bold", fontsize=11.5, pad=10,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=4, frameon=False, fontsize=8.5)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = OUT / f"F_rare_class_confusion.{ext}"
    fig.savefig(out)
    print(f"Wrote: {out}")

# Useful summary prints.
print()
print("Diagonal (correct) recall on each rare class:")
for r in rows:
    diag = r["dist"][r["true_class"]]
    misas_nevi = r["dist"][5]   # mel-nevi is class 5 (most common confusion)
    print(f"  {r['true_class_label']:<20s} under {r['algorithm']:<22s}"
          f"  recall={diag:.3f}   misas_nevi={misas_nevi:.3f}"
          f"   ({r['n_correct']}/{r['n_samples']} correct)")
plt.close(fig)
