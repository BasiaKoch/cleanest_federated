"""
Generate F_perfect_storm_headline.pdf -- the focused headline visual of the
largest FedAvg-vs-FedProx gap measured in the thesis (perfect-storm L4).

Three-panel row-normalised confusion matrices (test set, single seed=42 for
all three panels so the comparison is matched):

  Panel 1: FedAvg+drop          (macro-F1 = 0.087 +/- 0.049 over 3 seeds)
  Panel 2: FedProx + gamma-inex, mu=1.0   (0.365 +/- 0.023)
  Panel 3: FedProx + gamma-inex, mu=0.01  (0.491 +/- 0.003)

Headline message: FedAvg+drop collapses to predicting a single class for ~99%
of test images -- the model is broken, not just under-trained. The dominant
class is seed-dependent (seeds 42 and 123 collapse to nv; seed 456 to bkl),
but every seed exhibits the same one-class collapse. FedProx panels show
proper diagonal structure; the gap lives largely in the rare classes
(3 = dermatofibroma, 4 = melanoma, 6 = vascular).
"""
from pathlib import Path
import json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent.parent / "figures"
RES = Path(__file__).resolve().parent.parent.parent / "fedprox_perfect_storm_L4"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RARE_IDX = (3, 4, 6)  # dermatofibroma, melanoma, vascular
NUM_CLASSES = 7

# Single representative seed for all three panels (matched comparison).
# Cross-seed macro-F1 mean/SD is still computed from all 3 seeds for the
# panel titles.
SEED = 42

# --- per-condition prediction file globs ---
CONDITIONS = [
    ("FedAvg + drop",
     f"test_predictions_fedavg_mu0.0_E20_sh-random_stragglers_drop_s{SEED}.npz",
     "test_at_best_fedavg_mu0.0_E20_sh-random_stragglers_drop_s*.json"),
    (r"FedProx + $\gamma$-inexact, $\mu = 1.0$",
     f"test_predictions_fedprox_mu1.0_E20_sh-random_stragglers_s{SEED}.npz",
     "test_at_best_fedprox_mu1.0_E20_sh-random_stragglers_s*.json"),
    (r"FedProx + $\gamma$-inexact, $\mu = 0.01$",
     f"test_predictions_fedprox_mu0.01_E20_sh-random_stragglers_s{SEED}.npz",
     "test_at_best_fedprox_mu0.01_E20_sh-random_stragglers_s*.json"),
]

def load_panel(pred_file: str, json_glob: str):
    """Load predictions for one specific seed (for the CM) and compute the
    cross-seed macro-F1 mean +/- sample SD from all matched seeds."""
    npz = np.load(RES / pred_file)
    preds = npz["predictions"]
    targs = npz["targets"]
    json_files = sorted((RES / "").glob(json_glob))
    macros = [json.load(open(f))["macro_f1"] for f in json_files]
    macro_mean = float(np.mean(macros))
    macro_sd   = float(np.std(macros, ddof=1))
    return preds, targs, macro_mean, macro_sd, len(macros)

def row_normalised_cm(preds: np.ndarray, targs: np.ndarray) -> np.ndarray:
    """Return a row-normalised confusion matrix (rows = true, cols = pred).
    Diagonal entries are per-class recall."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for t, p in zip(targs, preds):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div0; will leave row at 0
    return cm / row_sums

# --- load everything ---
panels = []
for label, pred_file, jglob in CONDITIONS:
    preds, targs, m, sd, n = load_panel(pred_file, jglob)
    cm = row_normalised_cm(preds, targs)
    panels.append({"label": label, "cm": cm, "m": m, "sd": sd, "n": n})

# --- plot ---
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8),
                          gridspec_kw=dict(wspace=0.45))

# Mid-saturation blue, white at zero
cmap = LinearSegmentedColormap.from_list(
    "cm_blues", ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])

# Tick labels: bold/asterisk rare classes
def lab(i):
    name = CLASS_NAMES[i]
    if i in RARE_IDX:
        return rf"$\mathbf{{{name}}}$*"
    return name
tick_labels = [lab(i) for i in range(NUM_CLASSES)]

for ax, panel in zip(axes, panels):
    cm = panel["cm"]
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(tick_labels, fontsize=8.5, rotation=35, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=8.5)
    ax.set_xlabel("predicted class", fontsize=9)
    ax.set_ylabel("true class", fontsize=9)
    ax.set_title(f"{panel['label']}\n"
                 rf"macro-F1 $= {panel['m']:.3f} \pm {panel['sd']:.3f}$",
                 fontsize=10, pad=8)

    # Annotate each cell with the value (small text)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            if v >= 0.005:
                txt = f"{v:.2f}".lstrip("0") if v < 1 else "1.0"
                colour = "white" if v > 0.55 else "#333"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=colour)

    # Box around the rare-class rows on the y-axis
    for r in RARE_IDX:
        ax.add_patch(plt.Rectangle((-0.5, r - 0.5), NUM_CLASSES, 1,
                                    fill=False, edgecolor="#C44E52",
                                    lw=1.2, zorder=5))

# Single shared colourbar to the right
cbar_ax = fig.add_axes([0.93, 0.18, 0.011, 0.66])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("row-normalised count\n(diagonal = per-class recall)",
               fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Figure title
fig.suptitle(
    "Perfect-storm L4 confusion matrices: the $+0.404$ gap is a model "
    "collapse, not a small accuracy loss",
    fontsize=11.5, y=1.02)

# Caption-style footer
fig.text(0.5, -0.05,
         r"Row-normalised confusion matrices for seed $42$; macro-F1 in each "
         r"title is the mean $\pm$ sample SD over the three matched seeds "
         r"$\{42, 123, 456\}$. Rare classes (df, mel, vasc) outlined in red. "
         r"Under FedAvg+drop every row routes its mass into a single column "
         r"(nv for seeds $42, 123$; bkl for seed $456$) -- the model predicts "
         r"one class for $\approx\!99\%$ of test images regardless of input. "
         r"FedProx+$\gamma$-inexact restores diagonal structure; the "
         r"$\mu = 0.01$ panel sharpens it further.",
         ha="center", va="top", fontsize=9, style="italic", wrap=True)

out_pdf = OUT / "F_perfect_storm_headline.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Wrote: {out_pdf}")
plt.close(fig)
