"""Validation trajectories for the local-only small-hospital baselines.

Replaces the bar-chart figure with the more mechanistically informative
view: validation macro-F1 across the 3,000 epochs of local-only
training, for each client separately. Annotates the best-validation
checkpoint (the point at which test was actually evaluated) so the
reader sees the early-overfitting pattern that justifies the
federation-is-necessary framing.

Two panels:
    (a) Validation macro-F1 vs training epoch, for Client 0 and
        Client 1, with the best-val checkpoint marked. The federated
        FedAvg / FedProx macro-F1 values are drawn as horizontal
        reference lines.
    (b) Validation cross-entropy loss vs training epoch (zoomed to
        the early-training region) showing the overfitting pattern:
        train loss collapses to ~0 while val loss diverges.

Output: F_federation_value_curves.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
LOCAL_DIR = REPO_ROOT / "fl_dermamnist/results/small_hospital_local_only"
FED_DIR = REPO_ROOT / "fl_dermamnist/results/two_client_90_10_rare_stress"
OUT = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT.mkdir(parents=True, exist_ok=True)

COL_C0       = "#C9A227"     # gold (matches the bar-chart figure)
COL_C1       = "#8a6d12"     # darker gold
COL_FEDAVG   = "#7FBF94"
COL_FEDPROX  = "#3D5A80"
COL_CENT     = "#A37FBF"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_history(client_id: int) -> dict:
    h = json.load(open(LOCAL_DIR / f"history_local_only_c{client_id}_E3000_s42.json"))
    return {
        "epochs":    np.array([r["epoch"] for r in h]),
        "val_mf1":   np.array([r["macro_f1"] for r in h]),
        "val_loss":  np.array([r["loss"] for r in h]),
        "train_loss": np.array([r["train_loss"] for r in h]),
    }


def load_test_at_best(client_id: int) -> dict:
    return json.load(open(LOCAL_DIR / f"test_at_best_local_only_c{client_id}_E3000_s42.json"))


c0 = load_history(0)
c1 = load_history(1)
t0 = load_test_at_best(0)
t1 = load_test_at_best(1)

# Federated reference (best-val macro-F1 used as a horizontal line).
fa_test  = json.load(open(FED_DIR / "test_at_best_fedavg_mu0.0_E20_s42.json"))
fp_test  = json.load(open(FED_DIR / "test_at_best_fedprox_mu0.01_E20_s42.json"))


# --- Figure ------------------------------------------------------------
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(15.5, 5.6),
    gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.22},
)

# ===== Panel (a): validation macro-F1 vs epoch =====
for hx, label, color, t in [
    (c0, "Client 0 alone (n=6,049)", COL_C0, t0),
    (c1, "Client 1 alone (n=958)",   COL_C1, t1),
]:
    axA.plot(hx["epochs"], hx["val_mf1"], color=color, linewidth=1.7,
             label=label)
    best_ep = t["selected_epoch"]
    best_v  = t["best_val_macro_f1"]
    axA.scatter([best_ep], [best_v], color=color, s=85, zorder=5,
                edgecolor="white", linewidth=1.4)

# Annotation arrows: place them on opposite sides so they don't clash.
axA.annotate(
    f"Client 0 best-val\nat epoch {t0['selected_epoch']}\nval F1 = {t0['best_val_macro_f1']:.3f}",
    xy=(t0['selected_epoch'], t0['best_val_macro_f1']),
    xytext=(1100, 0.31),
    fontsize=9, color=COL_C0, ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color=COL_C0, alpha=0.55, lw=1.0),
)
axA.annotate(
    f"Client 1 best-val\nat epoch {t1['selected_epoch']}\nval F1 = {t1['best_val_macro_f1']:.3f}",
    xy=(t1['selected_epoch'], t1['best_val_macro_f1']),
    xytext=(550, 0.04),
    fontsize=9, color=COL_C1, ha="left", va="bottom",
    arrowprops=dict(arrowstyle="->", color=COL_C1, alpha=0.55, lw=1.0),
)

# Reference lines for the federated test macro-F1, with labels at the
# right edge so they don't fight the legend.
for y, color, name in [
    (fa_test["macro_f1"],  COL_FEDAVG,  "FedAvg test"),
    (fp_test["macro_f1"],  COL_FEDPROX, "FedProx test"),
]:
    axA.axhline(y, color=color, linestyle="--", linewidth=1.1, alpha=0.85)
    axA.text(2980, y + 0.005, f"{name} = {y:.3f}",
             color=color, ha="right", va="bottom", fontsize=8.5,
             fontweight="bold")

axA.set_xlabel("Local-training epoch (matched compute budget = 3,000)")
axA.set_ylabel("Validation macro-F1")
axA.set_xlim(0, 3000)
axA.set_ylim(0, 0.6)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.legend(loc="lower right", frameon=False, fontsize=9.5)
axA.set_title("(a) Validation macro-F1 — both clients overfit early",
              loc="left", fontweight="bold", fontsize=10.5)


# ===== Panel (b): validation balanced accuracy vs epoch =====
# Balanced accuracy is the average of per-class recall and is bounded
# [0, 1], so it shows the same early-overfit pattern without log-scale
# crowding. Useful because for Client 1, balanced accuracy ≠ macro-F1
# (Client 1's recall stays decent on its 3 trained classes while F1
# is destroyed by zero precision on the 4 unseen ones).
for hx, label, color, t in [
    (c0, "Client 0 alone", COL_C0, t0),
    (c1, "Client 1 alone", COL_C1, t1),
]:
    bal = np.array([r['balanced_accuracy'] for r in
                    json.load(open(LOCAL_DIR /
                              f"history_local_only_c{0 if 'Client 0' in label else 1}_E3000_s42.json"))])
    axB.plot(hx["epochs"], bal, color=color, linewidth=1.7,
             label=f"{label} (val bal-acc)")
    best_ep = t["selected_epoch"]
    axB.axvline(best_ep, color=color, linestyle=":", linewidth=1.1, alpha=0.6)

axB.set_xlabel("Local-training epoch")
axB.set_ylabel("Validation balanced accuracy")
axB.set_xlim(0, 3000)
axB.set_ylim(0, 0.6)
axB.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.legend(loc="lower right", frameon=False, fontsize=9.5)
axB.set_title("(b) Validation balanced accuracy",
              loc="left", fontweight="bold", fontsize=10.5)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = OUT / f"F_federation_value_curves.{ext}"
    fig.savefig(out)
    print(f"Wrote: {out}")
plt.close(fig)
