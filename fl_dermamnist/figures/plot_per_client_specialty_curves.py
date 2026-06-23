"""Per-client specialty trajectories under FedAvg vs FedProx.

Two panels --- one per client --- showing how the global federated
model's validation F1 evolves on each client's specialty classes over
the 150 communication rounds, separately for FedAvg and FedProx. Local-
only baselines for each client (on their own classes) are drawn as
dashed horizontal reference lines, so the reader can see how much
federation adds beyond what each client could achieve alone.

Setup (2-client 90/10 stress partition, seed 42):
  Client 0 specialty: classes 0, 1, 2, 5  (actinic, basal, benign keratosis, nevi)
  Client 1 specialty: classes 3, 4, 6     (dermatofibroma, melanoma, vascular)

Output: F_per_client_specialty_curves.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
FED_DIR   = REPO_ROOT / "fl_dermamnist/results/two_client_90_10_rare_stress"
LOCAL_DIR = REPO_ROOT / "fl_dermamnist/results/small_hospital_local_only"
OUT       = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Specialty class indices per client.
CLASSES_C0 = (0, 1, 2, 5)   # common: actinic, basal, benign keratosis, nevi
CLASSES_C1 = (3, 4, 6)      # rare:   dermato, melanoma, vascular

COL_FEDAVG  = "#7FBF94"     # mint green
COL_FEDPROX = "#3D5A80"     # dark slate blue
COL_LOCAL   = "#C9A227"     # gold (matches the local-only figure)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_fed_history(algo: str, mu: float) -> pd.DataFrame:
    f = FED_DIR / f"history_{algo}_mu{mu}_E20_s42.csv"
    return pd.read_csv(f)


def client_specialty_mean(df: pd.DataFrame, class_ids) -> np.ndarray:
    """Per-round mean of val_f1_class_<c> across the given class ids."""
    cols = [f"val_f1_class_{c}" for c in class_ids]
    return df[cols].mean(axis=1).to_numpy()


def local_only_specialty_mean(client_id: int, class_ids) -> float:
    """Best-val checkpoint's per-class F1 mean for the given classes,
    on the local-only run.

    Reads the history JSON to find the checkpoint with the highest
    val_macro_f1 (the selected epoch) and returns the mean of
    val_f1_class_<c> at that epoch over the specialty classes.
    """
    hist = json.load(open(
        LOCAL_DIR / f"history_local_only_c{client_id}_E3000_s42.json"))
    # Pick the entry with the highest val macro_f1 (== best-val checkpoint).
    best = max(hist, key=lambda r: r["macro_f1"])
    return float(np.mean([best["per_class_f1"][c] for c in class_ids]))


# --- Load all four federated trajectories ----------------------------
fa = load_fed_history("fedavg",  "0.0")
fp = load_fed_history("fedprox", "0.01")

c0_fa = client_specialty_mean(fa, CLASSES_C0)
c0_fp = client_specialty_mean(fp, CLASSES_C0)
c1_fa = client_specialty_mean(fa, CLASSES_C1)
c1_fp = client_specialty_mean(fp, CLASSES_C1)

c0_local = local_only_specialty_mean(0, CLASSES_C0)
c1_local = local_only_specialty_mean(1, CLASSES_C1)

rounds = fa["round"].to_numpy()


# --- Figure ----------------------------------------------------------
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(15, 5),
    gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.20},
)

def smooth(y: np.ndarray, window: int = 9) -> np.ndarray:
    """Centred moving average; reflects edges to avoid endpoint shrinkage."""
    pad = window // 2
    y_padded = np.concatenate([y[:pad][::-1], y, y[-pad:][::-1]])
    kernel = np.ones(window) / window
    return np.convolve(y_padded, kernel, mode="valid")


# ===== Panel A: Client 0 specialty (common classes) ==========
axA.plot(rounds, c0_fa, color=COL_FEDAVG,  linewidth=0.7, alpha=0.30)
axA.plot(rounds, c0_fp, color=COL_FEDPROX, linewidth=0.7, alpha=0.30)
axA.plot(rounds, smooth(c0_fa), color=COL_FEDAVG,  linewidth=2.0, label="FedAvg")
axA.plot(rounds, smooth(c0_fp), color=COL_FEDPROX, linewidth=2.0,
         label=r"FedProx ($\mu = 0.01$)")
axA.axhline(c0_local, color=COL_LOCAL, linestyle="--", linewidth=1.4,
            alpha=0.85,
            label=f"Client 0 alone (local-only) = {c0_local:.3f}")

axA.set_xlabel("Communication round")
axA.set_ylabel("Mean validation F1 over Client 0's classes")
axA.set_xlim(0, rounds.max())
axA.set_ylim(0, 0.85)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.legend(loc="lower right", frameon=False, fontsize=9.5)
axA.set_title(
    "(a) Client 0's specialty — actinic, basal, benign keratosis, nevi",
    loc="left", fontweight="bold", fontsize=10.5,
)

# ===== Panel B: Client 1 specialty (rare classes) ============
axB.plot(rounds, c1_fa, color=COL_FEDAVG,  linewidth=0.7, alpha=0.30)
axB.plot(rounds, c1_fp, color=COL_FEDPROX, linewidth=0.7, alpha=0.30)
axB.plot(rounds, smooth(c1_fa), color=COL_FEDAVG,  linewidth=2.0, label="FedAvg")
axB.plot(rounds, smooth(c1_fp), color=COL_FEDPROX, linewidth=2.0,
         label=r"FedProx ($\mu = 0.01$)")
axB.axhline(c1_local, color=COL_LOCAL, linestyle="--", linewidth=1.4,
            alpha=0.85,
            label=f"Client 1 alone (local-only) = {c1_local:.3f}")

axB.set_xlabel("Communication round")
axB.set_ylabel("Mean validation F1 over Client 1's classes")
axB.set_xlim(0, rounds.max())
axB.set_ylim(0, 0.85)
axB.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.legend(loc="lower right", frameon=False, fontsize=9.5)
axB.set_title(
    "(b) Client 1's specialty (rare) — dermatofibroma, melanoma, vascular",
    loc="left", fontweight="bold", fontsize=10.5,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = OUT / f"F_per_client_specialty_curves.{ext}"
    fig.savefig(out)
    print(f"Wrote: {out}")

# Useful diagnostic prints. Report the BEST-VAL-CHECKPOINT round so the
# numbers line up with the test-set results reported elsewhere (the
# best-val protocol freezes the model at the round with highest val
# macro-F1; the federated test numbers in tab:small-hospital-fed-value
# correspond to that frozen model).
fa_best = int(fa.loc[fa["val_macro_f1"].idxmax(), "round"])
fp_best = int(fp.loc[fp["val_macro_f1"].idxmax(), "round"])
print()
print("Best-val-checkpoint specialty means (validation):")
print(f"  Client 0: FedAvg@round{fa_best} = {c0_fa[fa_best-1]:.3f},  "
      f"FedProx@round{fp_best} = {c0_fp[fp_best-1]:.3f},  "
      f"local-only = {c0_local:.3f}")
print(f"  Client 1: FedAvg@round{fa_best} = {c1_fa[fa_best-1]:.3f},  "
      f"FedProx@round{fp_best} = {c1_fp[fp_best-1]:.3f},  "
      f"local-only = {c1_local:.3f}")
plt.close(fig)
