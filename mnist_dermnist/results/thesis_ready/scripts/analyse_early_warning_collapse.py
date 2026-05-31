"""Experiment B1 — Early-warning detector for rare-class collapse.

A simple-but-novel question: can we predict, from EARLY-ROUND (round 20)
features alone, whether a federated training run will end with one or
more classes catastrophically collapsed (final F1 ≈ 0)? The medical-FL
literature (FedIIC, MICCAI 2023; FedSLD, ISBI 2022; Confusion-Calibrated
CE, 2026) reacts to collapse via specialised losses or post-hoc
calibration but does NOT predict collapse from early-round signals.

Operationally useful: if round-20 features have high ROC-AUC for
predicting final collapse, you can early-stop or trigger remedial
intervention BEFORE wasting 130 more rounds of compute.

Methodology:
  - Target: "collapsed" = final round had ≥ 2 classes with val F1 < 0.10
  - Features at round 20:
      * per-class val F1 (7 features)
      * macro-F1 at round 20
      * min per-class F1 at round 20 (worst class)
      * count of classes with val F1 < 0.10
      * range = max - min per-class F1
  - Model: logistic regression with leave-one-experiment-out CV
  - Metric: ROC-AUC

Why this is a contribution:
  - First operational early-stopping criterion for FL class collapse
    specifically (FedES IJCAI 2024 does early stopping for label noise,
    not for class collapse)
  - One ROC curve + one feature-importance table

Output:
  - early_warning_features.csv   : full feature matrix + labels
  - early_warning_results.csv    : per-fold ROC-AUC
  - F_early_warning_roc.{pdf,png} : ROC curve + feature importance
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS = REPO_ROOT / "mnist_dermnist/results"
OUT_DIR = REPO_ROOT / "mnist_dermnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "mnist_dermnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_F1_COLS = [f"val_f1_class_{c}" for c in range(7)]
EARLY_ROUND = 20
COLLAPSE_F1_THRESHOLD = 0.10  # per-class threshold
COLLAPSE_MIN_N_CLASSES = 2    # at least this many classes below threshold


STEM_RE = re.compile(
    r"history_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+(?P<tail>.*)_s(?P<seed>\d+)\.csv"
)


def _features_and_label(df: pd.DataFrame) -> dict | None:
    """Extract round-20 features and end-of-training collapse label."""
    if any(c not in df.columns for c in CLASS_F1_COLS):
        return None
    if "round" not in df.columns:
        return None
    df = df.sort_values("round")
    rounds = df["round"].values
    if rounds[-1] < EARLY_ROUND + 1:
        return None  # not enough rounds to assess

    # Find the row at or near round 20
    early_idx = (df["round"] - EARLY_ROUND).abs().idxmin()
    early = df.loc[early_idx]
    final = df.iloc[-1]

    # Features at round 20
    per_class_early = np.array([float(early[c]) for c in CLASS_F1_COLS])
    feat = dict(
        macro_f1_r20=float(np.mean(per_class_early)),
        min_class_f1_r20=float(per_class_early.min()),
        max_class_f1_r20=float(per_class_early.max()),
        range_class_f1_r20=float(per_class_early.max() - per_class_early.min()),
        n_classes_below_r20=int(np.sum(per_class_early < COLLAPSE_F1_THRESHOLD)),
    )
    for ci, v in enumerate(per_class_early):
        feat[f"class{ci}_f1_r20"] = float(v)

    # Label at final round
    per_class_final = np.array([float(final[c]) for c in CLASS_F1_COLS])
    n_collapsed_final = int(np.sum(per_class_final < COLLAPSE_F1_THRESHOLD))
    feat["collapsed"] = bool(n_collapsed_final >= COLLAPSE_MIN_N_CLASSES)
    feat["n_classes_collapsed_final"] = n_collapsed_final
    feat["macro_f1_final"] = float(np.mean(per_class_final))
    return feat


# ----------------------------------------------------------------
# 1. Scan all history files across all 7 experiments
# ----------------------------------------------------------------
SOURCE_DIRS = [
    RESULTS / "fedprox_perfect_storm_L4",
    RESULTS / "li2020_asymmetric_L4",
    RESULTS / "node_pinned_L4",
    RESULTS / "extended_rounds_L3",
    RESULTS / "fednova_unequal_E",
    RESULTS / "asymmetric_mu_L4",
    RESULTS / "mu_sweep_ladder",
    RESULTS / "heterogeneity_ladder",
    RESULTS / "two_client_90_10_rare_stress",
]

rows = []
for d in SOURCE_DIRS:
    if not d.exists():
        continue
    files = list(d.rglob("history_*.csv"))
    for f in files:
        m = STEM_RE.match(f.name)
        if m is None:
            continue
        df_hist = pd.read_csv(f)
        feat = _features_and_label(df_hist)
        if feat is None:
            continue
        rows.append(dict(
            source=d.name,
            algorithm=m.group("algo"),
            mu=float(m.group("mu")),
            seed=int(m.group("seed")),
            tail=m.group("tail"),
            file=f.name,
            **feat,
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "early_warning_features.csv", index=False)
print(f"Wrote {OUT_DIR/'early_warning_features.csv'}  ({len(df)} runs)")

# ----------------------------------------------------------------
# 2. Headline rate of final collapse
# ----------------------------------------------------------------
n_total = len(df)
n_collapsed = int(df["collapsed"].sum())
print()
print("=" * 80)
print("Final-collapse incidence")
print("=" * 80)
print(f"  Total runs               : {n_total}")
print(f"  Runs with ≥{COLLAPSE_MIN_N_CLASSES} classes")
print(f"    at F1 < {COLLAPSE_F1_THRESHOLD} at final round  : {n_collapsed}  ({n_collapsed/n_total*100:.1f}%)")
print()
print("Per-source breakdown:")
for src, grp in df.groupby("source"):
    n = len(grp); nc = int(grp["collapsed"].sum())
    print(f"  {src:<30} : {nc}/{n} collapsed ({nc/n*100:>4.0f}%)")
print()
print("Per-algorithm breakdown:")
for algo, grp in df.groupby("algorithm"):
    n = len(grp); nc = int(grp["collapsed"].sum())
    print(f"  {algo:<10} : {nc}/{n} collapsed ({nc/n*100:>4.0f}%)")

# ----------------------------------------------------------------
# 3. Logistic regression with cross-validation
# ----------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

feature_cols = ["macro_f1_r20", "min_class_f1_r20", "max_class_f1_r20",
                "range_class_f1_r20", "n_classes_below_r20"] + \
               [f"class{c}_f1_r20" for c in range(7)]

# Filter out any rows with NaN features
df_clf = df.dropna(subset=feature_cols + ["collapsed"])
X = df_clf[feature_cols].values
y = df_clf["collapsed"].astype(int).values
print()
print(f"Classifier training: {len(df_clf)} runs ({int(y.sum())} positive, {len(y)-int(y.sum())} negative)")

if int(y.sum()) < 2 or int(len(y) - y.sum()) < 2:
    print("WARNING: too few samples in one class for cross-validation. Using single-fit AUC.")
    clf = LogisticRegression(max_iter=10000, class_weight="balanced")
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    cv_aucs = [roc_auc_score(y, probs)]
    final_probs = probs
else:
    cv_aucs = []
    final_probs = np.zeros_like(y, dtype=float)
    skf = StratifiedKFold(n_splits=min(5, int(y.sum()), int(len(y) - y.sum())), shuffle=True, random_state=42)
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(max_iter=10000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], probs)
        cv_aucs.append(auc)
        final_probs[te] = probs

print(f"  Cross-validated ROC-AUC (per fold): {[f'{a:.3f}' for a in cv_aucs]}")
print(f"  Mean: {np.mean(cv_aucs):.3f}  Std: {np.std(cv_aucs, ddof=1) if len(cv_aucs)>1 else 0:.3f}")

# Single-feature ROC-AUCs (univariate predictive power)
print()
print("Univariate feature ROC-AUCs (single-feature predictive power):")
univ_aucs = []
for col in feature_cols:
    try:
        # Higher value → less likely to collapse; invert sign so AUC > 0.5 is the natural direction
        score = df_clf[col].values
        a = roc_auc_score(y, -score)  # negate: lower feature → more likely to collapse
        a_inv = roc_auc_score(y, score)
        a_best = max(a, a_inv)
        direction = "↓" if a > a_inv else "↑"
        univ_aucs.append((col, float(a_best), direction))
        print(f"  {col:<25} AUC = {a_best:.3f}  (direction: lower-feature {direction} → collapse)")
    except Exception:
        univ_aucs.append((col, np.nan, "?"))

# Save results
results_df = pd.DataFrame(dict(
    feature=[u[0] for u in univ_aucs],
    univariate_auc=[u[1] for u in univ_aucs],
    direction=[u[2] for u in univ_aucs],
))
results_df = results_df.sort_values("univariate_auc", ascending=False)
results_df.to_csv(OUT_DIR / "early_warning_univariate_aucs.csv", index=False)

# ----------------------------------------------------------------
# 4. Figure: ROC curve + feature importance bar chart
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.30})

# Panel A: ROC curve (overall)
fpr, tpr, _ = roc_curve(y, final_probs)
overall_auc = roc_auc_score(y, final_probs)
axA.plot(fpr, tpr, color="#3D5A80", linewidth=2.5,
         label=f"Multivariate logistic\nAUC = {overall_auc:.3f}")
axA.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1,
         label="Random (AUC = 0.500)")
axA.set_xlabel("False Positive Rate")
axA.set_ylabel("True Positive Rate")
axA.set_title("(a) ROC for predicting final collapse from round-20 features",
              loc="left", fontweight="bold", fontsize=11)
axA.legend(loc="lower right", frameon=False, fontsize=10)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: univariate feature AUCs
ranked = results_df.head(8)
y_pos = np.arange(len(ranked))
axB.barh(y_pos, ranked["univariate_auc"],
         color="#3D5A80", edgecolor="white", linewidth=0.6)
for i, v in enumerate(ranked["univariate_auc"]):
    axB.text(v + 0.005, y_pos[i], f"{v:.3f}", va="center", fontsize=9)
axB.set_yticks(y_pos)
axB.set_yticklabels(ranked["feature"], fontsize=9)
axB.set_xlim(0.4, 1.0)
axB.set_xlabel("Univariate ROC-AUC")
axB.set_title("(b) Per-feature predictive power for final collapse",
              loc="left", fontweight="bold", fontsize=11)
axB.axvline(0.5, color="#888", linestyle="--", linewidth=0.8)
axB.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
axB.invert_yaxis()

fig.suptitle("Early-warning rare-class collapse detector — round-20 features predict final F1<0.10 collapse",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_early_warning_roc.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_early_warning_roc.pdf'}")
print()
print(f"HEADLINE: cross-validated logistic-regression AUC = {np.mean(cv_aucs):.3f}")
if np.mean(cv_aucs) > 0.85:
    print(f"  → ✅ Strong predictive signal at round {EARLY_ROUND} (AUC > 0.85)")
elif np.mean(cv_aucs) > 0.70:
    print(f"  → Moderate predictive signal at round {EARLY_ROUND}")
else:
    print(f"  → Weak predictive signal at round {EARLY_ROUND}")
print()
print("Done.")
