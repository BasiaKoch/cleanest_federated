"""Generate the three thesis figures 10–12 with a consistent style.

Outputs (all saved to ../figures/):
  10_partition_heatmap.png  - 7x7 partition heatmap with marginal bars
  11_per_client_curves.png  - 7-panel per-client val macro-F1 over rounds
  12_federation_tax.png     - per-class grouped bars (FedAvg/FedProx/Centralised)

Style is shared: serif font, blue=FedAvg, orange=FedProx, gray=Centralised.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Layout / style
# -----------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[3]
HEADLINE   = ROOT / "results" / "headline"
CENTRAL    = ROOT / "results" / "centralised"
FIG_DIR    = ROOT / "results" / "thesis_ready" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_FEDAVG  = "#1f77b4"
C_FEDPROX = "#ff7f0e"
C_CENTRAL = "#7f7f7f"

plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 150,
    "savefig.bbox"     : "tight",
})

# -----------------------------------------------------------------------------
# Partition spec — mirror of BALANCED_PAIRED_7_CLIENTS_SPEC in data/partition.py
# Class IDs:  0=actinic, 1=basal, 2=benign_kerat, 3=dermato, 4=melanoma,
#             5=nevi (melanocytic), 6=vascular
# -----------------------------------------------------------------------------
PARTITION_COUNTS: list[dict[int, int]] = [
    {0: 114, 1: 180, 5: 670},       # C0
    {0: 114, 1: 179, 5: 670},       # C1
    {2: 385, 3:  40, 5: 670},       # C2
    {2: 384, 3:  40, 5: 670},       # C3
    {4: 390, 6:  50, 5: 670},       # C4
    {4: 389, 6:  49, 5: 670},       # C5
    {5: 673},                        # C6 — nevi-only generalist
]

# Display order requested by the user (decreasing global prevalence):
#   nevi, melanoma, benign_kerat, basal, actinic, vascular, dermato
DISPLAY_ORDER_IDS = [5, 4, 2, 1, 0, 6, 3]
DISPLAY_ORDER_LABELS = ["nevi", "melanoma", "benign_kerat", "basal",
                        "actinic", "vascular", "dermato"]

CLIENT_LABELS = [f"C{i}" for i in range(7)]
SEEDS = [42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828]


# =============================================================================
# Figure A — Partition heatmap (figures/10_partition_heatmap.png)
# =============================================================================
def figure_a_partition_heatmap() -> None:
    grid = np.zeros((7, 7), dtype=int)
    for ci, comp in enumerate(PARTITION_COUNTS):
        for pos, cls_id in enumerate(DISPLAY_ORDER_IDS):
            grid[ci, pos] = comp.get(cls_id, 0)

    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)

    fig = plt.figure(figsize=(8.4, 6.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[6.0, 1.0],
        height_ratios=[1.0, 6.0],
        hspace=0.04, wspace=0.04,
    )
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    im = ax_main.imshow(grid, cmap="viridis", aspect="auto")
    ax_main.set_xticks(range(7))
    ax_main.set_xticklabels(DISPLAY_ORDER_LABELS, rotation=30, ha="right")
    ax_main.set_yticks(range(7))
    ax_main.set_yticklabels(CLIENT_LABELS)
    ax_main.set_xlabel("Class (decreasing global prevalence)")
    ax_main.set_ylabel("Client")

    vmax = grid.max()
    for ci in range(7):
        for cj in range(7):
            v = grid[ci, cj]
            if v == 0:
                continue
            colour = "white" if v < vmax * 0.55 else "black"
            ax_main.text(cj, ci, f"{v}", ha="center", va="center",
                         color=colour, fontsize=9)

    # Marginal: column sums (top)
    ax_top.bar(range(7), col_sums, color="#444444")
    for j, v in enumerate(col_sums):
        ax_top.text(j, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.set_ylabel("class\ntotal", fontsize=8, rotation=0, labelpad=24, va="center")

    # Marginal: row sums (right)
    ax_right.barh(range(7), row_sums, color="#444444")
    for i, v in enumerate(row_sums):
        ax_right.text(v, i, f" {v}", ha="left", va="center", fontsize=8)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks([])
    ax_right.set_xticks([])
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.set_xlabel("client total", fontsize=8)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.18,
                        orientation="vertical", location="left")
    cbar.set_label("samples")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    out = FIG_DIR / "10_partition_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# =============================================================================
# Figure B — Per-client validation macro-F1 curves
#   For each client, per-client macro-F1 = mean over per-class val F1s for the
#   classes that client holds. This is the natural per-client proxy when the
#   training pipeline only logs global per-class val F1 (DermaMNIST has no
#   per-client validation split).
# =============================================================================
CLIENT_CLASSES = [
    [0, 1, 5],   # C0
    [0, 1, 5],   # C1
    [2, 3, 5],   # C2
    [2, 3, 5],   # C3
    [4, 5, 6],   # C4
    [4, 5, 6],   # C5
    [5],         # C6
]

HIST_RE = re.compile(
    r"^history_(?P<algo>fedavg|fedprox)_mu(?P<mu>[\d.]+)_E20_s(?P<seed>\d+)\.csv$"
)


def _load_per_client_curves() -> dict[str, np.ndarray]:
    """Return {algo: array of shape (num_seeds, num_rounds, num_clients)}."""
    out: dict[str, list[np.ndarray]] = {"fedavg": [], "fedprox": []}
    for path in sorted(HEADLINE.glob("history_*.csv")):
        m = HIST_RE.match(path.name)
        if m is None:
            continue
        algo = m.group("algo")
        df = pd.read_csv(path).sort_values("round")
        rounds = df["round"].values
        per_class = df[[f"val_f1_class_{c}" for c in range(7)]].values
        per_client = np.empty((len(df), 7), dtype=float)
        for ci, cls_list in enumerate(CLIENT_CLASSES):
            per_client[:, ci] = per_class[:, cls_list].mean(axis=1)
        out[algo].append((int(m.group("seed")), rounds, per_client))

    # Align all seeds onto the union of round indices (they share 1..150).
    aligned: dict[str, np.ndarray] = {}
    for algo, rows in out.items():
        if not rows:
            raise FileNotFoundError(f"no history files found for {algo}")
        max_rounds = max(int(r.max()) for _, r, _ in rows)
        arr = np.full((len(rows), max_rounds, 7), np.nan, dtype=float)
        for k, (_, rounds, per_client) in enumerate(rows):
            arr[k, rounds - 1, :] = per_client
        aligned[algo] = arr
    return aligned


def figure_b_per_client_curves() -> None:
    curves = _load_per_client_curves()
    num_seeds = curves["fedavg"].shape[0]
    rounds = np.arange(1, curves["fedavg"].shape[1] + 1)

    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.6),
                              sharex=True, sharey=True)
    axes = axes.ravel()

    for ci in range(7):
        ax = axes[ci]
        for algo, colour, label in [
            ("fedavg",  C_FEDAVG,  "FedAvg"),
            ("fedprox", C_FEDPROX, "FedProx"),
        ]:
            arr = curves[algo][:, :, ci]
            mean = np.nanmean(arr, axis=0)
            sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(num_seeds)
            ax.plot(rounds, mean, color=colour, label=label, linewidth=1.4)
            ax.fill_between(rounds, mean - sem, mean + sem,
                            color=colour, alpha=0.25, linewidth=0)

        cls_names = ", ".join(
            DISPLAY_ORDER_LABELS[DISPLAY_ORDER_IDS.index(c)]
            for c in CLIENT_CLASSES[ci]
        )
        ax.set_title(f"C{ci}: {cls_names}", fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # Hide the unused 8th panel
    axes[7].axis("off")

    # Y-axis label only on left column, X-axis only on bottom row
    for k, ax in enumerate(axes):
        if k % 4 == 0 and k < 7:
            ax.set_ylabel("val macro-F1\n(over client's classes)")
        if k >= 4:
            ax.set_xlabel("round")

    # Shared legend in the empty 8th panel
    handles = [
        plt.Line2D([0], [0], color=C_FEDAVG,  linewidth=2, label="FedAvg"),
        plt.Line2D([0], [0], color=C_FEDPROX, linewidth=2, label="FedProx"),
    ]
    axes[7].legend(handles=handles, loc="center", frameon=False,
                   title=f"$n = {num_seeds}$ paired seeds\n(shaded: $\\pm$SEM)",
                   fontsize=10, title_fontsize=10)

    fig.suptitle("Per-client validation macro-F1 (over each client's "
                 "held classes) across paired seeds", y=1.00, fontsize=11)
    out = FIG_DIR / "11_per_client_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# =============================================================================
# Figure C — Federation tax (figures/12_federation_tax.png)
# =============================================================================
TEST_RE = re.compile(
    r"^test_at_best_(?P<algo>fedavg|fedprox)_mu(?P<mu>[\d.]+)_E20_s(?P<seed>\d+)\.json$"
)


def _per_class_test_f1(algo: str) -> np.ndarray:
    """Return array of shape (num_seeds, 7) with per-class test F1."""
    rows: list[np.ndarray] = []
    for path in sorted(HEADLINE.glob("test_at_best_*.json")):
        m = TEST_RE.match(path.name)
        if m is None or m.group("algo") != algo:
            continue
        with open(path) as f:
            doc = json.load(f)
        rows.append(np.array(doc["per_class_f1"], dtype=float))
    if not rows:
        raise FileNotFoundError(f"no test_at_best files for {algo}")
    return np.vstack(rows)


def _per_class_test_f1_centralised() -> np.ndarray:
    rows: list[np.ndarray] = []
    for path in sorted(CENTRAL.glob("centralised_seed*.json")):
        with open(path) as f:
            doc = json.load(f)
        rows.append(np.array(doc["per_class_f1"], dtype=float))
    if not rows:
        raise FileNotFoundError("no centralised_seed*.json files")
    return np.vstack(rows)


def figure_c_federation_tax() -> None:
    fa = _per_class_test_f1("fedavg")
    fp = _per_class_test_f1("fedprox")
    ce = _per_class_test_f1_centralised()

    def mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = arr.shape[0]
        return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(n)

    fa_mu, fa_sem = mean_sem(fa)
    fp_mu, fp_sem = mean_sem(fp)
    ce_mu, ce_sem = mean_sem(ce)

    # Reorder per-class results into display order (prevalence)
    fa_mu_disp  = fa_mu [DISPLAY_ORDER_IDS]
    fa_sem_disp = fa_sem[DISPLAY_ORDER_IDS]
    fp_mu_disp  = fp_mu [DISPLAY_ORDER_IDS]
    fp_sem_disp = fp_sem[DISPLAY_ORDER_IDS]
    ce_mu_disp  = ce_mu [DISPLAY_ORDER_IDS]
    ce_sem_disp = ce_sem[DISPLAY_ORDER_IDS]

    x = np.arange(7)
    width = 0.27

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.bar(x - width, fa_mu_disp, width, yerr=fa_sem_disp,
           color=C_FEDAVG,  label="FedAvg",  capsize=2.5, edgecolor="white",
           linewidth=0.5, error_kw=dict(linewidth=0.8))
    ax.bar(x,         fp_mu_disp, width, yerr=fp_sem_disp,
           color=C_FEDPROX, label="FedProx", capsize=2.5, edgecolor="white",
           linewidth=0.5, error_kw=dict(linewidth=0.8))
    ax.bar(x + width, ce_mu_disp, width, yerr=ce_sem_disp,
           color=C_CENTRAL, label="Centralised", capsize=2.5, edgecolor="white",
           linewidth=0.5, error_kw=dict(linewidth=0.8))

    # Horizontal macro-F1 reference lines
    fa_macro = float(fa.mean(axis=1).mean())
    fp_macro = float(fp.mean(axis=1).mean())
    ce_macro = float(ce.mean(axis=1).mean())
    for y, colour, lab in [
        (fa_macro, C_FEDAVG,  f"FedAvg macro-F1 = {fa_macro:.3f}"),
        (fp_macro, C_FEDPROX, f"FedProx macro-F1 = {fp_macro:.3f}"),
        (ce_macro, C_CENTRAL, f"Centralised macro-F1 = {ce_macro:.3f}"),
    ]:
        ax.axhline(y, color=colour, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.text(6.55, y, f" {lab}", color=colour, va="center", fontsize=8.5)

    # Federation-tax annotation
    gap_total   = ce_macro - fa_macro
    gap_closed  = fp_macro - fa_macro
    closed_frac = (gap_closed / gap_total) if gap_total > 0 else float("nan")

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_ORDER_LABELS, rotation=15, ha="right")
    ax.set_ylabel("Test F1")
    ax.set_title(
        f"Per-class test F1 (10 seeds FedAvg/FedProx; "
        f"{ce.shape[0]} seeds centralised).  "
        f"FedProx closes {100*closed_frac:.0f}% of the FedAvg"
        f"$\\to$centralised macro-F1 gap.",
        fontsize=10.5,
    )
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, ncol=3)

    out = FIG_DIR / "12_federation_tax.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")
    print(f"  federation tax: gap_total={gap_total:.4f}, "
          f"gap_closed={gap_closed:.4f}, closed_frac={closed_frac:.3f}")


# =============================================================================
if __name__ == "__main__":
    print("Generating thesis figures 10-12 to", FIG_DIR.relative_to(ROOT))
    figure_a_partition_heatmap()
    figure_b_per_client_curves()
    figure_c_federation_tax()
    print("Done.")
