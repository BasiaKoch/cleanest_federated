#!/usr/bin/env python3
"""F_optimizer_comparison_triptych.pdf -- academic triptych comparing the three
federated optimisers used in this thesis.

All three share ONE federated loop, drawn identically in every panel:
    broadcast w^t  ->  local training on the clients  ->  return update d_i
    ->  server aggregation  ->  w^{t+1}.
A single teal accent marks ONLY the mechanism that changes, and WHERE it acts:
    (a) FedAvg  -- sample-weighted averaging              (aggregation)
    (b) FedProx -- proximal term in the local objective   (local objective)
    (c) FedNova -- work-normalised, tau-scaled updates     (aggregation)

Formulas follow the report's Methods notation (main.tex):
    FedAvg   w^{t+1} = w^t + sum_i (n_i/N) d_i,   d_i = w_i^{t+1} - w^t
    FedProx  L_CE(w) + (mu/2) || w - w^t ||^2
    FedNova  w^{t+1} = w^t + a_eff sum_i p_i d_i / a_i,   a_i = tau_i (m=0)
Pure matplotlib mathtext (no LaTeX); vector PDF.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
})

EDGE = "#33415c"      # shared box outline (slate)
ARROW = "#8C8C8C"     # shared arrows
INK = "#1a1a1a"       # node text
MUTE = "#56606b"      # protocol labels (broadcast / return)
ACCENT = "#11806a"    # the single mechanism accent (teal)
ACC_FILL = "#e4f1ed"  # light accent fill for the changed stage
WB = dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.96)

# vertical anchors, identical in every panel
Y_GLOBAL, Y_CLIENT, Y_AGG, Y_OUT = 9.0, 6.5, 4.0, 2.8
CX = [2.7, 7.3]


def box(ax, cx, cy, w, h, label, *, accent=False, fs=11.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 lw=(2.1 if accent else 1.2),
                 edgecolor=(ACCENT if accent else EDGE),
                 facecolor=(ACC_FILL if accent else "white"), zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                 lw=1.0, color=ARROW, shrinkA=2, shrinkB=2, zorder=2))


def panel(ax, tag, *, accent_stage, mech_label, mech_eq, mech_sub=None):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.2, 9.7, tag, fontsize=13, fontweight="bold", color=INK, va="center")

    acc_clients = accent_stage == "local"
    acc_agg = accent_stage == "agg"

    # shared structure (identical across panels)
    box(ax, 5.0, Y_GLOBAL, 3.0, 0.85, "global $w^{t}$")
    for cx in CX:
        box(ax, cx, Y_CLIENT, 2.7, 0.85, "client %d" % (1 if cx < 5 else 2),
            accent=acc_clients)
    box(ax, 5.0, Y_AGG, 3.2, 0.85, "aggregate", accent=acc_agg)
    ax.text(5.0, Y_OUT, "$w^{t+1}$", ha="center", va="center", fontsize=12.5,
            color=INK, fontweight="bold")

    for cx in CX:
        arrow(ax, (5.0, Y_GLOBAL - 0.45), (cx, Y_CLIENT + 0.45))   # broadcast
        arrow(ax, (cx, Y_CLIENT - 0.45), (5.0, Y_AGG + 0.45))      # return
    arrow(ax, (5.0, Y_AGG - 0.45), (5.0, Y_OUT + 0.30))            # -> w^{t+1}

    # shared protocol labels, masked over the arrows
    ax.text(3.7, 7.85, "broadcast $w^{t}$", ha="center", va="center",
            fontsize=8.8, color=MUTE, bbox=WB, zorder=5)
    ax.text(6.35, 5.3, "return $d_{i}$", ha="center", va="center",
            fontsize=8.8, color=MUTE, bbox=WB, zorder=5)

    # changed-mechanism zone (bottom): names the stage + shows the formula
    ax.text(5.0, 1.75, mech_label, ha="center", va="center", fontsize=10.5,
            color=ACCENT, fontweight="bold")
    ax.text(5.0, 0.95, mech_eq, ha="center", va="center", fontsize=14.5, color=INK)
    if mech_sub:
        ax.text(5.0, 0.28, mech_sub, ha="center", va="center", fontsize=9,
                color=MUTE)


fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.3))
panel(axes[0], "(a) FedAvg", accent_stage="agg",
      mech_label="aggregation: sample-weighted",
      mech_eq=r"$w^{t+1} = w^{t} + \sum_i \frac{n_i}{N}\, d_i$")
panel(axes[1], "(b) FedProx", accent_stage="local",
      mech_label="local objective: $+$ proximal term",
      mech_eq=r"$\mathcal{L}_{\mathrm{CE}}(w) + \frac{\mu}{2}\,\|w - w^{t}\|^{2}$")
panel(axes[2], "(c) FedNova", accent_stage="agg",
      mech_label="aggregation: work-normalised",
      mech_eq=r"$w^{t+1} = w^{t} + a_{\mathrm{eff}} \sum_i p_i\, \frac{d_i}{a_i}$",
      mech_sub=r"$a_i = \tau_i$ when $m{=}0$  (effective local work)")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.07)
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_optimizer_comparison_triptych.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.05)
print("wrote", out)
