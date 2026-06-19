#!/usr/bin/env python3
"""
F_optimizer_comparison_triptych.pdf -- minimal academic schematic:
"How do FedAvg, FedProx, and FedNova differ?"

Three identical-layout panels (global w^t -> two clients -> aggregate). The ONLY
difference is one subtle accent-coloured mechanism label per panel:
  (a) FedAvg  -- weighted average        (at aggregation)
  (b) FedProx -- proximal local loss     (at the local objective)
  (c) FedNova -- work-normalised update  (at aggregation)
No sentences, no bottom notes -- interpretation lives in the caption.
Pure matplotlib, hairline strokes, mostly white, one accent hue.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

EDGE = "#9A9A9A"   # thin grey box outline
ARR  = "#8C8C8C"   # thin grey arrow
TXT  = "#1A1A1A"   # node text
ACC  = "#2F5D8C"   # single subtle accent (mechanism label only)
WBBOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95)


def node(ax, cx, cy, w, h, label):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 lw=0.9, edgecolor=EDGE, facecolor="white", zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5,
            color=TXT, zorder=4)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                 lw=0.9, color=ARR, shrinkA=2, shrinkB=2, zorder=2))


def panel(ax, tag, *, mech_at, mech_label, mech_eq):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.3, 9.3, tag, fontsize=12.5, fontweight="bold", color=TXT, va="center")

    node(ax, 5.0, 7.7, 2.6, 0.8, "global $w^{t}$")
    cxs = [3.05, 6.95]
    for cx in cxs:
        node(ax, cx, 5.2, 2.5, 0.8, "client %d" % (1 if cx < 5 else 2))
    node(ax, 5.0, 2.6, 2.9, 0.8, "aggregate")

    for cx in cxs:
        arrow(ax, (5.0, 7.3), (cx, 5.62))     # broadcast: global -> client
        arrow(ax, (cx, 4.78), (5.0, 3.02))    # return: client -> aggregate

    # one subtle accent mechanism label (text only; white-masked so it reads cleanly)
    yl = 1.55 if mech_at == "agg" else 3.95
    ax.text(5.0, yl, mech_label, ha="center", va="center", fontsize=10.0,
            color=ACC, fontweight="bold", zorder=5, bbox=WBBOX)
    if mech_eq:
        ax.text(5.0, yl - 0.62, mech_eq, ha="center", va="center", fontsize=10.0,
                color=ACC, zorder=5, bbox=WBBOX)


fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
panel(axes[0], "(a) FedAvg",  mech_at="agg",   mech_label="weighted average",
      mech_eq=r"$\sum_i (n_i/N)\,w_i$")
panel(axes[1], "(b) FedProx", mech_at="local", mech_label="proximal local loss",
      mech_eq=r"$L_k + \frac{\mu}{2}\,\|w-w^{t}\|^{2}$")
panel(axes[2], "(c) FedNova", mech_at="agg",   mech_label="work-normalised update",
      mech_eq=r"$\sum_i p_i\, d_i / a_i$")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.06)
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_optimizer_comparison_triptych.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
print("wrote", out)
