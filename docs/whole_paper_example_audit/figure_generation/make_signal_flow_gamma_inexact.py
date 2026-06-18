#!/usr/bin/env python3
"""
Generate F_signal_flow_gamma_inexact.pdf -- a conceptual (NOT quantitative)
schematic of rare-client signal flow under the two partial-update protocols.

Two stacked, visually symmetric panels share identical box positions and arrow
angles:
  (a) Drop handling   -> rare client's partial update discarded -> signal lost
  (b) gamma-inexact   -> rare client's partial update accepted  -> signal kept

Colour is used sparingly and consistently:
  grey   = normal (Client 0) update, server->global, broadcast
  orange = rare-client (Client 1) update + the dropped path / red mark in (a)
  green  = accepted rare-client update in (b)
Client 1 is highlighted orange in BOTH panels (it is the rare-class holder).

Pure matplotlib (no LaTeX engine). Output is a tight-bbox vector PDF.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects as pe

# --- consistent palette ------------------------------------------------------
GREY_FILL = "#ECECEC"
GREY_EDGE = "#555555"
RARE      = "#E67E22"   # orange: rare-class client / rare signal
DROP      = "#C0392B"   # red: dropped marker + label
ACCEPT    = "#1E8E4E"   # green: accepted update + marker + label
TXT       = "#1A1A1A"
WHITE_BBOX = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.95)
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]

# --- shared geometry (identical in both panels => symmetry) ------------------
C0 = (0.4, 3.20, 3.0, 1.00)   # Client 0  (x0, y0, w, h)  center y=3.70
C1 = (0.4, 0.70, 3.0, 1.00)   # Client 1                  center y=1.20
SV = (4.70, 1.85, 1.90, 1.20) # Server                    center y=2.45
GL = (7.50, 1.85, 2.10, 1.20) # Global model              center y=2.45
A0 = ((3.40, 3.50), (4.70, 2.70))   # Client0 -> server  (drop 0.80)
A1 = ((3.40, 1.40), (4.70, 2.20))   # Client1 -> server  (rise 0.80)  symmetric
MARK = (4.18, 1.92)                  # status glyph, on the Client1 path
PLAB = (3.70, 0.78)                  # protocol label, left-anchored clear of the box


def box(ax, spec, label, edge=GREY_EDGE, fill=GREY_FILL, lw=1.5, fs=11.5, bold=False):
    x, y, w, h = spec
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=TXT, zorder=4,
            fontweight=("bold" if bold else "normal"))


def arrow(ax, p0, p1, color=GREY_EDGE, lw=1.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                 linewidth=lw, color=color, linestyle=ls,
                 shrinkA=2, shrinkB=2, zorder=2))


def panel(ax, title, accept):
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(0.05, 4.72, title, ha="left", va="center",
            fontsize=13, fontweight="bold", color=TXT)

    # boxes (identical positions in both panels)
    box(ax, C0, "Client 0\n(common classes)")
    box(ax, C1, "Client 1\n(rare-class holder)", edge=RARE, lw=2.2)
    box(ax, SV, "Server\n(aggregation)")
    if accept:
        box(ax, GL, "Global model\n(rare signal\npreserved)", edge=RARE, lw=2.4, bold=True)
    else:
        box(ax, GL, "Global model\n(rare signal\nlost)")

    # Client 0 -> server : always solid grey (identical angle in both panels)
    arrow(ax, *A0, color=GREY_EDGE)

    # Client 1 -> server : protocol-dependent (same angle; colour/length differ)
    if accept:
        arrow(ax, A1[0], A1[1], color=ACCEPT, lw=2.4)            # full, reaches server
        ax.text(*MARK, "✓", ha="center", va="center", color=ACCEPT,
                fontsize=17, fontweight="bold", zorder=6, path_effects=HALO)
        ax.text(*PLAB, "partial update accepted ($\\gamma$-inexact)",
                ha="left", va="center", fontsize=11, color=ACCEPT,
                zorder=5, bbox=WHITE_BBOX)
    else:
        # same start/angle, but cut short before the server (update never arrives)
        arrow(ax, A1[0], (3.92, 1.73), color=RARE, lw=2.2)
        ax.text(*MARK, "✗", ha="center", va="center", color=DROP,
                fontsize=18, fontweight="bold", zorder=6, path_effects=HALO)
        ax.text(*PLAB, "partial update dropped",
                ha="left", va="center", fontsize=11, color=DROP,
                zorder=5, bbox=WHITE_BBOX)

    # server -> global model (grey)
    arrow(ax, (SV[0] + SV[2], 2.45), (GL[0], 2.45), color=GREY_EDGE)

    # broadcast (dashed): high arc server -> clients, bowed well clear of the
    # solid Client0->server arrow below it
    ax.add_patch(FancyArrowPatch((5.10, 3.10), (1.90, 4.22),
                 connectionstyle="arc3,rad=-0.48", arrowstyle="-|>",
                 mutation_scale=12, linewidth=1.3, color=GREY_EDGE,
                 linestyle=(0, (5, 3)), zorder=1))
    ax.text(5.15, 4.66, "broadcast global model $w^{t}$", ha="center", va="center",
            fontsize=10, color=GREY_EDGE, style="italic")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0))
panel(ax1, "(a) Drop handling", accept=False)
panel(ax2, "(b) $\\gamma$-inexact handling", accept=True)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.10)

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_signal_flow_gamma_inexact.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.03)
print("wrote", out)
