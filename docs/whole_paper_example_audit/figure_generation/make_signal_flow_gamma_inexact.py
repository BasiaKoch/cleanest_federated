#!/usr/bin/env python3
"""F_signal_flow_gamma_inexact.pdf -- the central rare-client signal-flow schematic.

In the L4 partition Client 1 uniquely holds the rare classes (df / mel / vasc).
Two panels, identical geometry, compare the two partial-update protocols:
  (a) drop          -- Client 1's partial update is blocked before aggregation;
                       the global model loses the rare-class signal.
  (b) gamma-inexact  -- Client 1's partial update is accepted into aggregation;
                       the global model preserves the rare-class signal.

Conceptual (protocol-level, not a FedAvg-vs-FedProx claim). Colour convention:
  amber  = the rare-client pathway (Client 1 + its update), the visual focus
  grey   = neutral common/plumbing (Client 0, server, broadcast, result arrow)
  muted red = dropped / lost        muted green = accepted / preserved
gamma is embedded as a TrueType glyph (pdf.fonttype 42) so it always renders.
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

GREY_F, GREY_E = "#ECECEC", "#7d858e"      # neutral fill / edge
ORANGE, ORANGE_F, ORANGE_DK = "#E67E22", "#fbe9d6", "#B9521E"   # rare pathway
RED = "#B5564A"                            # muted: dropped / lost
GREEN = "#4E9E6B"                          # muted: accepted / preserved
INK, MUTE = "#1a1a1a", "#6b7280"

# identical box geometry in both panels (x, y, w, h)
C0 = (0.45, 4.05, 2.75, 0.95)
C1 = (0.45, 2.00, 2.75, 0.95)
SV = (4.60, 2.95, 1.80, 1.05)
GL = (7.25, 2.85, 2.45, 1.25)


def cx(b): return b[0] + b[2] / 2
def cy(b): return b[1] + b[3] / 2


def titled_box(ax, spec, title, sub, *, edge, fill, sub_color, lw=1.4, sub_bold=False):
    x, y, w, h = spec
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                 lw=lw, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.4,
            color=sub_color, fontweight=("bold" if sub_bold else "normal"), zorder=4)


def arrow(ax, p0, p1, color, lw=1.4, ls="-", head=True, ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=("-|>" if head else "-"),
                 mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                 shrinkA=1, shrinkB=1, zorder=2))


def panel(ax, tag, accept):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.05, 5.72, tag, ha="left", va="center", fontsize=13,
            fontweight="bold", color=INK)

    # faint, secondary broadcast arc: server -> clients
    ax.add_patch(FancyArrowPatch((4.95, 4.02), (1.95, 5.28),
                 connectionstyle="arc3,rad=-0.40", arrowstyle="-|>",
                 mutation_scale=8, lw=1.0, color="#c2c7cd", linestyle=(0, (4, 3)), zorder=1))
    ax.text(4.6, 5.42, "broadcast $w^{t}$", ha="center", va="center",
            fontsize=8.2, color="#aeb4bb", style="italic")

    # boxes (identical positions across panels)
    titled_box(ax, C0, "Client 0", "common classes", edge=GREY_E, fill=GREY_F, sub_color=MUTE)
    titled_box(ax, C1, "Client 1", "rare: df · mel · vasc", edge=ORANGE, fill=ORANGE_F,
               sub_color=ORANGE_DK, lw=2.1, sub_bold=True)
    titled_box(ax, SV, "Server", "aggregation", edge=GREY_E, fill=GREY_F, sub_color=MUTE)

    # Client 0 -> server : neutral grey (same in both panels)
    arrow(ax, (C0[0] + C0[2], 4.30), (SV[0], 3.72), color=GREY_E, lw=1.4)
    # server -> global : neutral grey result arrow
    arrow(ax, (SV[0] + SV[2], cy(SV)), (GL[0], cy(SV)), color=GREY_E, lw=1.4)

    c1_out = (C1[0] + C1[2], 2.55)
    sv_in = (SV[0], 3.25)

    if accept:
        # rare update accepted: solid amber arrow enters the server, green check
        arrow(ax, c1_out, sv_in, color=ORANGE, lw=2.0)
        ax.text(4.18, 3.06, "✓", ha="center", va="center", color=GREEN,
                fontsize=14, fontweight="bold", zorder=6)
        titled_box(ax, GL, "Global model", "rare signal preserved",
                   edge=ORANGE, fill=ORANGE_F, sub_color=GREEN, lw=2.1, sub_bold=True)
        ax.text(4.75, 1.42, "partial update accepted  ($\\gamma$-inexact)",
                ha="center", va="center", fontsize=10.5, color=GREEN, fontweight="bold")
    else:
        # rare update dropped: dashed amber arrow stops at a neat block before server
        block = (4.18, 3.02)
        arrow(ax, c1_out, block, color=ORANGE, lw=1.8, ls=(0, (4, 2.4)), head=False)
        ax.text(block[0], block[1], "✕", ha="center", va="center", color=RED,
                fontsize=13.5, fontweight="bold", zorder=6)
        titled_box(ax, GL, "Global model", "rare signal lost",
                   edge=GREY_E, fill=GREY_F, sub_color=RED)
        ax.text(4.75, 1.42, "partial update dropped", ha="center", va="center",
                fontsize=10.5, color=RED, fontweight="bold")

    # subtle annotation under Client 1 (well spaced below the box)
    ax.text(cx(C1), 1.42, "only source of rare-class\nsignal in L4", ha="center",
            va="center", fontsize=7.8, color=ORANGE_DK, style="italic", linespacing=1.25)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.4))
panel(ax1, "(a) Drop handling", accept=False)
panel(ax2, "(b) $\\gamma$-inexact handling", accept=True)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.14)

HERE = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(HERE, "F_signal_flow_gamma_inexact")
fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(base + ".png", dpi=200, bbox_inches="tight", pad_inches=0.05)
print("wrote", base + ".pdf", "and .png")
