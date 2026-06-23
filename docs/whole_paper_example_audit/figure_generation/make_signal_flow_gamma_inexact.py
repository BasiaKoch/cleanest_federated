#!/usr/bin/env python3
"""F_signal_flow_gamma_inexact.pdf -- the central rare-client signal-flow schematic.

Two panels with identical geometry; only the rare client's partial-update fate
differs:
  (a) drop          -- Client 1's partial update is blocked before aggregation,
                       so the global model loses the rare-class signal.
  (b) gamma-inexact  -- Client 1's partial update is accepted, so the rare signal
                       flows through the server into the global model.

Colour convention (consistent with the other figures):
  amber       = the rare-class signal (Client 1 holds df / mel / vasc)
  grey        = common classes (Client 0), server, broadcast
  green check = accepted        red cross = dropped
Track the amber: it reaches the global model only under gamma-inexact.

Pure matplotlib mathtext; gamma is embedded as a TrueType glyph (pdf.fonttype 42)
so it renders correctly in every viewer.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects as pe

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
})

GREY_FILL, GREY_EDGE = "#ECECEC", "#4d555e"
RARE, RARE_FILL = "#E67E22", "#fbe9d6"   # amber = rare signal; light amber fill
RARE_DK = "#B9521E"
DROP = "#C0392B"      # red cross
ACCEPT = "#1E8E4E"    # green check
INK, MUTE = "#1a1a1a", "#6b7280"
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]
WB = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.96)

# shared geometry (identical box positions in both panels)
C0 = (0.40, 3.05, 3.10, 1.15)
C1 = (0.40, 0.55, 3.10, 1.15)
SV = (4.85, 1.75, 1.95, 1.20)
GL = (7.65, 1.65, 2.25, 1.35)


def titled_box(ax, spec, title, sub, *, edge, fill, sub_color, lw=1.5, sub_bold=False):
    x, y, w, h = spec
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 lw=lw, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=8.8,
            color=sub_color, fontweight=("bold" if sub_bold else "normal"), zorder=4)


def arrow(ax, p0, p1, color, lw=1.5, ls="-", alpha=1.0, ms=12, head=True):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=("-|>" if head else "-"),
                 mutation_scale=ms, lw=lw, color=color, linestyle=ls, alpha=alpha,
                 shrinkA=2, shrinkB=2, zorder=2))


def panel(ax, tag, accept):
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(0.05, 4.76, tag, ha="left", va="center", fontsize=13,
            fontweight="bold", color=INK)

    titled_box(ax, C0, "Client 0", "common classes",
               edge=GREY_EDGE, fill=GREY_FILL, sub_color=MUTE)
    titled_box(ax, C1, "Client 1", "rare: df · mel · vasc",
               edge=RARE, fill=RARE_FILL, sub_color=RARE_DK, lw=2.3, sub_bold=True)
    titled_box(ax, SV, "Server", "aggregation",
               edge=GREY_EDGE, fill=GREY_FILL, sub_color=MUTE)

    c0_out, sv_in0 = (C0[0] + C0[2], 3.45), (SV[0], 2.62)
    c1_out, sv_in1 = (C1[0] + C1[2], 1.32), (SV[0], 2.05)
    sv_out, gl_in = (SV[0] + SV[2], 2.35), (GL[0], 2.35)

    # Client 0 -> server : common update (grey), identical in both panels
    arrow(ax, c0_out, sv_in0, color=GREY_EDGE, lw=1.5)

    if accept:
        # rare signal flows all the way through: client 1 -> server -> global (amber)
        arrow(ax, c1_out, sv_in1, color=RARE, lw=2.1)
        ax.text(4.05, 1.86, "✓", ha="center", va="center", color=ACCEPT,
                fontsize=15, fontweight="bold", zorder=6, path_effects=HALO)
        arrow(ax, sv_out, gl_in, color=RARE, lw=2.1)
        titled_box(ax, GL, "Global model", "rare signal preserved",
                   edge=RARE, fill=RARE_FILL, sub_color=RARE_DK, lw=2.4, sub_bold=True)
        ax.text(4.3, 0.42, "partial update accepted  ($\\gamma$-inexact)",
                ha="center", va="center", fontsize=11, color=ACCEPT,
                fontweight="bold", zorder=5, bbox=WB)
    else:
        # rare signal blocked before aggregation: solid amber -> X -> faded ghost
        block = (4.02, 1.80)
        arrow(ax, c1_out, block, color=RARE, lw=2.1)
        arrow(ax, block, sv_in1, color=RARE, lw=1.3, ls=(0, (2, 3)), alpha=0.30, head=False)
        ax.text(block[0], block[1], "✗", ha="center", va="center", color=DROP,
                fontsize=17, fontweight="bold", zorder=7, path_effects=HALO)
        arrow(ax, sv_out, gl_in, color=GREY_EDGE, lw=1.5)        # no rare signal carried
        titled_box(ax, GL, "Global model", "rare signal lost",
                   edge=GREY_EDGE, fill=GREY_FILL, sub_color=DROP)
        ax.text(4.3, 0.42, "partial update dropped", ha="center", va="center",
                fontsize=11, color=DROP, fontweight="bold", zorder=5, bbox=WB)

    # broadcast (dashed arc, subtle) -- server pushes the global model to clients
    ax.add_patch(FancyArrowPatch((5.25, 2.98), (1.95, 4.25),
                 connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                 mutation_scale=10, lw=1.1, color=MUTE, linestyle=(0, (5, 3)), zorder=1))
    ax.text(5.25, 4.62, "broadcast global model $w^{t}$", ha="center", va="center",
            fontsize=9.5, color=MUTE, style="italic")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.4))
panel(ax1, "(a) Drop handling", accept=False)
panel(ax2, "(b) $\\gamma$-inexact handling", accept=True)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.12)

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_signal_flow_gamma_inexact.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
print("wrote", out)
