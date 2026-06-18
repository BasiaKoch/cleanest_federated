#!/usr/bin/env python3
"""
F_federated_learning_workflow.pdf -- a compact, publication-style FL-loop diagram
for reader orientation (conceptual, NOT quantitative, NOT a deployment claim).
Pure matplotlib, black/grey academic styling (no colour).

Server (global model + aggregation) centred at top; K clients evenly spaced below.
One round:  1 broadcast w^t  ->  2 local training on private D_i  ->
            3 return update d_i (not raw data)  ->  4 aggregate w^{t+1}
The whole loop repeats for R rounds. Private data never leaves the client.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- academic black/grey palette (no colour) --------------------------------
EDGE   = "#222222"   # box outlines
FILL   = "#F5F5F5"   # client fill
SVFILL = "#E6E6E6"   # server fill (slightly darker to read as the coordinator)
ARROW  = "#333333"   # all flow arrows
TXT    = "#111111"   # primary text
MUTE   = "#3A3A3A"   # step/annotation text
WBOX   = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92)

def box(ax, cx, cy, w, h, lines, fills, fs_list, weights, edge=EDGE, fill=FILL):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09",
                 lw=1.5, edgecolor=edge, facecolor=fill, zorder=3))
    n = len(lines)
    # stack lines vertically, centred
    ys = [cy + h / 2 - h * (k + 0.5) / n for k in range(n)]
    for y, ln, f, w_ in zip(ys, lines, fs_list, weights):
        ax.text(cx, y, ln, ha="center", va="center", fontsize=f,
                color=TXT, zorder=4, fontweight=w_)

def arrow(ax, p0, p1, lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                 lw=lw, color=ARROW, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}",
                 shrinkA=2, shrinkB=2, zorder=2))

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.set_aspect("equal")

# --- server (top, centred): holds global model + does aggregation -----------
SV_CY, SV_TOP, SV_BOT = 5.02, 5.55, 4.50
box(ax, 5.0, SV_CY, 4.9, 1.05,
    ["Server  $\\cdot$  global model $w^{t}$",
     "4. aggregate:  $w^{t+1}=\\sum_i \\dfrac{n_i}{N}\\, w_i^{t+1}$"],
    None, [12.5, 11.0], ["bold", "normal"], fill=SVFILL)

# --- repeat loop: one clean arc bowing up over the server --------------------
ax.add_patch(FancyArrowPatch((6.0, SV_TOP - 0.02), (4.0, SV_TOP - 0.02),
             arrowstyle="-|>", mutation_scale=13, lw=1.6, color=ARROW,
             connectionstyle="arc3,rad=0.6", shrinkA=1, shrinkB=1, zorder=6))
ax.text(5.0, 6.42, "repeat for $R$ rounds", ha="center", va="center",
        fontsize=11.0, color=MUTE, style="italic", zorder=6)

# --- clients (row, evenly spaced) -------------------------------------------
cx     = [1.7, 5.0, 8.3]
cname  = ["Client 1", "Client 2", "Client $K$"]
cdata  = ["private data $D_1$", "private data $D_2$", "private data $D_K$"]
CL_TOP, CL_CY = 2.36, 1.82
for x, nm, dl in zip(cx, cname, cdata):
    box(ax, x, CL_CY, 2.3, 1.08, [nm, dl], None, [12.0, 10.5],
        ["bold", "normal"], fill=FILL)

# --- broadcast (down) + return (up) arrows: symmetric, paired ----------------
anc = [4.05, 5.0, 5.95]   # server-edge anchors aligned under the clients
off = 0.17
for x, a in zip(cx, anc):
    arrow(ax, (a - off, SV_BOT), (x - off, CL_TOP))   # 1. broadcast  (down)
    arrow(ax, (x + off, CL_TOP), (a + off, SV_BOT))   # 3. return     (up)

# --- concise step labels (white-masked so they read over the arrows) ---------
ax.text(3.32, 3.46, "1. broadcast $w^{t}$", ha="center", va="center",
        fontsize=11.0, color=MUTE, bbox=WBOX, zorder=5)
ax.text(6.68, 3.46, "3. return update $d_i$", ha="center", va="center",
        fontsize=11.0, color=MUTE, bbox=WBOX, zorder=5)
ax.text(5.0, 0.70, "2. local training ($E$ epochs) on $D_i$ — data stays on the client",
        ha="center", va="center", fontsize=11.0, color=MUTE, style="italic", zorder=5)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_federated_learning_workflow.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.03)
print("wrote", out)
