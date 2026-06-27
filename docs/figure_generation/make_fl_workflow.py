#!/usr/bin/env python3
"""F_federated_learning_workflow.pdf -- polished, conceptual FL-loop schematic
for the Methods chapter (reader orientation; NOT quantitative, NOT a deployment
claim).

One communication round:
  1. the server broadcasts the current global model  w_t
  2. each client trains locally on its private data   D_i   (data never leaves)
  3. clients return only the model update             Delta_i   (not raw data)
  4. the server aggregates                            w_{t+1} = w_t + sum_i (n_i/N) Delta_i
The loop repeats for R rounds.  Central message: only model updates are shared.

Pure matplotlib (mathtext; no LaTeX). Restrained greyscale palette with a single
teal accent reserved for the "model-update-only" return path and the takeaway.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,   # embed TrueType — crisp, selectable text in the PDF
})

# --- restrained palette: greyscale + ONE teal accent ("update only") ---------
INK       = "#1b1b1b"   # primary text
EDGE      = "#454b54"   # client outlines
CLIENT_FC = "#f5f6f8"   # client fill
SERVER_EC = "#33415c"   # server outline (slate — reads as the coordinator)
SERVER_FC = "#eaeef4"   # server fill
GREY      = "#7a828c"   # broadcast path (neutral)
ACCENT    = "#11806a"   # return / model-update path (teal) — the central message
DATA_EC   = "#9aa1a9"   # dashed private-data enclosure
MUTE      = "#4d555e"   # secondary labels
WBOX = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.95)


def rbox(ax, cx, cy, w, h, ec, fc, lw=1.3, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 lw=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z))


def arrow(ax, p0, p1, color, lw=1.1, ms=11, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                 lw=lw, color=color, shrinkA=2, shrinkB=2, zorder=z))


def padlock(ax, cx, cy, s=0.17, color=DATA_EC):
    """A small, subtle vector padlock = data is locked inside the client."""
    ax.add_patch(FancyBboxPatch((cx - 0.6 * s, cy - 0.55 * s), 1.2 * s, 0.95 * s,
                 boxstyle="round,pad=0,rounding_size=0.03",
                 lw=1.0, edgecolor=color, facecolor="white", zorder=6))
    ax.add_patch(Arc((cx, cy + 0.40 * s), 0.78 * s, 0.78 * s, theta1=0, theta2=180,
                 lw=1.0, edgecolor=color, zorder=6))


fig, ax = plt.subplots(figsize=(7.3, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off"); ax.set_aspect("equal")

# --- server (top, centred) ---------------------------------------------------
SV_CX, SV_CY, SV_W, SV_H = 5.0, 5.38, 6.4, 1.18
SV_BOT, SV_TOP = SV_CY - SV_H / 2, SV_CY + SV_H / 2
rbox(ax, SV_CX, SV_CY, SV_W, SV_H, SERVER_EC, SERVER_FC, lw=1.6)
ax.text(SV_CX, SV_CY + 0.28, r"Server  $\cdot$  global model $w^{t}$", ha="center",
        va="center", fontsize=12.5, color=INK, fontweight="bold", zorder=4)
ax.text(SV_CX, SV_CY - 0.25,
        r"4. aggregate (FedAvg):   $w^{t+1} = \sum_i \frac{n_i}{N}\, w_i^{t+1}$",
        ha="center", va="center", fontsize=11.5, color=INK, zorder=4)

# repeat-loop arc bowing over the server
ax.add_patch(FancyArrowPatch((SV_CX + 0.95, SV_TOP + 0.02), (SV_CX - 0.95, SV_TOP + 0.02),
             arrowstyle="-|>", mutation_scale=11, lw=1.1, color=MUTE,
             connectionstyle="arc3,rad=0.55", shrinkA=1, shrinkB=1, zorder=6))
ax.text(SV_CX, SV_TOP + 0.80, r"repeat for $R$ rounds", ha="center", va="center",
        fontsize=10.5, color=MUTE, style="italic", zorder=6)

# --- clients (evenly spaced row) ---------------------------------------------
CX = [1.8, 5.0, 8.2]
CNAMES = ["Client 1", "Client 2", r"Client $K$"]
DLAB = [r"$D_1$", r"$D_2$", r"$D_K$"]
CL_CY, CL_W, CL_H = 1.60, 2.5, 1.44
CL_TOP = CL_CY + CL_H / 2
for x, nm, dl in zip(CX, CNAMES, DLAB):
    rbox(ax, x, CL_CY, CL_W, CL_H, EDGE, CLIENT_FC, lw=1.3)
    ax.text(x, CL_CY + 0.47, nm, ha="center", va="center", fontsize=11.5,
            color=INK, fontweight="bold", zorder=4)
    # dashed private-data enclosure: raw data stays inside the client
    rbox(ax, x, CL_CY - 0.27, CL_W - 0.6, 0.74, DATA_EC, "white", lw=1.0, ls="--", z=4)
    padlock(ax, x - 0.62, CL_CY - 0.27)
    ax.text(x + 0.18, CL_CY - 0.27, dl + r"  (private)", ha="center", va="center",
            fontsize=10.0, color=MUTE, zorder=5)

# --- broadcast (down, neutral) + model-update return (up, teal accent) -------
def sx(cx):  # server-edge anchor pulled toward the centre so arrows converge
    return SV_CX + (cx - SV_CX) * 0.42

OFF = 0.17
for x in CX:
    a = sx(x)
    arrow(ax, (a - OFF, SV_BOT), (x - OFF, CL_TOP), GREY, lw=1.0)          # broadcast
    arrow(ax, (x + OFF, CL_TOP), (a + OFF, SV_BOT), ACCENT, lw=1.4, z=3)   # update

# --- minimal step labels (white-masked so they read over the arrows) ---------
ax.text(2.45, 3.55, r"1. broadcast $w^{t}$", ha="center", va="center",
        fontsize=10.8, color=MUTE, bbox=WBOX, zorder=7)
ax.text(7.62, 3.55, r"3. return $d_i$", ha="center", va="center",
        fontsize=10.8, color=ACCENT, fontweight="bold", bbox=WBOX, zorder=7)
ax.text(5.0, 3.04, r"2. local training ($E$ epochs)", ha="center", va="center",
        fontsize=10.5, color=MUTE, style="italic", bbox=WBOX, zorder=7)

# --- central takeaway --------------------------------------------------------
ax.text(5.0, 0.30,
        r"Only model updates ($d_i$) are shared — raw data $D_i$ never leaves the client.",
        ha="center", va="center", fontsize=10.8, color=ACCENT, fontweight="bold",
        zorder=7)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_federated_learning_workflow.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.05)
print("wrote", out)
