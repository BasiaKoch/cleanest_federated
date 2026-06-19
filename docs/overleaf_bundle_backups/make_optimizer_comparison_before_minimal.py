#!/usr/bin/env python3
"""
F_optimizer_comparison_triptych.pdf -- "How the three optimisers differ".

Three aligned mini-diagrams (FedAvg / FedProx / FedNova) that share the SAME
broadcast -> local-train -> aggregate skeleton, so the single defining mechanism
of each method is the only strong visual contrast:
  FedAvg  -- plain weighted averaging (baseline, no correction)
  FedProx -- proximal penalty in the LOCAL objective (constrain drift)
  FedNova -- aggregation NORMALISED by unequal local work tau_i
Conceptual only; complements (does not duplicate) the FL-loop figure.
Pure matplotlib, house black/grey style + one calm-blue accent for the mechanism.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- house palette (matches the other conceptual diagrams) ------------------
EDGE   = "#222222"
FILL   = "#F5F5F5"   # client fill
GLFILL = "#E6E6E6"   # global-model / server fill (the coordinator)
ARROW  = "#333333"
TXT    = "#111111"
MUTE   = "#3A3A3A"
ACC_F  = "#D9E6F2"   # accent fill  (the ONE distinctive mechanism box)
ACC_E  = "#2F5D8C"   # accent edge
ACC_T  = "#1F4366"   # accent text


def box(ax, cx, cy, w, h, lines, fs, weights, edge=EDGE, fill=FILL, tcol=TXT, lw=1.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 lw=lw, edgecolor=edge, facecolor=fill, zorder=3))
    n = len(lines)
    ys = [cy + h / 2 - h * (k + 0.5) / n for k in range(n)]
    for y, ln, f, wt in zip(ys, lines, fs, weights):
        ax.text(cx, y, ln, ha="center", va="center", fontsize=f,
                color=tcol, zorder=4, fontweight=wt)


def arr(ax, p0, p1, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                 lw=lw, color=ARROW, shrinkA=2, shrinkB=2, zorder=2))


def panel(ax, title, tag, *, prox=False, fednova=False):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 9.65, title, ha="center", va="center", fontsize=14,
            fontweight="bold", color=TXT)
    ax.text(5, 8.95, tag, ha="center", va="center", fontsize=10.5,
            color=(ACC_T if (prox or fednova) else MUTE), style="italic")

    # --- global model (broadcast source) ---
    box(ax, 5, 7.95, 5.6, 0.9, ["global model  $w^{t}$"], [11], ["normal"], fill=GLFILL)

    # --- two clients (local training) ---
    cxs = [2.55, 7.45]
    if fednova:
        worklines = ["client A", "2 local steps"], ["client B", "5 local steps"]
    else:
        worklines = ["client A", "local SGD ($E$)"], ["client B", "local SGD ($E$)"]
    for cx, wl in zip(cxs, worklines):
        box(ax, cx, 5.75, 3.0, 1.0, wl, [10.5, 9.5], ["bold", "normal"])
        arr(ax, (cx, 7.5), (cx, 6.28))           # broadcast down

    # --- FedProx: the ONE difference is in the LOCAL objective (accent) ---
    if prox:
        box(ax, 5, 4.35, 7.4, 0.85,
            ["local objective:  CE $+\\;\\frac{\\mu}{2}\\,\\|w-w^{t}\\|^{2}$"],
            [11], ["bold"], edge=ACC_E, fill=ACC_F, tcol=ACC_T, lw=2.0)
        for cx in cxs:
            arr(ax, (cx, 5.25), (cx, 4.80))        # client -> local-objective box
        arr(ax, (5, 3.90), (5, 3.48))              # local-objective -> aggregation
    else:
        for cx in cxs:
            arr(ax, (cx, 5.25), (5, 3.48))         # clients -> aggregation

    # --- aggregation (server) -- accent only for FedNova ---
    if fednova:
        box(ax, 5, 2.95, 7.4, 1.0,
            ["server aggregation", "normalise by work $\\tau_i$:  $\\sum_i p_i\\, d_i / a_i$"],
            [10.5, 11], ["bold", "bold"], edge=ACC_E, fill=ACC_F, tcol=ACC_T, lw=2.0)
    else:
        box(ax, 5, 2.95, 7.4, 1.0,
            ["server aggregation", "weighted average  $\\sum_i \\frac{n_i}{N}\\, w_i$"],
            [10.5, 11], ["bold", "normal"])

    # --- one short thesis-relevance note (muted, does not dominate) ---
    if prox:
        note = "stabilises drift via $\\mu$; the L4 rescue is mainly\n$\\gamma$-inexact acceptance, not $\\mu$"
    elif fednova:
        note = "helps under deterministic unequal work,\nbut brittle under random $\\tau$"
    else:
        note = "no drift control or work normalisation;\nunder-weights a small rare-class client"
    ax.text(5, 1.15, note, ha="center", va="center", fontsize=8.5,
            color=MUTE, style="italic")


fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.2))
panel(axes[0], "FedAvg", "baseline averaging")
panel(axes[1], "FedProx", "constrained local drift", prox=True)
panel(axes[2], "FedNova", "normalise unequal work", fednova=True)

# faint separators between panels
for x in (0.365, 0.635):
    fig.add_artist(plt.Line2D([x, x], [0.06, 0.92], color="#CCCCCC", lw=0.8))

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.10)
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_optimizer_comparison_triptych.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
print("wrote", out)
