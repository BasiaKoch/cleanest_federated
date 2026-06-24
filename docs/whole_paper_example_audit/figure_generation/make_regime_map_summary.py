#!/usr/bin/env python3
"""F_regime_map_summary.pdf -- qualitative synthesis of Chapter 5 (NOT new results).

A final-discussion regime map: rows = heterogeneity regimes, columns =
FedAvg / FedProx / FedNova, plus a concise "key mechanism" column. Each cell is
triple-coded (a glyph + a short word + a muted colour) so the status reads even
in greyscale or for colourblind readers; it is NOT a traffic-light slide. Cells
where an optimiser was not run in a regime are a recessive grey "not tested"
(no invented results, no universal-ranking claim).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
})

# muted, colourblind-safer palette (teal/sand/clay/grey -- not green/red), each
# category ALSO carries a distinct glyph so colour is never load-bearing.
TEAL = "#7FA89B"    # preserved / competitive / rescue / robust   ✓
SAND = "#DCC68C"    # fragile / no extra gain                     ~
CLAY = "#C68B77"    # collapse / brittle                          ✗
GREY = "#EFEFEF"    # not tested in this thesis                   — (recessive)
EDGE, TXT, DIM = "#9a9a9a", "#222222", "#9a9a9a"
CAT = {"good": (TEAL, "✓"), "part": (SAND, "~"), "bad": (CLAY, "✗"), "nt": (GREY, "—")}

# rows: regime, (cat,label) for FA / FP / FN, concise mechanism
rows = [
    ("IID / near-IID",
     ("good", "competitive"), ("part", "no gain"), ("good", "competitive"),
     "no heterogeneity to exploit"),
    ("Statistical\nheterogeneity",
     ("good", "competitive"), ("part", "fragile gain"), ("nt", "not tested"),
     "small per-class rebalancing"),
    ("L4 straggler\n(rare-client signal)",
     ("bad", "collapse"), ("good", "rescue"), ("nt", "not tested"),
     "$\\gamma$-inexact admits the dropped\nrare-client update"),
    ("Deterministic\nLR asymmetry",
     ("bad", "collapse"), ("bad", "collapse"), ("good", "robust"),
     "$\\tau$-normalisation rescales the\nweakened client"),
    ("Random-$\\tau$\nstragglers",
     ("good", "competitive"), ("good", "competitive"), ("bad", "collapse"),
     "$\\tau$-normalisation amplifies\nupdate noise"),
    ("Loss-side\ncorrection (L4)",
     ("good", "competitive"), ("part", "no extra gain"), ("nt", "not tested"),
     "weighted-CE re-weights the\nrare classes"),
]

# geometry
col_reg = (0.0, 2.7)
cols = {"FedAvg": (2.85, 4.45), "FedProx": (4.55, 6.15), "FedNova": (6.25, 7.85)}
col_mech = (8.1, 12.2)
rh, n = 1.0, len(rows)
top = 0.95 + n * rh
fig, ax = plt.subplots(figsize=(9.8, 6.6))
ax.set_xlim(0, 12.2); ax.set_ylim(0, top + 2.0); ax.axis("off")

# title + subtitle
ax.text(0.0, top + 1.55, "Regime map: optimiser outcomes by heterogeneity regime",
        ha="left", va="center", fontsize=13, fontweight="bold", color=TXT)
ax.text(0.0, top + 1.08,
        "Qualitative synthesis of the tested regimes — not new results, and not a universal optimiser ranking.",
        ha="left", va="center", fontsize=9.2, style="italic", color="#5b6168")

# header
def header(a, b, label):
    ax.text((a + b) / 2, top + 0.42, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=TXT)
header(col_reg[0] + 0.1, col_reg[0] + 1.0, "Regime")
for name, (a, b) in cols.items():
    header(a, b, name)
header(col_mech[0], col_mech[0] + 2.0, "Key mechanism")
ax.plot([0, col_mech[1]], [top + 0.02, top + 0.02], color="#cccccc", lw=1.0)

# column separators
for x in (cols["FedProx"][0] - 0.05, cols["FedNova"][0] - 0.05):
    ax.plot([x, x], [0.95, top], color="#eeeeee", lw=0.8, zorder=0)
ax.plot([col_mech[0] - 0.12, col_mech[0] - 0.12], [0.95, top], color="#cccccc", lw=1.0)

# rows
for i, (reg, fa, fp, fn, mech) in enumerate(rows):
    y = top - (i + 1) * rh
    ax.text(col_reg[0] + 0.05, y + rh / 2, reg, ha="left", va="center",
            fontsize=9, color=TXT, fontweight="bold")
    for (cat, label), (a, b) in zip([fa, fp, fn], cols.values()):
        colour, glyph = CAT[cat]
        secondary = cat == "nt"
        ax.add_patch(FancyBboxPatch((a, y + 0.13), b - a - 0.04, rh - 0.26,
                     boxstyle="round,pad=0.01,rounding_size=0.06",
                     facecolor=colour, edgecolor=("none" if secondary else EDGE),
                     linewidth=0.9, zorder=2))
        ax.text((a + b) / 2, y + rh * 0.60, glyph, ha="center", va="center",
                fontsize=12.5, color=(DIM if secondary else TXT),
                fontweight="bold", zorder=3)
        ax.text((a + b) / 2, y + rh * 0.28, label, ha="center", va="center",
                fontsize=8, color=(DIM if secondary else TXT),
                style=("italic" if secondary else "normal"), zorder=3)
    ax.text(col_mech[0], y + rh / 2, mech, ha="left", va="center",
            fontsize=8.4, color="#3a3f45")
    if i > 0:
        ax.plot([0, col_mech[1]], [y + rh, y + rh], color="#f0f0f0", lw=0.6, zorder=0)

# legend (glyph + muted swatch + label)
items = [("good", "preserved / competitive / rescue / robust"),
         ("part", "fragile / no extra gain"),
         ("bad", "collapse / brittle"),
         ("nt", "not tested in this thesis")]
lx, ly = 0.0, 0.18
for cat, label in items:
    colour, glyph = CAT[cat]
    ax.add_patch(FancyBboxPatch((lx, ly), 0.40, 0.40,
                 boxstyle="round,pad=0.01,rounding_size=0.06",
                 facecolor=colour, edgecolor=("none" if cat == "nt" else EDGE), linewidth=0.8))
    ax.text(lx + 0.20, ly + 0.20, glyph, ha="center", va="center", fontsize=9.5,
            color=(DIM if cat == "nt" else TXT), fontweight="bold")
    ax.text(lx + 0.52, ly + 0.20, label, ha="left", va="center", fontsize=7.8, color=TXT)
    lx += 0.52 + len(label) * 0.082 + 0.55

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "F_regime_map_summary.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.05)
print("wrote", out)
