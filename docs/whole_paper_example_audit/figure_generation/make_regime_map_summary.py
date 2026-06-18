#!/usr/bin/env python3
"""
F_regime_map_summary.pdf -- qualitative synthesis of Chapter 5 (NOT new results).
Rows = heterogeneity regimes; columns = FedAvg / FedProx / FedNova + mechanism.
Cells are qualitative outcomes, coloured by category. "not tested" is used
wherever an optimiser was not run in that regime (no invented results, no
universal-ranking claim). Pure matplotlib -> clean vector PDF for the flat bundle.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# pastel category palette (dark text stays readable)
# colours chosen so the four categories also separate by luminance (grayscale-
# /colourblind-robust): grey(lightest) > amber > green > red(darkest). Cell text
# labels carry the meaning independently of colour.
GREEN = "#9CCC90"   # preserved / competitive / rescue / robust   (lum ~ 185)
AMBER = "#F0CE84"   # fragile gain / no extra gain                (lum ~ 207)
RED   = "#C66B5E"   # collapse / brittle                          (lum ~ 132)
GREY  = "#E8E8E8"   # not tested in this thesis                   (lum ~ 232)
EDGE  = "#888888"; TXT = "#222222"

# rows: (regime label, FA, FP, FN, mechanism)  -- each optimiser = (text, colour)
rows = [
    ("IID / near-IID",
     ("competitive", GREEN), ("no gain", AMBER), ("competitive", GREEN),
     "no heterogeneity to separate"),
    ("Statistical\nheterogeneity only",
     ("competitive", GREEN), ("fragile gain", AMBER), ("not tested", GREY),
     "uneven rare-class ownership;\nsmall per-class rebalancing"),
    ("L4 straggler\n(rare-client signal)",
     ("collapse", RED), ("rescue\n($\\gamma$-inex)", GREEN), ("not tested", GREY),
     "$\\gamma$-inexact admits the rare\nclient's dropped partial update"),
    ("Deterministic\nLR asymmetry",
     ("collapse", RED), ("collapse", RED), ("robust", GREEN),
     "$\\tau$-normalisation rescales the\nweakened rare client"),
    ("Random-$\\tau$\nstragglers",
     ("competitive", GREEN), ("competitive", GREEN), ("collapse", RED),
     "$\\tau$-normalisation amplifies\nnoisy partial updates"),
    ("Loss-side\ncorrection (L4)",
     ("competitive", GREEN), ("no extra gain", AMBER), ("not tested", GREY),
     "weighted CE re-weights rare\nclasses (a distinct layer)"),
]

# geometry
x0 = 0.0
col_reg = (0.0, 2.7)
cols = {"FedAvg": (2.7, 4.3), "FedProx": (4.3, 5.9), "FedNova": (5.9, 7.5)}
col_mech = (7.5, 12.2)
rh = 1.0
n = len(rows)
top = 0.8 + n * rh           # header band sits above the rows
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.set_xlim(0, 12.2); ax.set_ylim(0, top + 0.9 + 0.9); ax.axis("off")

# header
hy = top
def header(x0, x1, label):
    ax.text((x0 + x1) / 2, hy + 0.45, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=TXT)
header(*col_reg, "Regime")
for name, (a, b) in cols.items():
    header(a, b, name)
header(*col_mech, "Key mechanism")

# rows (top to bottom)
for i, (reg, fa, fp, fn, mech) in enumerate(rows):
    y = top - (i + 1) * rh
    # regime label
    ax.text(col_reg[0] + 0.1, y + rh / 2, reg, ha="left", va="center",
            fontsize=9, color=TXT, fontweight="bold")
    # optimiser cells
    for (text, colour), (a, b) in zip([fa, fp, fn], cols.values()):
        ax.add_patch(Rectangle((a, y + 0.08), b - a - 0.06, rh - 0.16,
                               facecolor=colour, edgecolor=EDGE, linewidth=0.8))
        ax.text((a + b) / 2 - 0.03, y + rh / 2, text, ha="center", va="center",
                fontsize=8.2, color=TXT)
    # mechanism cell
    ax.text(col_mech[0] + 0.12, y + rh / 2, mech, ha="left", va="center",
            fontsize=8.2, color=TXT)
    # light row separator
    ax.plot([col_reg[0], col_mech[1]], [y, y], color="#EEEEEE", lw=0.6, zorder=0)

# legend
ly = 0.05
items = [("preserved / competitive / rescue / robust", GREEN),
         ("fragile / no extra gain", AMBER),
         ("collapse / brittle", RED),
         ("not tested in this thesis", GREY)]
lx = 0.0
for label, colour in items:
    ax.add_patch(Rectangle((lx, ly + 0.18), 0.34, 0.34, facecolor=colour,
                           edgecolor=EDGE, linewidth=0.8))
    ax.text(lx + 0.45, ly + 0.35, label, ha="left", va="center", fontsize=7.8, color=TXT)
    lx += 0.45 + len(label) * 0.083 + 0.5

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = "F_regime_map_summary.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
