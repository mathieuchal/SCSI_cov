"""Shared plotting style (dataviz method: fixed-order categorical hues, thin marks, recessive grid, ink tokens)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
# categorical slots 1-3 of the reference palette (validated all-pairs) + neutral gray for the raw baseline
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
YELLOW = "#eda100"      # slot 4; keep it away from orange in grouped bars (documented confusable pair)
COL = {"Nonlinear shrinkage (LW)": YELLOW, "LW nonlinear shrinkage": YELLOW, "SC-SI (ours)": BLUE, "IW conjugate (ML-fitted)": ORANGE, "Linear shrinkage (OAS)": AQUA, "Sample cov.": GRAY,
       "SC-SI": BLUE, "IW": ORANGE, "OAS shrinkage": AQUA, "SCM": GRAY}


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.family": "sans-serif", "figure.dpi": 130, "savefig.dpi": 200,
    })
