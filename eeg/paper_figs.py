"""Camera-ready figures for the experiments section (vector PDF, sized for one text-width column) -> paper/fig_real/.

  fig_toy_logscore : validation log-score gap to the oracle vs EM iteration, four d=8 toys
  fig_toy_stats    : posterior of six statistics at a *median-error* test task, four d=8 toys
  fig_eeg          : (a) log-score gain, (b) 80% coverage per metric, (c) selective classification, cross-validated over all subjects   [PNG, 7.4 in wide]
  fig_finance      : (a) min-variance ratio vs OAS, (b) calibration through time, (c) log-score gain through time   [PNG, 7.4 in wide]

All inputs are committed result files (results/*.json) plus the per-case arrays (gitignored) they were derived from.
Usage: python paper_figs.py [name ...]   (no argument = all)
"""
import json, os, re, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from plotstyle import *
from toy_eval import STATS, w1_norm

setup()
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6.2, "pdf.fonttype": 42, "axes.titlelocation": "left", "lines.linewidth": 1.4})
PDF, PNG = "paper/fig_real", "figs/paper"
os.makedirs(PDF, exist_ok=True); os.makedirs(PNG, exist_ok=True)
EEG_TAG = "eeg_w4_s0_k1.0_e0.65"
FIN_TAG = "fin_d12_nw63_s0_wcsel_wide_k30_ks6-2-6"
HI = "#eb6834"
AR1 = "ar1_d8_M8000_n120"                                                       # training run of the AR(1) toy shown in the paper
LIGHT_BLUE, LIGHT_ORANGE = "#79aee8", "#f2a88e"


def save(fig, name, fmt="pdf"):
    if fmt == "png":                                                             # raster-only figure, shipped as PNG (no PDF, no preview copy)
        fig.savefig(f"{PDF}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote", f"{PDF}/{name}.png"); return
    fig.savefig(f"{PDF}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{PNG}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", f"{PDF}/{name}.pdf")

# ------------------------------------------------------------------------------------------------ EEG
def fig_eeg():
    """Wider than the other figures (7.4 in) so that the three panels are individually readable; PNG output."""
    # (a),(b): 5-fold subject-disjoint CV over all 109 subjects, pooled by eeg_cv_pool_calib.py (same run as panel (c))
    nw = json.load(open(f"results/{EEG_TAG}_rvlogscore_cv5_nextwindow.json")); sh = json.load(open(f"results/{EEG_TAG}_rvlogscore_cv5_splithalf.json"))
    M = {"Sample cov.": ("sample cov.", GRAY), "Linear shrinkage (OAS)": ("OAS", AQUA), "Nonlinear shrinkage (LW)": ("LW-NLS", YELLOW),
         "IW conjugate (ML-fitted)": ("IW conj", ORANGE), "SC-SI (ours)": ("SC-SI", BLUE)}
    rc = {"font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "normal", "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.4, 2.4)); gs = fig.add_gridspec(1, 3, width_ratios=[0.98, 1.38, 1.05], wspace=0.5, left=0.06, right=0.995, top=0.9, bottom=0.34)
        a0, a1, a2 = (fig.add_subplot(gs[0, k]) for k in range(3))
        # (a) log-score gain vs sample covariance
        names = ["Linear shrinkage (OAS)", "Nonlinear shrinkage (LW)", "IW conjugate (ML-fitted)", "SC-SI (ours)"]; ys = np.arange(len(names))[::-1] * 1.0
        for y, n in zip(ys, names):
            g, lo, hi = sh["logscore"][n]["gain"]; c = M[n][1]
            a0.plot([lo, hi], [y + 0.16] * 2, color=c, lw=1.8, solid_capstyle="round"); a0.plot(g, y + 0.16, "o", color=c, ms=4.6, mec=SURFACE, mew=0.7)
            g2, lo2, hi2 = nw["methods"][n]["gain_vs_scm"], *nw["methods"][n]["gain_ci"]
            a0.plot([lo2, hi2], [y - 0.16] * 2, color=c, lw=1.8, alpha=0.5, solid_capstyle="round"); a0.plot(g2, y - 0.16, "o", mfc=SURFACE, mec=c, mew=1.3, ms=4.6)
        a0.axvline(0, color=INK3, lw=0.9); a0.set_yticks(ys); a0.set_yticklabels([M[n][0] for n in names], rotation=90, va="center", fontsize=6.3); a0.grid(axis="y", visible=False)
        a0.set_xlabel("nats / window"); a0.set_title("(a) Log-score gain", loc="left"); a0.set_xlim(-25, 42)
        a0.plot([], [], "o", color=INK2, ms=4.2, label="split-half"); a0.plot([], [], "o", mfc=SURFACE, mec=INK2, mew=1.2, ms=4.2, label="next window")
        a0.legend(loc="upper right", fontsize=6.3, frameon=False, borderaxespad=0.1, handletextpad=0.2, labelspacing=0.25)
        # (b) coverage of the central 80% interval, per metric (split-half)
        groups = [("dir.\nvar.", lambda m: sh["dir_cov"][m]["0.8"]), ("log\ndet", lambda m: sh["functional"]["log det C"][m]["cov"]["0.8"]),
                  ("log\nOz", lambda m: sh["functional"]["log C[Oz,Oz]"][m]["cov"]["0.8"]), ("top\nshare", lambda m: sh["functional"]["top-eigenvalue share"][m]["cov"]["0.8"]),
                  ("log\ncond.", lambda m: sh["functional"]["log cond. number"][m]["cov"]["0.8"])]
        xs = np.arange(len(groups)); wb = 0.16
        for j, m in enumerate(M):
            a1.bar(xs + (j - 2) * wb, [g[1](m) for g in groups], wb * 0.92, color=M[m][1])
        a1.axhline(0.8, color=INK, lw=1.3, ls=(0, (4, 2))); a1.set_ylim(0, 1.0)
        a1.set_xticks(xs); a1.set_xticklabels([g[0] for g in groups]); a1.grid(axis="x", visible=False)
        a1.set_ylabel("80% coverage"); a1.set_title("(b) Calibration, split-half", loc="left")
        # (c) selective classification, eyes open vs closed: 5-fold subject-disjoint CV, all 109 subjects (eeg_cv_classif.py)
        cv = json.load(open(f"results/{EEG_TAG}_cv5_wide_classification.json"))["curves"]; fr = np.array(cv["fraction"])
        for lab, c in (("sample cov.", GRAY), ("OAS", AQUA), ("LW-NLS", YELLOW), ("SC-SI", BLUE)):
            a2.plot(fr, cv[lab], color=c, lw=1.9 if lab == "SC-SI" else 1.3)
        a2.set_xlabel("fraction kept (confident first)"); a2.set_ylabel("accuracy on kept"); a2.set_title("(c) Open vs closed, CV", loc="left"); a2.set_xlim(0.2, 1.0)
        h = [plt.Line2D([], [], color=c, lw=3.5) for (_, c) in M.values()]
        fig.legend(h, [v[0] for v in M.values()], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.4)
        save(fig, "fig_eeg", fmt="png")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig_eeg"]
    for n in todo:
        globals()[n]()