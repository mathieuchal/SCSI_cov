"""Selective classification, eyes open vs closed (panel (c) of paper_figs.fig_eeg, on its own).

5-fold subject-disjoint CV over all 109 subjects; curves = accuracy on the fraction of windows kept, most confident
first (|p - 0.5|), as written by eeg_cv_classif.py.  One figure per window length.

Usage: python figs_classif.py [win ...]        win in seconds, default 4      e.g.  python figs_classif.py 4 2
Writes paper/fig_real/fig_classif_w{win}.png
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import *

setup()
OUT = "paper/fig_real"
os.makedirs(OUT, exist_ok=True)
RUNS = {4: "results/eeg_w4_s0_k1.0_e0.65_cv5_wide_classification.json",            # MLP 512, n_eff 0.65, log-score checkpoint
        2: "results/eeg_w2_s0_k1.0_e0.65_rv_cv5_wide_classification.json",         # same settings, 2-s windows (N = 14)
        8: "results/eeg_w8_s0_k1.0_e0.65_rv_cv5_wide_classification.json"}         # same settings, 8-s windows (N = 56)
METHODS = (("sample cov.", GRAY), ("OAS", AQUA), ("LW-NLS", YELLOW), ("SC-SI", BLUE))


def fig_classif(win):
    z = json.load(open(RUNS[win])); cv = z["curves"]; fr = np.array(cv["fraction"])
    rc = {"font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "normal", "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        for lab, c in METHODS:
            ax.plot(fr, cv[lab], color=c, lw=1.9 if lab == "SC-SI" else 1.3, label=f"{lab} ({z['pooled'][lab]['acc']:.3f})")
        ax.set_xlim(0.2, 1.0); ax.set_xlabel("fraction kept (confident first)"); ax.set_ylabel("accuracy on kept")
        ax.set_title(f"Open vs closed, CV, {win} s windows", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, handlelength=1.4, columnspacing=1.2,
                  title="method (accuracy on all windows)", title_fontsize=7.5)
        fig.tight_layout()
        path = f"{OUT}/fig_classif_w{win}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote", path)


if __name__ == "__main__":
    for w in [int(a) for a in sys.argv[1:]] or [4]:
        fig_classif(w)
