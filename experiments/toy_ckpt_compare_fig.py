"""One-off diagnostic: overlay the old (log-score-selected) and new (worst-case-PIT-KS-selected) SC-SI
checkpoints against the reference posterior, for a handful of test observations, to visualize the ldv_bot/
top_share/ldv_top trade-off from switching checkpoint-selection criteria (see toy_train.py, covutils.
worst_case_pit_ks). Both checkpoints come from the same trained run (results/toys/{toy}_k{K}.pt), so this
isolates the effect of which outer iteration was picked, not a different random training realization.

Usage: python toy_ckpt_compare_fig.py <toy> <k_old> <k_new> <obs1> [<obs2> ...]"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import *
from toy_eval import STATS, LABELS

setup()
toy, k_old, k_new = sys.argv[1], sys.argv[2], sys.argv[3]
obs_list = [int(x) for x in sys.argv[4:]] or [0]

D_new = np.load(f"results/toys/{toy}_stats.npz")               # canonical = current best (new criterion)
D_old = np.load(f"results/toys/{toy}_stats_k{k_old}.npz")
SHORT = {"logdet": "log det C", "logcond": "log cond. number", "top_share": "top-eig. share",
         "ldv_top": "log dir. var.\n(top emp. eigvec)", "ldv_bot": "log dir. var.\n(bottom emp. eigvec)", "corr01": "correlation C01"}

nO, nS = len(obs_list), len(STATS)
fig, axs = plt.subplots(nO, nS, figsize=(2.55 * nS, 2.15 * nO), squeeze=False)
for i, OBS in enumerate(obs_list):
    for j, s in enumerate(STATS):
        ax = axs[i, j]
        ref = D_new[f"ref__{s}"][OBS]
        old, new = D_old[f"scsi__{s}"][OBS], D_new[f"scsi__{s}"][OBS]
        tv = D_new[f"truth__{s}"][OBS]
        allv = np.concatenate([ref, old, new, [tv]]); lo, hi = np.percentile(allv, [0.3, 99.7])
        lo, hi = min(lo, tv), max(hi, tv); pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
        bins = np.linspace(lo, hi, 36)
        ax.hist(ref, bins=bins, density=True, color=GRAY, alpha=0.40, lw=0, label="reference posterior")
        ax.hist(old, bins=bins, density=True, histtype="step", color=INK3, lw=1.6, ls=(0, (3, 1.5)), label=f"SC-SI old (k={k_old}, log-score)")
        ax.hist(new, bins=bins, density=True, histtype="step", color=BLUE, lw=1.9, label=f"SC-SI new (k={k_new}, worst-case PIT-KS)")
        ax.axvline(tv, color=INK, lw=1.4, ls=(0, (4, 2)), label="true value")
        ax.set_xlim(lo, hi); ax.set_yticks([]); ax.grid(axis="y", visible=False)
        if i == 0: ax.set_title(SHORT[s], fontsize=9)
        if j == 0: ax.set_ylabel(f"obs #{OBS}", fontsize=9, rotation=0, ha="right", va="center", labelpad=28)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"{toy}: old vs new checkpoint-selection criterion, same trained run", fontsize=10, y=1.0)
fig.tight_layout(rect=(0, 0.045, 1, 0.98))
fig.savefig(f"figs/{toy}_ckpt_compare_k{k_old}_vs_k{k_new}.png")
print(f"wrote figs/{toy}_ckpt_compare_k{k_old}_vs_k{k_new}.png")
