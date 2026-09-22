import json, sys
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import *

setup()
TAG = sys.argv[1] if len(sys.argv) > 1 else "eeg_w4_s0_k1.0_e0.65"
nw = json.load(open(f"results/{TAG}_results.json"))            # next-window protocol (drift included)
sh = json.load(open(f"results/{TAG}_splithalf.json"))          # split-half protocol (common C)
arr = np.load(f"results/{TAG}_arrays.npz", allow_pickle=True)
methods = ["Sample cov.", "Nonlinear shrinkage (LW)", "Linear shrinkage (OAS)", "IW conjugate (ML-fitted)", "SC-SI (ours)"]
short = {"Sample cov.": "Sample cov.", "Nonlinear shrinkage (LW)": "LW nonlinear shrinkage", "Linear shrinkage (OAS)": "OAS shrinkage", "IW conjugate (ML-fitted)": "IW conjugate", "SC-SI (ours)": "SC-SI (ours)"}

fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.1), gridspec_kw={"width_ratios": [0.9, 1.05, 1.5]})
# (a) directional variance coverage, split-half
xs = np.arange(3); wb = 0.16
for j, m in enumerate(methods):
    vals = list(sh["dir_cov"][m].values())
    ax[0].bar(xs + (j - 2) * wb, vals, wb - 0.02, color=COL[m], label=short[m])
for i, lvl in enumerate((0.5, 0.8, 0.95)):
    ax[0].hlines(lvl, i - 0.42, i + 0.42, color=INK, lw=1.5)
ax[0].set_xticks(xs); ax[0].set_xticklabels(["50%", "80%", "95%"]); ax[0].set_ylim(0, 1.3)
ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[0].set_xlabel("nominal level (black line = ideal)"); ax[0].set_ylabel("empirical coverage")
ax[0].set_title("Directional variances, split-half"); ax[0].legend(fontsize=7, loc="upper left", ncol=2, columnspacing=1.0); ax[0].grid(axis="x", visible=False)

# (b) log-score gains: split-half (solid) vs next window (hollow)
names_b = ["Linear shrinkage (OAS)", "Nonlinear shrinkage (LW)", "IW conjugate (ML-fitted)", "SC-SI (ours)"]
ys = np.arange(len(names_b))[::-1] * 1.0
for y, n in zip(ys, names_b):
    g, lo, hi = sh["logscore"][n]["gain"]
    ax[1].plot([lo, hi], [y + 0.13] * 2, color=COL[n], lw=2.2, solid_capstyle="round")
    ax[1].plot(g, y + 0.13, "o", color=COL[n], ms=8, mec=SURFACE, mew=1.5)
    r = nw["methods"][n]; g2, lo2, hi2 = r["gain_vs_scm"], *r["gain_ci"]
    ax[1].plot([lo2, hi2], [y - 0.13] * 2, color=COL[n], lw=2.2, alpha=0.5, solid_capstyle="round")
    ax[1].plot(g2, y - 0.13, "o", mfc=SURFACE, mec=COL[n], mew=2, ms=8)
    ax[1].text(max(hi, hi2) + 0.5, y, f"{g:+.1f} / {g2:+.1f}", va="center", fontsize=8, color=INK2)
ax[1].axvline(0, color=INK3, lw=1)
ax[1].set_yticks(ys); ax[1].set_yticklabels([short[n] for n in names_b], fontsize=8.5); ax[1].grid(axis="y", visible=False)
ax[1].set_xlabel("predictive log-score gain vs sample cov. (nats / window)")
ax[1].set_title("Log-score gain (95% CI over subjects)")
ax[1].plot([], [], "o", color=INK2, label="split-half"); ax[1].plot([], [], "o", mfc=SURFACE, mec=INK2, mew=2, label="next window")
ax[1].legend(fontsize=7.6, loc="upper right"); ax[1].set_xlim(-24, 38)

# (c) 80% coverage for functionals of C
fun = ["log det C", "log C[Oz,Oz]", "top-eigenvalue share", "log cond. number"]
vals = {m: [sh["functional"][f][m]["cov"]["0.8"] for f in fun] + [sh["distance"][m]["cov"]["0.8"]] for m in methods}
labels = ["log det C", "log power\nat Oz", "top-eigenvalue\nshare", "log cond.\nnumber", "open-vs-closed\nRiemannian dist."]
xs = np.arange(len(labels)); wb = 0.16
for j, m in enumerate(methods):
    ax[2].bar(xs + (j - 2) * wb, vals[m], wb - 0.02, color=COL[m], label=short[m])
ax[2].axhline(0.8, color=INK, lw=1.5)
ax[2].text(len(labels) - 0.55, 0.815, "nominal 80%", fontsize=8, color=INK2, ha="right")
ax[2].set_xticks(xs); ax[2].set_xticklabels(labels, fontsize=8); ax[2].set_ylim(0, 1.02)
ax[2].set_ylabel("coverage of central 80% predictive interval"); ax[2].set_title("Nonlinear functionals of C, split-half")
ax[2].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"figs/{TAG}_E1_calibration.png"); plt.close(fig)

# ------------------------------------------------------------------ decision with abstention
prob = arr["prob"]; pn = list(arr["prob_names"]); y = arr["te_y"]
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9))
cols = {n: (GRAY if n.startswith("SCM ->") else AQUA if n.startswith("OAS") else YELLOW if n.startswith("NLS") else "#79aee8" if n.startswith("SCM-trained") else BLUE) for n in pn}
fr = np.linspace(0.2, 1.0, 33)
for i, n in enumerate(pn):
    p = prob[i]; order = np.argsort(-np.abs(p - 0.5))
    acc = []
    for f in fr:
        k = max(int(f * len(p)), 1); acc.append(((p[order[:k]] > 0.5).astype(int) == y[order[:k]]).mean())
    ax[0].plot(fr, acc, color=cols[n], lw=2.0, label=n)
ax[0].set_xlabel("fraction of windows kept (most confident first)"); ax[0].set_ylabel("accuracy on kept windows")
ax[0].set_title("Selective classification, eyes open vs closed"); ax[0].legend(fontsize=7.4, loc="lower left")
cl = nw["classif"]; xs = np.arange(len(pn))
ax[1].bar(xs, [cl[n]["nll"][0] for n in pn], color=[cols[n] for n in pn], width=0.55)
for xx, n in zip(xs, pn):
    lo, hi = cl[n]["nll"][1], cl[n]["nll"][2]
    ax[1].plot([xx, xx], [lo, hi], color=INK2, lw=1.2)
    ax[1].text(xx, 0.02, f"acc {cl[n]['acc'][0]:.3f}", ha="center", fontsize=8, color="white", rotation=90, va="bottom")
ax[1].set_xticks(xs); ax[1].set_xticklabels(["SCM\nplug-in", "OAS\nplug-in", "LW-NLS\nplug-in", "SCM-LR,\nposterior avg.", "post.-LR,\nposterior avg."], fontsize=7.5)
ax[1].set_ylabel("negative log-likelihood (lower is better)"); ax[1].set_title("Bars: NLL with 95% CI over subjects")
ax[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"figs/{TAG}_E3_decision.png"); plt.close(fig)
print("figures written for", TAG)
