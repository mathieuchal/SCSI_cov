"""Camera-ready figures for the experiments section (vector PDF, sized for one text-width column) -> ../paper/fig_real/.

  fig_toy_logscore : validation log-score gap to the oracle vs EM iteration, three d=8 toys
  fig_toy_stats    : posterior of six statistics at a *median-error* test task, three d=8 toys
  fig_eeg          : (a) log-score gain, (b) 80% coverage per metric, (c) selective classification, cross-validated over all subjects   [PNG, 7.4 in wide]
  fig_finance      : (a) min-variance ratio vs OAS, (b) calibration through time, (c) log-score gain through time   [PNG, 7.4 in wide]

All inputs are committed result files (results/*.json) plus the per-case arrays (gitignored) they were derived from.
Usage: python paper_figs.py [name ...]   (no argument = all)
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from plotstyle import *
from toy_eval import STATS, w1_norm

setup()
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6.2, "pdf.fonttype": 42, "axes.titlelocation": "left", "lines.linewidth": 1.4})
PDF, PNG = "../paper/fig_real", "figs/paper"
os.makedirs(PDF, exist_ok=True); os.makedirs(PNG, exist_ok=True)
EEG_TAG = "eeg_w4_s0_k1.0_e0.65"
FIN_TAG = "fin_d12_nw63_s0_wcsel_wide_k30_ks6-2-6"
HI = "#eb6834"
LIGHT_BLUE, LIGHT_ORANGE = "#79aee8", "#f2a88e"


def save(fig, name, fmt="pdf"):
    if fmt == "png":                                                             # raster-only figure, shipped as PNG (no PDF, no preview copy)
        fig.savefig(f"{PDF}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote", f"{PDF}/{name}.png"); return
    fig.savefig(f"{PDF}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{PNG}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", f"{PDF}/{name}.pdf")


# ------------------------------------------------------------------------------------------------ toy: log-score vs EM iteration
def fig_toy_logscore():
    toys = [("iw_ri_d8", "IW, RI  ($d{=}8$)"), ("iw_nonri_d8", "IW, non-RI  ($d{=}8$)"), ("factor_d8", "Factor model  ($d{=}8$)")]
    fig, axs = plt.subplots(1, 3, figsize=(5.6, 2.05))
    for ax, (t, title) in zip(axs, toys):
        tr = json.load(open(f"results/toys/{t}_train.json")); o = json.load(open(f"results/toys/{t}_oracle_logscore.json"))
        ls = {int(k): v for k, v in tr["val_curve"].items()}; ks = sorted(ls)
        orc = o["oracle_J64"]; gap = np.array([ls[k] - orc for k in ks]); iw = o["iw_ml_J64"] - orc
        ax.axhline(0, color=INK, lw=1.3)
        ax.axhline(iw, color=ORANGE, lw=1.4, ls=(0, (4, 2)))
        ax.plot(ks, gap, "-o", color=BLUE, ms=2.6)
        ksel, kls = tr["best_k"], max(ls, key=ls.get)
        if kls != ksel:
            ax.plot([kls], [ls[kls] - orc], "D", ms=4.6, mfc="none", mec=INK3, mew=1.0)
        ax.plot([ksel], [ls[ksel] - orc], "o", ms=7, mfc="none", mec=INK, mew=1.2)
        lo = min(gap.min(), iw); ax.set_ylim(lo - 0.10 * abs(lo), 0.16 * abs(lo))
        ax.set_xlim(-0.8, 24.8); ax.set_xticks([0, 4, 8, 12, 16, 20, 24]); ax.set_xlabel("EM iteration $k$")
        ax.set_title(title)
        if ax is axs[0]:
            ax.set_ylabel("val. log-score $-$ oracle\n(nats / task)")
    h = [plt.Line2D([], [], color=BLUE, marker="o", ms=3.5), plt.Line2D([], [], color=ORANGE, ls=(0, (4, 2))), plt.Line2D([], [], color=INK, lw=1.3),
         plt.Line2D([], [], color=INK, marker="o", mfc="none", mew=1.2, ms=6.5, ls="none"), plt.Line2D([], [], color=INK3, marker="D", mfc="none", mew=1.0, ms=4.6, ls="none")]
    fig.legend(h, ["SC-SI", "IW conjugate (ML-fitted)", "oracle posterior", "checkpoint selected (worst-case PIT-KS)", "log-score maximiser (if different)"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.0, 1, 1)); save(fig, "fig_toy_logscore")


# ------------------------------------------------------------------------------------------------ toy: posterior statistics
def fig_toy_stats():
    toys = [("iw_ri_d8", "IW, RI\n$d{=}8,\\ N{=}40$"), ("iw_nonri_d8", "IW, non-RI\n$d{=}8,\\ N{=}28$"), ("factor_d8", "Factor model\n$d{=}8,\\ N{=}28$")]
    titles = ["log det $C$", "log cond.\nnumber", "top-eig.\nshare", "log dir. var.\n(top eigvec)", "log dir. var.\n(bottom eigvec)", "corr. $C_{01}$"]
    fig, axs = plt.subplots(3, 6, figsize=(5.6, 3.3))
    for i, (t, lab) in enumerate(toys):
        D = np.load(f"results/toys/{t}_stats.npz")
        w = np.mean([w1_norm(D[f"scsi__{s}"], D[f"ref__{s}"]) for s in STATS], axis=0)
        obs = int(np.argsort(w)[len(w) // 2])                                    # the median-error task: neither best nor worst
        for j, s in enumerate(STATS):
            ax = axs[i, j]
            ref, si, iw, tv = D[f"ref__{s}"][obs], D[f"scsi__{s}"][obs], D[f"iwfit__{s}"][obs], D[f"truth__{s}"][obs]
            lo, hi = np.percentile(np.concatenate([ref, si, iw]), [0.3, 99.7]); lo, hi = min(lo, tv), max(hi, tv); pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
            bins = np.linspace(lo, hi, 32)
            ax.hist(ref, bins=bins, density=True, color=GRAY, alpha=0.40, lw=0)
            ax.hist(iw, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.2)
            ax.hist(si, bins=bins, density=True, histtype="step", color=BLUE, lw=1.5)
            ax.axvline(tv, color=INK, lw=1.1, ls=(0, (4, 2)))
            ax.set_xlim(lo, hi); ax.set_yticks([]); ax.grid(axis="y", visible=False); ax.tick_params(axis="x", labelsize=5.3, pad=1.2)
            if i == 0:
                ax.set_title(titles[j], fontsize=6, loc="center", fontweight="normal")
            if j == 0:
                ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=20, fontsize=6.4)
    h = [plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.4), plt.Line2D([], [], color=BLUE, lw=1.5), plt.Line2D([], [], color=ORANGE, lw=1.2), plt.Line2D([], [], color=INK, ls=(0, (4, 2)))]
    fig.legend(h, ["oracle posterior", "SC-SI", "IW conjugate (ML-fitted)", "true value"], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.045), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=0.6, w_pad=0.3); save(fig, "fig_toy_stats")


# ------------------------------------------------------------------------------------------------ EEG
def fig_eeg():
    """Wider than the other figures (7.4 in) so that the three panels are individually readable; PNG output."""
    nw = json.load(open(f"results/{EEG_TAG}_wide_k30_k8_results.json")); sh = json.load(open(f"results/{EEG_TAG}_wide_k30_splithalf_k8.json"))
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


# ------------------------------------------------------------------------------------------------ finance
def fig_finance():
    """7.4 in wide PNG with the same type scale as fig_eeg; crisis periods shaded in a single light gray; shared legend below."""
    res = json.load(open(f"results/{FIN_TAG}_results.json")); z = np.load(f"results/{FIN_TAG}_arrays.npz", allow_pickle=True)
    dates = z["dates"].astype("datetime64[D]"); meta, fold = z["meta"], z["fold"]; end = meta[:, 1]
    ue = np.unique(end); mi = np.searchsorted(ue, end); t = dates[ue]; nm_ = len(ue); cnt = np.bincount(mi, minlength=nm_)
    bym = lambda x: np.bincount(mi, weights=x, minlength=nm_) / cnt
    def roll(x, w=12):
        k = np.ones(w); return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")
    refit = [t[np.searchsorted(ue, end[fold == f].min())] for f in (1, 2)]
    crises = [("dot-com", "2000-03-01", "2002-10-01"), ("GFC", "2007-12-01", "2009-06-30"), ("COVID", "2020-02-15", "2020-05-01")]
    pit = {n: z["pit_" + n] for n in ("SC-SI", "IW", "OAS_shrinkage", "SCM")}
    cov = {n: roll(bym(((p > 0.1) & (p < 0.9)).mean(1))) for n, p in pit.items()}
    ll = {n: z["ll_" + n] for n in ("SC-SI", "IW", "OAS_shrinkage", "SCM", "LW_nonlinear_shrinkage")}
    gain = {n: roll(bym(ll[n] - ll["SCM"])) for n in ("SC-SI", "IW", "OAS_shrinkage", "LW_nonlinear_shrinkage")}
    order = ["SCM", "LW nonlinear shrinkage", "PCA K=GD (plug-in)", "OAS shrinkage", "IW post. mean", "SC-SI Stein-opt.", "SC-SI post. mean"]
    label = {"SCM": "sample cov.", "LW nonlinear shrinkage": "LW-NLS", "PCA K=GD (plug-in)": "PCA, GD rule", "OAS shrinkage": "OAS (ref.)",
             "IW post. mean": "IW post. mean", "SC-SI Stein-opt.": "SC-SI Stein-opt.", "SC-SI post. mean": "SC-SI post. mean"}
    col = {"SCM": GRAY, "LW nonlinear shrinkage": YELLOW, "PCA K=GD (plug-in)": GRAY, "OAS shrinkage": AQUA, "IW post. mean": ORANGE,
           "SC-SI Stein-opt.": LIGHT_BLUE, "SC-SI post. mean": BLUE}
    rc = {"font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "normal", "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.4, 2.45))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.75], hspace=0.36, wspace=0.34, left=0.135, right=0.995, top=0.915, bottom=0.33)
        a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)
        ys = np.arange(len(order))[::-1]
        for y, n in zip(ys, order):
            m, lo, hi_ = res["mv"][n]["log_var_ratio_vs_OAS"]
            a0.plot([lo, hi_], [y, y], color=col[n], lw=2.0, solid_capstyle="round"); a0.plot(m, y, "o", color=col[n], ms=5, mec=SURFACE, mew=0.8)
        a0.axvline(0, color=AQUA, lw=1.3)
        halo = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]                # hides the OAS reference line behind the text instead of striking through it
        for y, n in zip(ys, order):                                              # annualised realised volatility, next to each interval (never left of the x=0 OAS line, which would strike through the text)
            a0.text(max(res["mv"][n]["log_var_ratio_vs_OAS"][2] + 0.006, 0.009), y, f"{res['mv'][n]['ann_vol']:.2f}%", ha="left", va="center", fontsize=7.4, color=INK2, path_effects=halo)
        a0.set_xlim(-0.055, 0.205); a0.set_ylim(-0.6, len(order) - 0.4); a0.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
        a0.set_yticks(ys); a0.set_yticklabels([label[n] for n in order], fontsize=7.4); a0.grid(axis="y", visible=False)
        a0.set_xlabel("mean log(realised var. / OAS)"); a0.set_title("(a) Min-variance portfolio", loc="left")
        for a in (a1, a2):
            for _, d0, d1 in crises:
                a.axvspan(np.datetime64(d0), np.datetime64(d1), color="#dcdcda", alpha=0.7, lw=0, zorder=0)
            for r in refit:
                a.axvline(r, color=INK3, lw=0.8, ls=(0, (2, 3)))
        for n, c, lw in (("SCM", GRAY, 0.7), ("OAS_shrinkage", AQUA, 0.7), ("IW", ORANGE, 0.8), ("SC-SI", BLUE, 1.25)):
            a1.plot(t, cov[n], color=c, lw=lw)
        a1.axhline(0.8, color=INK, lw=0.9, ls=(0, (4, 2))); a1.set_ylim(0.45, 1.06); a1.set_yticks([0.6, 0.8, 1.0]); a1.set_ylabel("coverage")
        for lab, d0, d1 in crises:
            a1.text(np.datetime64(d0) + (np.datetime64(d1) - np.datetime64(d0)) // 2, 1.045, lab, ha="center", va="top", fontsize=6.6, color=INK2)
        a1.set_title("(b) 80% interval coverage of forward variances", loc="left"); plt.setp(a1.get_xticklabels(), visible=False)
        for n, c, lw in (("LW_nonlinear_shrinkage", YELLOW, 0.7), ("OAS_shrinkage", AQUA, 0.7), ("IW", ORANGE, 0.8), ("SC-SI", BLUE, 1.25)):
            a2.plot(t, gain[n], color=c, lw=lw)
        a2.axhline(0, color=GRAY, lw=0.8); a2.set_ylabel("nats"); a2.set_title("(c) Forward log-score gain vs sample cov. (nats / window)", loc="left")
        a2.set_xlim(t[0] - np.timedelta64(20, "D"), t[-1] + np.timedelta64(20, "D")); a2.xaxis.set_major_locator(mdates.YearLocator(5)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        h = [plt.Line2D([], [], color=c, lw=3.5) for c in (GRAY, AQUA, YELLOW, ORANGE, BLUE)]
        fig.legend(h, ["sample cov.", "OAS", "LW-NLS", "IW conj", "SC-SI"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.4)
        save(fig, "fig_finance", fmt="png")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig_toy_logscore", "fig_toy_stats", "fig_eeg", "fig_finance"]
    for n in todo:
        globals()[n]()
