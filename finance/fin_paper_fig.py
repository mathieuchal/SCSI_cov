"""Finance figure of the paper -> figs/fig_finance.png (7.4 in wide, 300 dpi).

  (a) min-variance portfolio: mean log(realised variance / OAS) with month-block bootstrap 95% CI, and annualised
      volatility; OAS and LW-NLS use the sample size n tuned on each fold's validation pairs (fin_shrink_n_tune.py),
      the tuned OAS is the reference
  (b) coverage of the 50/80/95% central predictive intervals of forward variances, 12-month rolling
  (c) tempered (N_forward) matrix log-score gain over the sample covariance (nats / 21-day window)
  (d) daily-return log-score gain over the sample covariance (nats / day): Gaussian, and Student-t with nu fitted per fold

Inputs: results/{RUN}_results.json and _arrays.npz (fin_exp.py --select tscore), results/{RUN}_shrinkn_tuned.npz
(fin_shrink_n_tune.py). Nothing is recomputed except the bootstrap intervals of panel (a).

    python fin_paper_fig.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "log-space", "toys"))            # shared code: plotstyle, covutils
from plotstyle import *
from covutils import bootstrap_ci

setup()
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6.2, "pdf.fonttype": 42, "axes.titlelocation": "left", "lines.linewidth": 1.4})
RUN = "fin_d12_nw63_s0_mlp_w1024_d4_neff03_nfe6_o30_selT"     # MLP-4 w1024, 30 EM iters; N=19 / N_forward=6 (coverage calibration, s=0.3);
                                                               # checkpoint per fold = max validation Student-t daily log-score (10/4/6)
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figs", "fig_finance.png")
LIGHT_BLUE = "#79aee8"
LEVELS = (0.5, 0.8, 0.95)


def fig_finance():
    res = json.load(open(f"{RES}/{RUN}_results.json")); z = np.load(f"{RES}/{RUN}_arrays.npz", allow_pickle=True)
    dates = z["dates"].astype("datetime64[D]"); meta, fold = z["meta"], z["fold"]; end = meta[:, 1]
    ue = np.unique(end); mi = np.searchsorted(ue, end); t = dates[ue]; nm_ = len(ue); cnt = np.bincount(mi, minlength=nm_)
    bym = lambda x: np.bincount(mi, weights=x, minlength=nm_) / cnt
    def roll(x, w=12):
        k = np.ones(w); return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")
    refit = [t[np.searchsorted(ue, end[fold == f].min())] for f in (1, 2)]
    crises = [("dot-com", "2000-03-01", "2002-10-01"), ("GFC", "2007-12-01", "2009-06-30"), ("COVID", "2020-02-15", "2020-05-01")]
    pit = {n: z["pit_" + n] for n in ("SC-SI", "IW", "OAS_shrinkage", "SCM")}
    levels = LEVELS
    inside = lambda p, lv: ((p > (1 - lv) / 2) & (p < 1 - (1 - lv) / 2))     # central interval of the predictive law
    cov = {lv: {n: roll(bym(inside(p, lv).mean(1))) for n, p in pit.items()} for lv in levels}
    for lv in levels:
        print(f"    overall {lv:.0%} coverage: " + "  ".join(f"{n} {inside(p, lv).mean():.3f}" for n, p in pit.items()))
    meths = ("SC-SI", "IW", "OAS_shrinkage", "SCM", "LW_nonlinear_shrinkage")
    ll = {n: z["ll_" + n] for n in meths}                                             # tempered (N_forward) matrix log-score, nats / window
    gain = {n: roll(bym(ll[n] - ll["SCM"])) for n in meths[:3] + meths[4:]}
    daily = True
    dl = {n: z["dl_" + n] for n in meths}                                             # Gaussian log-score of the forward daily returns, nats / day
    dgain = {n: roll(bym(dl[n] - dl["SCM"])) for n in meths[:3] + meths[4:]}
    tgain = {n: roll(bym(z["dt_" + n] - z["dt_SCM"])) for n in meths[:3] + meths[4:]}  # Student-t daily log-score, nu per fold
    nu_lab = "/".join(f"{float(v):g}" for v in z["nu_hat"])
    # panel (a): per-case realised variances; OAS and LW use the sample size n tuned on each fold's validation pairs
    V = {n: z[f"var_{i}"] for i, n in enumerate(res["mv"])}
    tz = np.load(f"{RES}/{RUN}_shrinkn_tuned.npz")
    V["OAS shrinkage"], V["LW nonlinear shrinkage"] = tz["oas_tuned"], tz["lw_tuned"]
    shr_lab = "tuned n"
    mv = {n: dict(log_var_ratio_vs_OAS=bootstrap_ci(np.log(v / V["OAS shrinkage"]), end), ann_vol=np.sqrt(252 * v.mean())) for n, v in V.items()}
    for n in ("SC-SI Stein-opt.", "SC-SI post. mean", "LW nonlinear shrinkage", "IW post. mean"):
        print(f"    {n:24s} {mv[n]['log_var_ratio_vs_OAS'][0]:+.4f}  vol {mv[n]['ann_vol']:.2f}%")
    order = ["SCM", "LW nonlinear shrinkage", "PCA K=GD (plug-in)", "OAS shrinkage", "IW post. mean", "SC-SI Stein-opt.", "SC-SI post. mean"]
    label = {"SCM": "sample cov.", "LW nonlinear shrinkage": f"LW-NLS, {shr_lab}", "PCA K=GD (plug-in)": "PCA, GD rule", "OAS shrinkage": f"OAS, {shr_lab} (ref.)",
             "IW post. mean": "IW post. mean", "SC-SI Stein-opt.": "SC-SI Stein-opt.", "SC-SI post. mean": "SC-SI post. mean"}
    col = {"SCM": GRAY, "LW nonlinear shrinkage": YELLOW, "PCA K=GD (plug-in)": GRAY, "OAS shrinkage": AQUA, "IW post. mean": ORANGE,
           "SC-SI Stein-opt.": LIGHT_BLUE, "SC-SI post. mean": BLUE}
    rc = {"font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "normal", "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5}
    with plt.rc_context(rc):
        nb = len(levels); hb = 1.0 if nb == 1 else 1.9                         # panel (b) taller when it stacks several levels
        if daily:
            fig = plt.figure(figsize=(7.4, 3.3 + (0.7 if nb > 1 else 0)))
            hd = 1.0 if tgain is None else 1.7                                  # panel (d) taller with the Student-t strip
            H = 2.55 + (0.45 if nb > 1 else 0) + (0.35 if tgain is not None else 0)   # compact vertically
            fig = plt.figure(figsize=(7.4, H))
            gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.75], height_ratios=[hb, 1, hd], hspace=0.48, wspace=0.34, left=0.135, right=0.995,
                                  top=1 - 0.2 / H, bottom=0.78 / H)
        else:
            fig = plt.figure(figsize=(7.4, 2.6 + (0.7 if nb > 1 else 0)))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.75], height_ratios=[hb, 1], hspace=0.55, wspace=0.34, left=0.135, right=0.995, top=0.915, bottom=0.30 - (0.04 if nb > 1 else 0))
        a0 = fig.add_subplot(gs[:, 0])
        sb = gs[0, 1].subgridspec(nb, 1, hspace=0.08)
        a1s = [fig.add_subplot(sb[0])]
        a1s += [fig.add_subplot(sb[i], sharex=a1s[0]) for i in range(1, nb)]
        a1 = a1s[0]; a2 = fig.add_subplot(gs[1, 1], sharex=a1)
        a3 = fig.add_subplot(gs[2, 1], sharex=a1) if daily else a2
        a4 = None
        if tgain is not None:                                                    # (d) split: Gaussian strip on top, Student-t below
            a3.remove(); sd = gs[2, 1].subgridspec(2, 1, hspace=0.08)
            a3 = fig.add_subplot(sd[0], sharex=a1); a4 = fig.add_subplot(sd[1], sharex=a1)
        ys = np.arange(len(order))[::-1]
        for y, n in zip(ys, order):
            m, lo, hi_ = mv[n]["log_var_ratio_vs_OAS"]
            a0.plot([lo, hi_], [y, y], color=col[n], lw=2.0, solid_capstyle="round"); a0.plot(m, y, "o", color=col[n], ms=5, mec=SURFACE, mew=0.8)
        a0.axvline(0, color=AQUA, lw=1.3)
        halo = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]                # hides the OAS reference line behind the text instead of striking through it
        for y, n in zip(ys, order):                                              # annualised realised volatility, next to each interval (never left of the x=0 OAS line, which would strike through the text)
            a0.text(max(mv[n]["log_var_ratio_vs_OAS"][2] + 0.006, 0.009), y, f"{mv[n]['ann_vol']:.2f}%", ha="left", va="center", fontsize=7.4, color=INK2, path_effects=halo)
        a0.set_xlim(-0.055, 0.205); a0.set_ylim(-0.6, len(order) - 0.4); a0.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
        a0.set_yticks(ys); a0.set_yticklabels([label[n] for n in order], fontsize=7.4); a0.grid(axis="y", visible=False)
        a0.set_xlabel("mean log(realised var. / OAS)"); a0.set_title("(a) Min-variance portfolio", loc="center")
        for a in {*a1s, a2, a3} | ({a4} if a4 is not None else set()):
            for _, d0, d1 in crises:
                a.axvspan(np.datetime64(d0), np.datetime64(d1), color="#dcdcda", alpha=0.7, lw=0, zorder=0)
            for r in refit:
                a.axvline(r, color=INK3, lw=0.8, ls=(0, (2, 3)))
        YL = {0.5: ((0.15, 0.85), [0.3, 0.5, 0.7]), 0.8: ((0.45, 1.06), [0.6, 0.8, 1.0]), 0.95: ((0.7, 1.03), [0.8, 0.9, 1.0])}
        for a, lv in zip(a1s, levels):
            for n, c, lw in (("SCM", GRAY, 0.7), ("OAS_shrinkage", AQUA, 0.7), ("IW", ORANGE, 0.8), ("SC-SI", BLUE, 1.25)):
                a.plot(t, cov[lv][n], color=c, lw=lw)
            a.axhline(lv, color=INK, lw=0.9, ls=(0, (4, 2)))
            (ylo, yhi), yt = YL.get(lv, ((0.0, 1.05), [0, 0.5, 1]))
            a.set_ylim(ylo, yhi); a.set_yticks(yt)
            if nb > 1:
                a.set_yticks([lv]); a.set_yticklabels([f"{lv:.0%}"])          # one tick: the nominal level of this strip
            plt.setp(a.get_xticklabels(), visible=False)
        (ylo, yhi), _ = YL.get(levels[0], ((0.0, 1.05), None))
        for lab, d0, d1 in crises:
            a1.text(np.datetime64(d0) + (np.datetime64(d1) - np.datetime64(d0)) // 2, yhi - 0.015 * (yhi - ylo), lab, ha="center", va="top", fontsize=6.6, color=INK2)
        a1s[nb // 2].set_ylabel("coverage")
        a1.set_title("(b) Interval coverage of forward variances (" + ", ".join(f"{lv:.0%}" for lv in levels) + ")" if nb > 1
                     else f"(b) {levels[0]:.0%} interval coverage of forward variances", loc="center")
        for n, c, lw in (("LW_nonlinear_shrinkage", YELLOW, 0.7), ("OAS_shrinkage", AQUA, 0.7), ("IW", ORANGE, 0.8), ("SC-SI", BLUE, 1.25)):
            a2.plot(t, gain[n], color=c, lw=lw)
        a2.axhline(0, color=GRAY, lw=0.8); a2.set_ylabel("nats"); a2.set_title("(c) Tempered log-score gain vs sample cov. (nats / window)", loc="center")
        if daily:
            plt.setp(a2.get_xticklabels(), visible=False)
            for n, c, lw in (("LW_nonlinear_shrinkage", YELLOW, 0.7), ("OAS_shrinkage", AQUA, 0.7), ("IW", ORANGE, 0.8), ("SC-SI", BLUE, 1.25)):
                a3.plot(t, dgain[n], color=c, lw=lw)
            a3.axhline(0, color=GRAY, lw=0.8); a3.set_ylabel("nats / day"); a3.set_title("(d) Daily-return log-score gain vs sample cov. (nats / day)", loc="center")
            if a4 is not None:
                plt.setp(a3.get_xticklabels(), visible=False)
                for n, c, lw in (("LW_nonlinear_shrinkage", YELLOW, 0.7), ("OAS_shrinkage", AQUA, 0.7), ("IW", ORANGE, 0.8), ("SC-SI", BLUE, 1.25)):
                    a4.plot(t, tgain[n], color=c, lw=lw)
                a4.axhline(0, color=GRAY, lw=0.8)
                tmax = max(np.nanmax(v) for v in tgain.values()); a4.set_ylim(min(0, min(np.nanmin(v) for v in tgain.values())) - 0.05, tmax * 1.45)   # headroom for the strip label
                a4.set_yticks([0, 0.5])                                           # no top tick: it would collide with the strip above
                for a, lab in ((a3, "Gaussian"), (a4, f"Student-t, ν={nu_lab}")):
                    a.text(0.005, 0.93, lab, transform=a.transAxes, ha="left", va="top", fontsize=7, color=INK2,
                           path_effects=[pe.withStroke(linewidth=2.4, foreground=SURFACE)])
                a3.set_ylabel(""); a4.set_ylabel(""); fig.text(a3.get_position().x0 - 0.05, (a3.get_position().y1 + a4.get_position().y0) / 2, "nats / day", rotation=90, ha="center", va="center", fontsize=8.5, color=INK2)
                a3 = a4
        a3.set_xlim(t[0] - np.timedelta64(20, "D"), t[-1] + np.timedelta64(20, "D")); a3.xaxis.set_major_locator(mdates.YearLocator(5)); a3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        h = [plt.Line2D([], [], color=c, lw=3.5) for c in (GRAY, AQUA, YELLOW, ORANGE, BLUE)]
        fig.legend(h, ["sample cov.", "OAS", "LW-NLS", "IW conj", "SC-SI"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.4)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        fig.savefig(OUT, dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote", OUT)


if __name__ == "__main__":
    fig_finance()
