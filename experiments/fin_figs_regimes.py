"""F3: how the SC-SI posterior (and the decisions made from it) behave across market regimes.

Needs the per-case arrays saved by fin_exp.py (pit_*, ll_*, pm_post/pm_plug, nm_mean/nm_sd, var_*), so the run must have been
(re)evaluated after those were added.  Regimes are data-driven: terciles of the trailing 63-day realised volatility of an
equal-weight portfolio of all listed industries, measured at the forecast origin (no look-ahead).  All quantities are first
averaged over the 8 baskets of a month (baskets within a month are strongly dependent), so the unit of inference is a month.

Usage: python fin_figs_regimes.py <tag>          e.g. fin_d12_nw63_s0_wcsel_wide_k30_ks6-2-6
"""
import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from plotstyle import *

setup()
TAG = sys.argv[1]
z = np.load(f"results/{TAG}_arrays.npz", allow_pickle=True)
res = json.load(open(f"results/{TAG}_results.json"))
dates = z["dates"].astype("datetime64[D]")
meta, fold = z["meta"], z["fold"]
end = meta[:, 1]
ue = np.unique(end)
mi = np.searchsorted(ue, end)                                           # month index of every case
t = dates[ue]
nm_ = len(ue)
names = list(z["names"])
var = {n: z[f"var_{i}"] for i, n in enumerate(names)}
cnt = np.bincount(mi, minlength=nm_)


def bym(x):                                                             # mean over the baskets of each month
    return np.bincount(mi, weights=x, minlength=nm_) / cnt


def roll(x, w=12):                                                      # centred moving average, edge-normalised
    k = np.ones(w)
    return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")


# ---------------------------------------------------------------- market regime from the raw data (measured at the forecast origin)
df = pd.read_pickle("data/ff48_vw_daily.pkl")
assert (df.index.values.astype("datetime64[D]")[ue] == t).all()
ew = np.nanmean(df.values, axis=1)                                      # equal-weight market, % daily
vol = np.array([np.sqrt(np.mean(ew[e - 62:e + 1] ** 2) * 252) for e in ue])
q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
reg = (vol > q1).astype(int) + (vol > q2).astype(int)                   # 0 low, 1 mid, 2 high
REG = ["low vol", "mid vol", "high vol"]
HI = "#eb6834"

# ---------------------------------------------------------------- per-month series
pit = {n: z["pit_" + n] for n in ("SC-SI", "IW", "OAS_shrinkage", "SCM", "LW_nonlinear_shrinkage")}
cov80 = {n: bym(((p > 0.1) & (p < 0.9)).mean(1)) for n, p in pit.items()}
ll = {n: z["ll_" + n] for n in ("SC-SI", "IW", "OAS_shrinkage", "SCM", "LW_nonlinear_shrinkage")}
gain = {n: bym(ll[n] - ll["SCM"]) for n in ("SC-SI", "IW", "OAS_shrinkage", "LW_nonlinear_shrinkage")}
lvr = {n: bym(np.log(var[n] / var["OAS shrinkage"])) for n in ("SC-SI post. mean", "SC-SI Stein-opt.", "IW post. mean", "LW nonlinear shrinkage", "SCM")}
pm = np.stack([np.bincount(mi, weights=z["pm_post"][:, k], minlength=nm_) / cnt for k in range(6)], 1)      # (months, 6)
nmean, nsd = bym(z["nm_mean"]), bym(z["nm_sd"])
plug = bym(z["pm_plug"].sum(1))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(13.2, 16.6))
gs = fig.add_gridspec(6, 3, height_ratios=[0.95, 1.6, 1.15, 1.15, 1.25, 1.55], hspace=0.42, wspace=0.28)
axs = [fig.add_subplot(gs[i, :]) for i in range(5)]
for a in axs[1:]:
    a.sharex(axs[0])
x0, x1 = t[0] - np.timedelta64(20, "D"), t[-1] + np.timedelta64(20, "D")

# contiguous high-vol spans, shaded in every time panel
runs, i = [], 0
while i < nm_:
    if reg[i] == 2:
        j = i
        while j + 1 < nm_ and reg[j + 1] == 2:
            j += 1
        runs.append((t[i] - np.timedelta64(15, "D"), t[j] + np.timedelta64(15, "D"))); i = j + 1
    else:
        i += 1
refit = [t[np.searchsorted(ue, end[fold == f].min())] for f in (1, 2)]
for a in axs:
    for s0, s1 in runs:
        a.axvspan(s0, s1, color=HI, alpha=0.10, lw=0)
    for r in refit:
        a.axvline(r, color=INK3, lw=1.0, ls=(0, (2, 3)))
    a.set_xlim(x0, x1)
    if a is not axs[-1]:
        plt.setp(a.get_xticklabels(), visible=False)

# (0) market regime
a = axs[0]
a.plot(t, vol, color=INK, lw=1.4)
a.axhline(q1, color=INK3, lw=0.9, ls=(0, (4, 3))); a.axhline(q2, color=INK3, lw=0.9, ls=(0, (4, 3)))
a.set_ylim(3, 88)
a.text(t[1], 62, f"dashed: volatility terciles ({q1:.0f}% / {q2:.0f}%)\nlow / mid / high regime", va="center", fontsize=7.8, color=INK2)
a.set_ylabel("ann. vol (%)"); a.set_title("Market regime: trailing 63-day realised volatility of the equal-weight market  (shaded = top-tercile months, in every panel)")
for lab, d0 in (("dot-com", "2001-03-01"), ("GFC", "2008-10-15"), ("COVID", "2020-03-20")):
    a.text(np.datetime64(d0), 80, lab, fontsize=8, color=INK2, ha="center", va="center")
for r, lab in zip(refit, ("refit (fold 1)", "refit (fold 2)")):
    a.text(r, 86, lab + " ", rotation=90, ha="right", va="top", fontsize=7.5, color=INK3)

# (1) posterior over the number of meaningful factors
a = axs[1]
cm = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#a9c8f0", "#2a78d6", "#0d2b57"])
im = a.imshow(pm[:, 1:5].T, aspect="auto", origin="lower", cmap=cm, vmin=0, vmax=1, interpolation="nearest",
              extent=[mdates.date2num(x0), mdates.date2num(x1), 1, 5])
halo = [pe.Stroke(linewidth=3.6, foreground=SURFACE), pe.Normal()]
a.plot(t, nmean, color=HI, lw=2.0, label="posterior mean number of meaningful factors", path_effects=halo)
a.plot(t, plug, color=INK, lw=1.2, ls=(0, (3, 2)), label="same rule on the sample spectrum (plug-in)", path_effects=halo)
a.set_ylim(1, 5); a.set_yticks(np.arange(1.5, 5)); a.set_yticklabels([f"factor {k}" for k in range(2, 6)]); a.grid(False)
a.set_title("How many factors? P(eigen-direction k is meaningful | data), population eigenvalue > 3 x median  (factor 1 is ~always meaningful: omitted)")
a.text(0.995, 0.03, "posterior level is fold-specific (different model per fold): compare the two lines within a fold, not across the dotted refit lines",
       transform=a.transAxes, ha="right", va="bottom", fontsize=7.6, color=INK, path_effects=[pe.Stroke(linewidth=2.6, foreground=SURFACE), pe.Normal()])
a.legend(loc="upper left", fontsize=8, ncol=2, facecolor=SURFACE, framealpha=0.9, frameon=True, edgecolor="none")
cax = a.inset_axes([1.006, 0.0, 0.008, 1.0]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7)
cb.set_label("P(meaningful)", fontsize=7.5, labelpad=2)

# (2) calibration
a = axs[2]
for n, c, lw, lab in (("SCM", GRAY, 1.3, "sample cov."), ("OAS_shrinkage", AQUA, 1.3, "OAS"), ("IW", ORANGE, 1.5, "IW conjugate"), ("SC-SI", BLUE, 2.2, "SC-SI")):
    a.plot(t, roll(cov80[n]), color=c, lw=lw, label=lab)
a.axhline(0.8, color=INK, lw=1.2, ls=(0, (4, 2)))
a.text(1.004, 0.8, "nominal\n0.8", transform=a.get_yaxis_transform(), va="center", fontsize=8, color=INK)
a.set_ylim(0.45, 1.0); a.set_ylabel("80% coverage")
a.set_title("Calibration through time: share of realised forward variances inside the central 80% interval (12 asset axes + equal weight; 12-month mean)")
a.legend(loc="lower left", ncol=4, fontsize=8)

# (3) predictive log-score
a = axs[3]
for n, c, lw, lab in (("LW_nonlinear_shrinkage", YELLOW, 1.3, "LW nonlinear shrinkage"), ("OAS_shrinkage", AQUA, 1.3, "OAS"), ("IW", ORANGE, 1.5, "IW conjugate"), ("SC-SI", BLUE, 2.2, "SC-SI")):
    a.plot(t, roll(gain[n]), color=c, lw=lw, label=lab)
a.axhline(0, color=GRAY, lw=1.2)
a.text(1.004, 0, "sample\ncov.", transform=a.get_yaxis_transform(), va="center", fontsize=8, color=GRAY)
a.set_ylabel("nats / window"); a.set_title("Posterior quality through time: forward predictive log-score gain over the sample covariance, 12-month centred mean")
a.legend(loc="upper left", ncol=4, fontsize=8)

# (4) decision quality
a = axs[4]
for n, c, lw, lab in (("LW nonlinear shrinkage", YELLOW, 1.3, "LW nonlinear shrinkage"), ("IW post. mean", ORANGE, 1.5, "IW posterior mean"),
                      ("SC-SI Stein-opt.", "#79aee8", 1.6, "SC-SI Stein-optimal"), ("SC-SI post. mean", BLUE, 2.3, "SC-SI posterior mean")):
    a.plot(t, np.cumsum(lvr[n]), color=c, lw=lw, label=lab)
a.axhline(0, color=AQUA, lw=1.6); a.text(1.004, 0, "OAS", transform=a.get_yaxis_transform(), va="center", fontsize=8, color=AQUA)
a.set_ylabel("cumulative log(var / var_OAS)")
a.set_title("Decision quality through time: cumulative log(realised min-variance-portfolio variance / OAS); downward = better than OAS")
a.text(0.005, 0.04, "sample covariance not shown (off-scale: +35 by 2026)", transform=a.transAxes, ha="left", va="bottom", fontsize=7.8, color=GRAY)
a.legend(loc="upper left", ncol=2, fontsize=8)
a.xaxis.set_major_locator(mdates.YearLocator(4)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ---------------------------------------------------------------- by-regime summaries (month-block bootstrap)
rng = np.random.default_rng(0)


def by_regime(series, B=2000):
    out = np.zeros((3, 3))
    for r in range(3):
        v = series[reg == r]
        bs = rng.choice(v, (B, len(v))).mean(1)
        out[r] = [v.mean(), *np.percentile(bs, [2.5, 97.5])]
    return out


def bars(a, spec, title, ylab, hline=None, ylim=None):
    w = 0.8 / len(spec)
    for j, (series, c, lab) in enumerate(spec):
        o = by_regime(series)
        xs = np.arange(3) + (j - (len(spec) - 1) / 2) * w
        a.bar(xs, o[:, 0], w * 0.9, color=c, label=lab)
        a.errorbar(xs, o[:, 0], yerr=[o[:, 0] - o[:, 1], o[:, 2] - o[:, 0]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=2)
    if hline is not None:
        a.axhline(hline, color=INK, lw=1.2, ls=(0, (4, 2)))
    a.set_xticks(range(3)); a.set_xticklabels([f"{REG[r]}\n({(reg == r).sum()} months)" for r in range(3)], fontsize=8.5)
    a.set_title(title, fontsize=9.5); a.set_ylabel(ylab); a.grid(axis="x", visible=False)
    if ylim: a.set_ylim(*ylim)


a1, a2, a3 = (fig.add_subplot(gs[5, k]) for k in range(3))
bars(a1, [(cov80["SCM"], GRAY, "sample cov."), (cov80["OAS_shrinkage"], AQUA, "OAS"), (cov80["IW"], ORANGE, "IW"), (cov80["SC-SI"], BLUE, "SC-SI")],
     "80% coverage by regime", "coverage", hline=0.8, ylim=(0.5, 1.0))
a1.legend(loc="upper center", fontsize=7.5, ncol=4, columnspacing=0.8, handlelength=1.2)
bars(a2, [(gain["LW_nonlinear_shrinkage"], YELLOW, "LW NLS"), (gain["OAS_shrinkage"], AQUA, "OAS"), (gain["IW"], ORANGE, "IW"), (gain["SC-SI"], BLUE, "SC-SI")],
     "Log-score gain vs sample cov.", "nats / window", hline=0, ylim=(-1.6, 10.0))
a2.legend(loc="upper center", fontsize=7.5, ncol=4, columnspacing=0.8, handlelength=1.2)
bars(a3, [(lvr["SCM"], GRAY, "sample cov."), (lvr["LW nonlinear shrinkage"], YELLOW, "LW NLS"), (lvr["IW post. mean"], ORANGE, "IW post. mean"),
          (lvr["SC-SI post. mean"], BLUE, "SC-SI post. mean")], "Realised variance vs OAS (lower = better)", "mean log(var / var_OAS)", hline=0, ylim=(-0.085, 0.29))
a3.legend(loc="upper left", fontsize=7.5, ncol=2, columnspacing=0.8, handlelength=1.2)
fig.suptitle("Finance: posterior quality and decisions across market regimes  (wide net, 800 steps/iteration; checkpoints 6/2/6 by worst-KS excl. ldv_bot)",
             x=0.01, y=0.985, ha="left", fontsize=11.5, fontweight="semibold")
fig.text(0.01, 0.004, "Months averaged over 8 fixed baskets; 21-day forecasts; N_eff = 19 (forward 6). Dotted verticals = model refit. "
         "Bars: month-block bootstrap 95% intervals.\nRegimes = terciles of trailing 63-day equal-weight-market volatility at the forecast origin (no look-ahead).",
         fontsize=7.8, color=INK2, va="bottom")
fig.subplots_adjust(top=0.955, bottom=0.075, left=0.065, right=0.955)
fig.savefig(f"figs/{TAG}_F3_regimes.png")
print(f"saved figs/{TAG}_F3_regimes.png")

# ---------------------------------------------------------------- numbers behind the summary bars
print(f"\nvol terciles at {q1:.1f}% / {q2:.1f}%   months: " + ", ".join(f"{REG[r]}={int((reg == r).sum())}" for r in range(3)))
for lab, ser in (("80% cov SC-SI", cov80["SC-SI"]), ("80% cov IW", cov80["IW"]), ("80% cov OAS", cov80["OAS_shrinkage"]), ("80% cov SCM", cov80["SCM"]),
                 ("logscore gain SC-SI", gain["SC-SI"]), ("logscore gain IW", gain["IW"]), ("logscore gain OAS", gain["OAS_shrinkage"]), ("logscore gain LW", gain["LW_nonlinear_shrinkage"]),
                 ("log var/OAS SC-SI post", lvr["SC-SI post. mean"]), ("log var/OAS SC-SI Stein", lvr["SC-SI Stein-opt."]), ("log var/OAS IW post", lvr["IW post. mean"]),
                 ("log var/OAS LW", lvr["LW nonlinear shrinkage"]), ("log var/OAS SCM", lvr["SCM"])):
    o = by_regime(ser)
    print(f"{lab:26s} " + "   ".join(f"{REG[r]}: {o[r,0]:+.3f} [{o[r,1]:+.3f},{o[r,2]:+.3f}]" for r in range(3)))
print("posterior sd of #factors by regime:", [round(float(nsd[reg == r].mean()), 2) for r in range(3)],
      "| mean posterior #factors:", [round(float(nmean[reg == r].mean()), 2) for r in range(3)], "| plug-in:", [round(float(plug[reg == r].mean()), 2) for r in range(3)])
