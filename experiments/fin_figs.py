import json, sys
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import *

setup()
TAG = sys.argv[1] if len(sys.argv) > 1 else "fin_d12_nw63_s0"
res = json.load(open(f"results/{TAG}_results.json"))
arr = np.load(f"results/{TAG}_arrays.npz", allow_pickle=True)
dates = arr["dates"].astype("datetime64[D]")
names = list(arr["names"])
nf = len({k.split("_")[0] for k in arr.files if k.startswith("path")})

# --------------------------------------------------------------- F1: time series for one fixed basket
def cat(key): return np.concatenate([arr[f"path{fi}_{key}"] for fi in range(nf)])
end = cat("end"); t = dates[end]
fig, ax = plt.subplots(3, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.8]})
for k, (a, lab) in enumerate(zip(ax[:2], ["largest eigen-direction (market)", "second eigen-direction"])):
    a.fill_between(t, cat("share_lo")[:, k], cat("share_hi")[:, k], color=BLUE, alpha=0.22, lw=0)
    a.plot(t, cat("share_mean")[:, k], color=BLUE, lw=1.8, label="SC-SI posterior mean and 80% band")
    a.plot(t, cat("plug_share")[:, k], color=GRAY, lw=1.0, label="sample-covariance plug-in")
    a.set_ylabel("variance share"); a.set_title(f"Share of total variance carried by the {lab}")
ax[0].legend(loc="upper right", ncol=2, fontsize=8)
Kg, Kp = cat("K_gd"), cat("K_post")
ax[2].step(t, Kg + 0.06, where="post", color=ORANGE, lw=1.4, label="Gavish-Donoho hard threshold on sample spectrum")
ax[2].step(t, Kp - 0.06, where="post", color=BLUE, lw=1.8, label="Bayes rule (posterior expected Stein loss)")
ax[2].set_ylabel("number of factors K"); ax[2].set_title("Selected number of meaningful eigen-directions")
ax[2].legend(loc="upper right", ncol=2, fontsize=8); ax[2].set_yticks(range(0, int(max(Kg.max(), Kp.max())) + 2))
for a in ax:
    for d0, d1 in (("2000-03-01", "2002-10-01"), ("2007-12-01", "2009-06-30"), ("2020-02-15", "2020-05-01")):
        a.axvspan(np.datetime64(d0), np.datetime64(d1), color=INK3, alpha=0.10, lw=0)
fig.tight_layout(); fig.savefig(f"figs/{TAG}_F1_timeseries.png"); plt.close(fig)

# --------------------------------------------------------------- F2: portfolio + K
fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.0), gridspec_kw={"width_ratios": [1.5, 0.9, 0.9]})
order = ["SCM", "OAS shrinkage", "LW nonlinear shrinkage", "PCA K=1", "PCA K=3", "PCA K=GD (plug-in)", "PCA K=Bayes (IW)", "PCA K=Bayes (SC-SI)",
         "IW post. mean", "SC-SI post. mean", "SC-SI Stein-opt."]
colr = {"SCM": GRAY, "OAS shrinkage": AQUA, "LW nonlinear shrinkage": YELLOW, "PCA K=1": GRAY, "PCA K=3": GRAY, "PCA K=GD (plug-in)": ORANGE,
        "PCA K=Bayes (IW)": "#f2a88e", "PCA K=Bayes (SC-SI)": BLUE, "IW post. mean": "#f2a88e", "SC-SI post. mean": BLUE, "SC-SI Stein-opt.": "#79aee8"}
ys = np.arange(len(order))[::-1]
for y, n in zip(ys, order):
    m, lo, hi = res["mv"][n]["log_var_ratio_vs_OAS"]
    ax[0].plot([lo, hi], [y, y], color=colr[n], lw=2.2, solid_capstyle="round")
    ax[0].plot(m, y, "o", color=colr[n], ms=8, mec=SURFACE, mew=1.5)
    ax[0].text(hi + 0.004, y, f"{res['mv'][n]['ann_vol']:.2f}%", va="center", fontsize=8, color=INK2)
ax[0].axvline(0, color=INK3, lw=1)
ax[0].set_yticks(ys); ax[0].set_yticklabels(order, fontsize=8.5); ax[0].grid(axis="y", visible=False)
ax[0].set_xlabel("mean log(realised variance / OAS)   (left = lower risk; label = ann. vol)")
ax[0].set_title("Out-of-sample min-variance portfolio (21-day)")
K = res["K"]; dmax = len(K["GD_hist"])
x = np.arange(dmax); w = 0.38
ax[1].bar(x - w / 2, np.array(K["GD_hist"]) / sum(K["GD_hist"]), w, color=ORANGE, label="Gavish-Donoho")
ax[1].bar(x + w / 2, np.array(K["post_hist"]) / sum(K["post_hist"]), w, color=BLUE, label="Bayes (SC-SI)")
ax[1].set_xlabel("selected K"); ax[1].set_ylabel("fraction of months"); ax[1].set_title("Choice of K"); ax[1].legend(fontsize=8)
ax[1].grid(axis="x", visible=False)
lv = res["coverage"]
xs = np.arange(3); wb = 0.16
for j, (n, c, lab) in enumerate((("SCM", GRAY, "SCM"), ("LW nonlinear shrinkage", YELLOW, "LW NLS"), ("OAS shrinkage", AQUA, "OAS"), ("IW", ORANGE, "IW"), ("SC-SI", BLUE, "SC-SI"))):
    vals = list(lv[n].values())
    ax[2].bar(xs + (j - 2) * wb, vals, wb - 0.02, color=c, label=lab)
for lvl in (0.5, 0.8, 0.95):
    ax[2].hlines(lvl, xs[[0, 1, 2]][int(np.argmin(np.abs(np.array([0.5, 0.8, 0.95]) - lvl)))] - 0.42,
                 xs[int(np.argmin(np.abs(np.array([0.5, 0.8, 0.95]) - lvl)))] + 0.42, color=INK, lw=1.4)
ax[2].set_xticks(xs); ax[2].set_xticklabels(["50%", "80%", "95%"]); ax[2].set_ylim(0, 1)
ax[2].set_title("Forward variance intervals"); ax[2].set_xlabel("nominal level (black line)"); ax[2].set_ylabel("coverage")
ax[2].legend(fontsize=7.5, loc="upper left"); ax[2].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"figs/{TAG}_F2_summary.png"); plt.close(fig)
print("saved figures for", TAG)
