"""Training trajectory of the d=8 toys: SC-SI log-score minus oracle (top row) and SC-SI W1 (bottom row) vs the
EM outer iteration k -- one column per toy, two separate y-axes (no dual axis).

Inputs
  log-score (toy_oracle_log_score.py, J=2e6, 10 validation pairs): SC-SI trajectory = "val_curve" of the run's
             results/toys/{toy}_{TAGS[toy]}_train.json; oracle and IW-ML from results/toys/{toy}_logscore.json
             (the oracle is the closed form for the IW toys, HMC/grid Monte-Carlo at the same J otherwise)
  W1 (default median over the test tasks, --mean the mean): SC-SI trajectory = "w1_curve_median" / "w1_curve_mean"
             of the run's _train.json (toy_W1_mean.py); IW-ML from results/toys/{toy}_W1.json (toy_eval.py).

    python toy_logscore_w1_curve.py               -> figs/logscore_w1_vs_k_d8.png
    python toy_logscore_w1_curve.py --no_hetero   -> figs/logscore_w1_vs_k_d8_nohetero.png
    python toy_logscore_w1_curve.py --mean        -> figs/logscore_w1_vs_k_d8_meanW1.png
"""
import json, math, os, re, sys
import sys
import numpy as np
import scipy.stats as sps
import matplotlib.ticker as mticker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotstyle import *

setup()
R = "results/toys"
TOYS = [("iw_ri_d8_inf", "IW, RI -- MLP-4 w512"), ("iw_nonri_d8_inf", "IW, non-RI -- MLP-4 w512"),
        ("factor_d8_inf", "Factor -- MLP-4 w512"), ("iso_diag_d8_inf", "Iso-diagonal -- MLP-4 w1024"),
        ("ar1_d8_inf", "AR(1) -- ResNet-4 w1024"), ("diag_d8_inf", "Diagonal -- MLP-4 w512"),
        ("hetero_d8_inf", "Heteroskedastic -- MLP-4 w512")]
STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
# production run of each toy (its _train.json holds the log-score trajectory val_curve)
TAGS = {"iw_ri_d8_inf": "mlp_w512_d4", "iw_nonri_d8_inf": "mlp_w512_d4", "factor_d8_inf": "mlp_w512_d4",
        "hetero_d8_inf": "mlp_w512_d4", "diag_d8_inf": "mlp_w512_d4", "iso_diag_d8_inf": "mlp_w1024_d4",
        "ar1_d8_inf": "resnet_w1024_d4"}
# common checkpoint grid for every toy (the AR1 ResNet one): 24, 60, 100..1000 by 40, then 1100..2000 by 100
GRID = {24, 60, *range(100, 1001, 40), 1000, *range(1100, 2001, 100)}
COL_W = 2.35                    # inches per toy column (was 3.3): compact horizontally
IW_EXACT_EXPECTED = -0.0010    # E[IW-ML - oracle] for iw_ri / iw_nonri d8 (closed form, 20000 pairs)
LS_MAIN, LS_STRIP = (-3.4, 0.3), (-10.5, -3.4)   # shared log-score y-ranges (main panel, strip below the break)
W1_MAIN, W1_STRIP = (0.0, 0.9), (0.9, 4.6)       # shared W1 y-ranges, linear (main panel, strip above the break)


def w1_case(x, ref):
    """per-test-case W1 / sd(ref), as toy_eval.w1_norm."""
    sd = ref.std(1)
    w = np.array([sps.wasserstein_distance(x[m], ref[m]) for m in range(x.shape[0])])
    return np.where(sd > 0, w / np.where(sd > 0, sd, 1.0), np.nan)       # degenerate stat (e.g. corr01 on diag): nan


def paired(a, b):
    x = np.array(a["per_task"]) - np.array(b["per_task"])
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def main(J=2000000, n_val=10, out="figs/logscore_w1_vs_k_d8.png", agg="median", toys=None):
    TOYS = toys if toys is not None else globals()["TOYS"]
    fig, axs = plt.subplots(2, len(TOYS), figsize=(COL_W * len(TOYS), 4.8), sharex="col",
                            gridspec_kw={"height_ratios": [1, 0.9]})
    for c, (toy, title) in enumerate(TOYS):
        o = json.load(open(f"{R}/{toy}_logscore.json"))
        assert (o["J"], o["n_val"]) == (J, n_val), toy
        stem = f"{toy}_{TAGS[toy]}"
        exact = "oracle_exact" in o
        orc = o["oracle_exact"] if exact else o["oracle"]
        # --- top: log-score gap ---
        ax = axs[0, c]
        tr = json.load(open(f"{R}/{stem}_train.json"))
        assert (tr["val_curve_J"], tr["val_curve_n_val"]) == (J, n_val), stem
        vc = {int(k): v for k, v in tr["val_curve"].items()}
        ks = sorted(k for k in vc if k in GRID or 0 < k < 24)
        g = np.array([vc[k] - orc["mean"] for k in ks])          # same pairs: mean gap = mean of the paired gaps
        ax.axhline(0, color=INK, lw=1.2, zorder=2)
        lo, hi = min(g.min(), 0.0), max(g.max(), 0.0)
        if exact:
            # conjugate toys: the 10-pair estimate (~ +0.03 +- 0.015) is noise; plot the expected gap instead, computed in
            # closed form on 20000 fresh pairs with the same ML fit: -0.0010 +- 0.0003
            iwg = IW_EXACT_EXPECTED
        else:
            iwg = paired(o["iw_ml"], orc)[0] if "iw_ml" in o else None
        # same y-scale for every toy: a main panel LS_MAIN and, below a // break, a compressed strip LS_STRIP that
        # holds what falls far below (early AR1 checkpoints, the IW-ML of diag / AR1)
        sub = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[4, 1.1], hspace=0.10)
        ax.set_subplotspec(sub[0])
        axb = fig.add_subplot(sub[1], sharex=ax)
        for a in (ax, axb):
            a.plot(ks, g, color=BLUE, lw=2, marker="o", ms=2.5, zorder=3)
            if iwg is not None:
                a.axhline(iwg, color=ORANGE, lw=1.4, ls=(0, (4, 2)), zorder=2)
        ax.set_ylim(*LS_MAIN)
        axb.set_ylim(*LS_STRIP)
        axb.set_yticks([-10, -6])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axb.spines["top"].set_visible(False)
        axb.tick_params(axis="x", labelbottom=False)
        kwb = dict(color=INK2, clip_on=False, lw=1)
        ax.plot((-0.02, 0.02), (-0.015, 0.015), transform=ax.transAxes, **kwb)
        axb.plot((-0.02, 0.02), (0.94, 1.06), transform=axb.transAxes, **kwb)
        ax.set_title(title.replace(" -- ", "\n"), fontsize=9.5)            # architecture on a second line
        if c == 0:
            ax.set_ylabel(f"log-score: SC-SI $-$ oracle\n(J=2e6)")
        ax.annotate("oracle exact" if exact else "oracle MC", (0.02, 0.0), xycoords=ax.get_yaxis_transform(),
                    textcoords="offset points", xytext=(0, 3), ha="left", va="bottom", fontsize=7.5, color=INK3)
        # --- bottom: mean-W1 ---
        ax = axs[1, c]
        w = {int(k): v for k, v in tr[f"w1_curve_{agg}"].items()}
        iww = json.load(open(f"{R}/{toy}_W1.json"))[agg]["iw_ml"]["value"]
        # every checkpoint k>0 (early ones k=2..20 included: they carry the steep initial decay)
        kw = sorted(k for k in w if k > 0 and (k in GRID or k < 24) and math.isfinite(w[k]))
        yw = [w[k] for k in kw]
        # linear scale, same for every toy: main panel W1_MAIN and, above a // break, a compressed strip W1_STRIP
        # for what rises far above (early AR1 checkpoints, the AR1 IW-ML)
        sub = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[1.1, 4], hspace=0.10)
        ax.set_subplotspec(sub[1])
        axt = fig.add_subplot(sub[0], sharex=ax)
        for a in (ax, axt):
            a.plot(kw, yw, color=BLUE, lw=2, marker="o", ms=3, zorder=3)
            if math.isfinite(iww):
                a.axhline(iww, color=ORANGE, lw=1.4, ls=(0, (4, 2)), zorder=2)
        if math.isfinite(iww):
            a_lab = axt if iww > W1_MAIN[1] else ax
            a_lab.text(kw[-1], iww, f"IW-ML {iww:.3f}", ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_ylim(*W1_MAIN)
        axt.set_ylim(*W1_STRIP)
        axt.set_yticks([2, 4])
        ax.spines["top"].set_visible(False)
        axt.spines["bottom"].set_visible(False)
        axt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        kwb = dict(color=INK2, clip_on=False, lw=1)
        ax.plot((-0.02, 0.02), (0.985, 1.015), transform=ax.transAxes, **kwb)
        axt.plot((-0.02, 0.02), (-0.06, 0.06), transform=axt.transAxes, **kwb)
        ax.set_xlabel("EM outer iteration k")
        if c == 0:
            ax.set_ylabel("mean W1 / ref sd" if agg == "mean" else "W1 / ref sd")
    h = [plt.Line2D([], [], color=BLUE, lw=2),
         plt.Line2D([], [], color=ORANGE, lw=1.4, ls=(0, (4, 2)))]
    fig.legend(h, ["SC-SI", "IW conjugate (ML-fitted)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("d=8 toys, N=40 -- SC-SI log-score gap to the oracle and " + ("mean-W1" if agg == "mean" else "W1") + " along training", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), w_pad=0.4)
    fig.savefig(out, dpi=170)
    print("saved", out)


if __name__ == "__main__":
    if "--no_hetero" in sys.argv:                # same figure without the heteroskedastic toy
        main(out="figs/logscore_w1_vs_k_d8_nohetero.png", toys=[t for t in TOYS if t[0] != "hetero_d8_inf"])
    elif "--mean" in sys.argv:                     # old aggregation (mean over test cases)
        main(out="figs/logscore_w1_vs_k_d8_meanW1.png", agg="mean")
    else:
        main()
