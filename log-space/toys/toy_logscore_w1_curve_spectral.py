"""Same figure as toy_logscore_w1_curve.py (median-W1 version, heteroskedastic toy left out), with the
spectral-Follmer SC-SI runs overlaid in green in the IW-RI and iso-diagonal columns: log-score gap to the oracle
(top) and W1 (bottom). The spectral runs were trained and evaluated in SCSI_cov/spectra-space; only their curves are
kept here.

Inputs (results/toys_M=-1/spectra/)
  {toy}_spectral_follmer_J200000_fullcurve.json   {"checkpoints": {k: mean log-score over the 10 validation pairs}}, J=2e5;
                                                  iso-diag also has its own "oracle" and "gaps" (on its own pairs)
  w1_mean_d8_spectral_follmer.json                {toy: {"checkpoints": {k: W1/ref-sd, median over the test tasks, mean
                                                  over the statistics}}}
The spectral log-score files hold only the mean over the pairs (no per-pair values), so the green gap is a difference
of means, not a paired difference, estimated at J=2e5 (the standard curves: J=2e6, paired): the run's own "gaps" when
present (iso-diag), else mean - mean(oracle of {toy}_logscore.json) (IW-RI: the closed form).

    python toy_logscore_w1_curve_spectral.py     -> figs/logscore_w1_vs_k_d8_spectral.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import toy_logscore_w1_curve as base
from plotstyle import AQUA
import paths

GREEN = AQUA
SP = f"{paths.res_dir('inf')}/spectra"
SPECTRAL_TOYS = ["iw_ri_d8_inf", "iso_diag_d8_inf"]
TOYS = [t for t in base.TOYS if t[0] != "hetero_d8_inf"]
OUT = f"{paths.figs_dir('inf')}/logscore_w1_vs_k_d8_spectral.png"


def curves():
    """{column title: (k log-score, gap to the oracle, k W1, W1)} of the spectral runs"""
    W1 = json.load(open(f"{SP}/w1_mean_d8_spectral_follmer.json"))
    out = {}
    for toy in SPECTRAL_TOYS:
        sp = json.load(open(f"{SP}/{toy}_spectral_follmer_J200000_fullcurve.json"))
        if "gaps" in sp:          # gap to the oracle computed by the spectral run itself, on its own validation pairs
            gaps = {int(k): v for k, v in sp["gaps"].items()}
        else:                     # no own oracle: subtract the standard run's oracle (same validation pairs assumed)
            o = json.load(open(f"{base.R}/{toy}_logscore.json"))
            orc = (o["oracle_exact"] if "oracle_exact" in o else o["oracle"])["mean"]
            gaps = {int(k): v - orc for k, v in sp["checkpoints"].items()}
        w1 = W1[toy]["checkpoints"]
        k_ls, k_w1 = sorted(gaps), sorted(int(k) for k in w1)
        out[dict(base.TOYS)[toy]] = (k_ls, [gaps[k] for k in k_ls], k_w1, [w1[str(k)] for k in k_w1])
    return out


def overlay(fig):
    """green curves in the columns of the spectral toys (found by their titles and y-ranges), then the legend"""
    close = lambda a, b: np.allclose(a, b, atol=1e-9)
    for title, (k_ls, g_ls, k_w1, y_w1) in curves().items():
        tt = (title, title.replace(" -- ", "\n"))                               # the base figure may wrap the title
        top = [a for a in fig.axes if {a.get_title("left"), a.get_title(), a.get_title("right")} & set(tt)][0]
        x0, x1 = top.get_position().x0, top.get_position().x1
        for a in fig.axes:
            if abs(a.get_position().x0 - x0) > 1e-3 or abs(a.get_position().x1 - x1) > 1e-3:
                continue
            yl = a.get_ylim()
            if close(yl, base.LS_MAIN) or close(yl, base.LS_STRIP):
                a.plot(k_ls, g_ls, color=GREEN, lw=2, marker="o", ms=2.5, zorder=4)
            elif close(yl, base.W1_MAIN) or close(yl, base.W1_STRIP):
                a.plot(k_w1, y_w1, color=GREEN, lw=2, marker="o", ms=3, zorder=4)
    for lg in list(fig.legends):
        lg.remove()
    h = [plt.Line2D([], [], color=base.BLUE, lw=2), plt.Line2D([], [], color=GREEN, lw=2),
         plt.Line2D([], [], color=base.ORANGE, lw=1.4, ls=(0, (4, 2)))]
    fig.legend(h, ["SC-SI", "SC-SI spectral", "IW conjugate (ML-fitted)"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01))


if __name__ == "__main__":
    base.main(out=OUT, agg="median", toys=TOYS, overlay=overlay)
