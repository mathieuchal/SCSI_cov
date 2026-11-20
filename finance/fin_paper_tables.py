"""Finance tables of the paper -> figs/fin_paper_tables.txt (LaTeX).

  estimators : min-variance portfolio, annualised realised volatility and mean log(realised variance / OAS), OAS and
               LW-NLS with the run's sample size n = N = 19 (as stored by fin_exp.py)
  (a)        : coverage of the 50/80/95% central predictive intervals of forward variances, overall and (80%) by
               tercile of market volatility; log-score gains over the sample covariance: tempered (N_forward) matrix
               score (nats / 21-day window), Gaussian and Student-t daily-return scores (nats / day)
  (b)        : mean log(realised variance / OAS) by tercile of market volatility

Market-volatility terciles: trailing 63-day realised volatility of the equal-weight portfolio of all industries at the
forecast origin (no look-ahead), terciles over the test months. 95% CI: month-block bootstrap (covutils.bootstrap_ci,
1000 resamples, seed 0). Bold: best point estimate per column (lowest log ratio; coverage closest to the nominal 80%;
highest gain).

Inputs: results/{RUN}_results.json and _arrays.npz (fin_exp.py --select tscore), data/ff48_vw_daily.pkl.

    python fin_paper_tables.py
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "log-space", "toys"))
from covutils import bootstrap_ci

RUN = "fin_d12_nw63_s0_mlp_w1024_d4_neff03_nfe6_o30_selT"
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figs", "fin_paper_tables.txt")
ci = lambda lo, hi, dg: rf"{{\scriptsize$[{lo:.{dg}f},\,{hi:.{dg}f}]$}}"
bf = lambda s, b: rf"$\mathbf{{{s}}}$" if b else f"${s}$"


def table_estimators(res):
    rows = [("OAS", "OAS shrinkage"), ("Sample cov.", "SCM"), ("LW-NLS", "LW nonlinear shrinkage"), None,
            ("IW-ML posterior mean", "IW post. mean"), (r"\scsi{} posterior mean", "SC-SI post. mean"),
            (r"\scsi{} Stein-optimal", "SC-SI Stein-opt."), None,
            (r"PCA, $K{=}1$", "PCA K=1"), (r"PCA, $K{=}3$", "PCA K=3"), ("PCA, Gavish--Donoho $K$", "PCA K=GD (plug-in)"),
            ("PCA, Bayes $K$ (IW-ML)", "PCA K=Bayes (IW)"), (r"PCA, Bayes $K$ (\scsi{})", "PCA K=Bayes (SC-SI)")]
    mv = res["mv"]
    best = min((r[1] for r in rows if r), key=lambda k: round(mv[k]["log_var_ratio_vs_OAS"][0], 3))
    out = [r"\begin{tabular}{@{}lcc@{}}", r"    \toprule", r"    Estimator & ann.\ vol.\ (\%) & log ratio to OAS $\downarrow$ \\", r"    \midrule"]
    for r in rows:
        if r is None:
            out.append(r"    \midrule"); continue
        lab, k = r; m, lo, hi = mv[k]["log_var_ratio_vs_OAS"]
        cell = "$0$" if k == "OAS shrinkage" else bf(f"{m:+.3f}", k == best) + " " + ci(lo, hi, 3)
        out.append(f"    {lab} & {mv[k]['ann_vol']:.2f} & {cell} \\\\")
    return out + [r"    \bottomrule", r"\end{tabular}"]


def table_a(z, reg, end):
    rows = [("Sample cov.", "SCM"), ("OAS", "OAS_shrinkage"), ("LW-NLS", "LW_nonlinear_shrinkage"), ("IW-ML", "IW"), (r"\scsi{}", "SC-SI")]
    inside = lambda p, lv: ((p > (1 - lv) / 2) & (p < 1 - (1 - lv) / 2)).mean(1)
    terc = {k: [float(inside(z["pit_" + k], 0.8)[reg == r].mean()) for r in range(3)] for _, k in rows}
    gains = {k: [bootstrap_ci(z[p + k] - z[p + "SCM"], end) for p in ("ll_", "dl_", "dt_")] for _, k in rows if k != "SCM"}
    bt = [{k for k in terc if round(abs(terc[k][r] - 0.8), 3) == min(round(abs(terc[x][r] - 0.8), 3) for x in terc)} for r in range(3)]
    bg = [max(gains, key=lambda k: round(gains[k][j][0], 2 if j else 1)) for j in range(3)]
    out = [r"\begin{tabular}{@{}lccccccc@{}}", r"    \toprule",
           r"     & overall & \multicolumn{3}{c}{80\% by volatility tercile} & \multicolumn{3}{c}{log-score gain vs.\ sample cov.\ $\uparrow$} \\",
           r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"    (a) Method & 50/80/95 & low & mid & high & tempered & Gaussian & Student-$t$ \\", r"    \midrule"]
    for lab, k in rows:
        ov = "/".join(f"{inside(z['pit_' + k], lv).mean():.2f}" for lv in (0.5, 0.8, 0.95))
        tc = [rf"\textbf{{{v:.2f}}}" if k in bt[r] else f"{v:.2f}" for r, v in enumerate(terc[k])]
        if k == "SCM":
            gc = ["$0$"] * 3
        else:
            gc = [bf(f"{m:+.{1 if j == 0 else 2}f}", k == bg[j]) + " " + ci(lo, hi, 1 if j == 0 else 2) for j, (m, lo, hi) in enumerate(gains[k])]
        out.append(f"    {lab} & {ov} & " + " & ".join(tc + gc) + r" \\")
    return out + [r"    \bottomrule", r"\end{tabular}"]


def table_b(res, z, reg, end):
    rows = [(r"\scsi{} posterior mean", "SC-SI post. mean"), (r"\scsi{} Stein-optimal", "SC-SI Stein-opt."),
            ("IW-ML posterior mean", "IW post. mean"), ("LW-NLS", "LW nonlinear shrinkage"), ("Sample cov.", "SCM")]
    V = {n: z[f"var_{i}"] for i, n in enumerate(res["mv"])}
    val = {k: [bootstrap_ci(np.log(V[k] / V["OAS shrinkage"])[reg == r], end[reg == r]) for r in range(3)] for _, k in rows}
    best = [min(val, key=lambda k: round(val[k][r][0], 3)) for r in range(3)]
    out = [r"\begin{tabular}{@{}lccc@{}}", r"    \toprule", r"    (b) Estimator & low vol. & mid vol. & high vol. \\", r"    \midrule"]
    for lab, k in rows:
        out.append(f"    {lab} & " + " & ".join(bf(f"{m:+.3f}", k == best[r]) + " " + ci(lo, hi, 3)
                                            for r, (m, lo, hi) in enumerate(val[k])) + r" \\")
    return out + [r"    \bottomrule", r"\end{tabular}"]


if __name__ == "__main__":
    res = json.load(open(f"{RES}/{RUN}_results.json")); z = np.load(f"{RES}/{RUN}_arrays.npz")
    ew = np.nanmean(pd.read_pickle(os.path.join(HERE, "data", "ff48_vw_daily.pkl")).values, axis=1)
    end = z["meta"][:, 1]; ue = np.unique(end)
    vol = np.array([np.sqrt(np.mean(ew[e - 62:e + 1] ** 2) * 252) for e in ue])
    q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
    reg = ((vol > q1).astype(int) + (vol > q2).astype(int))[np.searchsorted(ue, end)]
    nu = "/".join(f"{float(v):g}" for v in z["nu_hat"])
    head = (f"% run {RUN}: N={res['N']}, N_forward={res['N_forward']}, {res['n_cases']} test cases; OAS/LW with n=N;\n"
            f"% volatility terciles {q1:.1f}% / {q2:.1f}%; Student-t nu per fold {nu}; month-block bootstrap 95% CI")
    txt = "\n\n".join([head, "% --- estimators (min-variance portfolio)", "\n".join(table_estimators(res)),
                       "% --- (a) calibration and scores", "\n".join(table_a(z, reg, end)),
                       "% --- (b) log ratio to OAS by volatility tercile", "\n".join(table_b(res, z, reg, end))]) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(txt)
    print(txt); print("wrote", OUT)
