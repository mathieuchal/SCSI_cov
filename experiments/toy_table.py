"""Rows of the toy table (paper/experiments.tex, Table tab:toy-w1) from results/toys/.

First column: held-out predictive log-score of Ce_val given Ce_cal (nats per task, 300 validation pairs, J=64 draws; the quantity
plotted in fig_toy_logscore) for SC-SI at its selected checkpoint, the ML-fitted IW baseline and the oracle posterior.
Then: W1 to the oracle posterior in oracle-s.d. units, mean over the 128 held-out tasks, for SC-SI and IW-ML.
Usage: python toy_table.py [ar1_stem]      (default: the AR(1) run used in paper_figs.py)
"""
import json, re, sys
import numpy as np
from toy_eval import STATS

AR1 = sys.argv[1] if len(sys.argv) > 1 else "ar1_d8_M8000_n120"
strip = lambda t: re.sub(r"_n[0-9]+(s[0-9]+)?$", "", t)          # training variant -> toy name (oracle files are per toy)
ROWS = [("iw_ri_d8", "IW, RI"), ("iw_nonri_d8", "IW, non-RI"), ("factor_d8", "Factor model"), (AR1, "AR(1)")]

def best(vals, higher):
    """indices of the best value(s) among the two methods, compared at the printed precision (ties -> both)"""
    r = [round(v, 2) for v in vals]; b = max(r) if higher else min(r)
    return {k for k, x in enumerate(r) if x == b}


for i, (stem, name) in enumerate(ROWS):
    m = json.load(open(f"results/toys/{stem}_metrics.json"))["metrics"]
    tr = json.load(open(f"results/toys/{stem}_train.json")); o = json.load(open(f"results/toys/{strip(stem)}_oracle_logscore.json"))
    ls = {"scsi": tr["val_curve"][str(tr["best_k"])], "iwfit": o["iw_ml_J64"], "oracle": o["oracle_J64"]}
    W = {k: [m[s][k]["w1"] for s in STATS] for k in ("scsi", "iwfit")}
    W = {k: v + [float(np.mean(v))] for k, v in W.items()}                                              # six statistics + mean
    bold_ls = best([ls["scsi"], ls["iwfit"]], higher=True)                                             # bold: best of the two methods (oracle = reference)
    bold_w = [best([W["scsi"][c], W["iwfit"][c]], higher=False) for c in range(7)]
    if i:
        print(r"    \midrule")
    for j, (key, meth) in enumerate([("scsi", r"\scsi{}"), ("iwfit", "IW-ML"), ("oracle", "Oracle")]):
        lsc = f"{ls[key]:.2f}"; lsc = f"$\\mathbf{{{'-' if ls[key] < 0 else ''}{abs(ls[key]):.2f}}}$" if j in bold_ls and key != "oracle" else f"${lsc}$"
        if key == "oracle":
            w = " & ".join(["--"] * 7)
        else:
            w = " & ".join((f"\\textbf{{{W[key][c]:.2f}}}" if j in bold_w[c] else f"{W[key][c]:.2f}") for c in range(7))
        print(f"    {name if j == 0 else ''} & {meth} & {lsc} & {w} \\\\")
