"""Rows of the toy W1 table (paper/experiments.tex, Table tab:toy-w1) from results/toys/{stem}_metrics.json.

W1 to the oracle posterior in oracle-s.d. units, mean over the held-out tasks, for SC-SI and the ML-fitted IW baseline.
Usage: python toy_table.py [ar1_stem]      (default: the AR(1) run used in paper_figs.py)
"""
import json, sys
import numpy as np
from toy_eval import STATS

AR1 = sys.argv[1] if len(sys.argv) > 1 else "ar1_d8_M8000_n120"
ROWS = [("iw_ri_d8", "IW, RI"), ("iw_nonri_d8", "IW, non-RI"), ("factor_d8", "Factor model"), (AR1, "AR(1)")]

for i, (stem, name) in enumerate(ROWS):
    m = json.load(open(f"results/toys/{stem}_metrics.json"))["metrics"]
    if i:
        print(r"    \midrule")
    for j, (key, meth) in enumerate([("scsi", r"\scsi{}"), ("iwfit", "IW-ML")]):
        w = [m[s][key]["w1"] for s in STATS]
        print(f"    {name if j == 0 else ''} & {meth} & " + " & ".join(f"{x:.2f}" for x in w) + f" & {np.mean(w):.2f} \\\\")
