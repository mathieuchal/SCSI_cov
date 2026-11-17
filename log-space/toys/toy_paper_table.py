"""Paper table of the d=8 toys: predictive log-score gap to the oracle and W1 distance to the oracle posterior, for SC-SI
(the production run at its best_k) and IW-ML.

Inputs (no computation, only reads):
  results/toys_M=-1/{toy}_logscore.json   (toy_oracle_log_score.py): per-pair log-scores of the oracle, IW-ML and SC-SI at
                                     best_k (J=2e6, 10 validation pairs)
  results/toys_M=-1/{toy}_W1.json         (toy_eval.py): W1 / oracle sd per statistic, mean or median over the 128 test tasks

Columns: log-score gap = oracle - method (nats per task, paired over the validation pairs; lower is better, negative =
above the oracle), then the six W1 statistics and their average ("avg"; corr01 is "--" for the diagonal toy, whose
reference has zero sd there). The oracle is the closed form for the IW toys ("oracle_exact"), the Monte-Carlo
oracle at the same J otherwise. For the IW toys the IW-ML gap is the closed-form expected gap IW_EXACT_EXPECTED (the
10-pair estimate is noise there), as in the figure toy_logscore_w1_curve.py. Bold: the better of SC-SI / IW-ML in each
column, at the printed precision.

    python toy_paper_table.py                 -> figs/toy_paper_table.txt (LaTeX rows, W1 median and W1 mean tables)
    python toy_paper_table.py --no_hetero     (leave out the heteroskedastic toy)
"""
import argparse, json, math
from toy_logscore_w1_curve import IW_EXACT_EXPECTED, paired

R = "results/toys_M=-1"
STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
HEAD = ["log-score", r"$\log\det C$", r"$\log\kappa$", "top share", r"$\mathrm{dv}_{\mathrm{top}}$",
        r"$\mathrm{dv}_{\mathrm{bot}}$", r"$C_{01}$", "avg"]
ROWS = [("IW, RI", "iw_ri_d8_inf"), ("IW, non-RI", "iw_nonri_d8_inf"), ("Factor model", "factor_d8_inf"),
        ("Iso-Diagonal", "iso_diag_d8_inf"), ("AR(1)", "ar1_d8_inf"), ("Diagonal", "diag_d8_inf"),
        ("Heteroskedastic", "hetero_d8_inf")]
METH = [("scsi", r"\scsi{}"), ("iw_ml", "IW-ML")]
PREC = [2] + [2] * len(STATS) + [3]          # decimals per column


def values(toy, agg):
    """{method: [log-score gap, W1 per statistic..., W1 avg]} for one toy; None where undefined."""
    L = json.load(open(f"{R}/{toy}_logscore.json"))
    W = json.load(open(f"{R}/{toy}_W1.json"))
    assert W[agg]["scsi"]["tag"] == L["scsi"]["tag"] and W[agg]["scsi"]["k"] == L["scsi"]["k"], \
        f"{toy}: _W1.json and _logscore.json describe different checkpoints"
    exact = "oracle_exact" in L
    orc = L["oracle_exact"] if exact else L["oracle"]
    fin = lambda x: x if x is not None and math.isfinite(x) else None
    gap = {"scsi": -paired(L["scsi"], orc)[0],
           "iw_ml": -IW_EXACT_EXPECTED if exact else -paired(L["iw_ml"], orc)[0]}
    out = {m: [gap[m]] + [fin(W[agg][m]["detail"][s]["w1"]) for s in STATS] + [fin(W[agg][m]["value"])] for m, _ in METH}
    return out, L["scsi"]["k"], L["scsi"]["tag"]


def table(agg, rows):
    lines, info = [], []
    for i, (name, toy) in enumerate(rows):
        v, k, tag = values(toy, agg)
        info.append(f"%   {name}: {toy}_{tag}, k={k}")
        r = lambda x, c: round(x, PREC[c])
        bold = [{m for m in v if v[m][c] is not None and
                 r(v[m][c], c) == min(r(v[x][c], c) for x in v if v[x][c] is not None)} for c in range(len(PREC))]
        if i:
            lines.append(r"    \midrule")
        for j, (m, lab) in enumerate(METH):
            cells = []
            for c, x in enumerate(v[m]):
                if x is None:
                    cells.append("--"); continue
                s = f"{x + 0.0:.{PREC[c]}f}".replace("-0.00", "0.00")
                cells.append(rf"\textbf{{{s}}}" if m in bold[c] else s)
            lines.append(f"    {name if j == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
    head = [f"% W1 / oracle sd, {agg} over the 128 test tasks; log-score gap = oracle - method (J=2e6, 10 pairs)", *info,
            r"\begin{tabular}{@{}ll" + "c" * len(PREC) + "@{}}", r"    \toprule",
            "    Prior & Method & " + " & ".join(HEAD) + r" \\", r"    \midrule"]
    return "\n".join(head + lines + [r"    \bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_hetero", action="store_true")
    ap.add_argument("--out", default="figs/toy_paper_table.txt")
    a = ap.parse_args()
    rows = [r for r in ROWS if not (a.no_hetero and r[1] == "hetero_d8_inf")]
    txt = "\n\n".join(table(agg, rows) for agg in ("median", "mean")) + "\n"
    open(a.out, "w").write(txt)
    print(txt)
    print("wrote", a.out)
