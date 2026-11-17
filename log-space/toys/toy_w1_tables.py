"""W1 tables of the d=8 toys, from results/toys_M=-1/{toy}_W1.json (toy_W1_oracle.py, toy_W1_scsi.py): SC-SI (the production run at its best_k)
and IW-ML.

W1 / ref-sd per statistic, aggregated over the 128 test tasks by the mean (table "mean") or the median (table "median");
last column = average over the statistics ("value"; corr01 left out for the diagonal toy, whose reference has zero sd
there). Bold: the better of SC-SI / IW-ML in each column, at the printed precision.

    python toy_w1_tables.py      -> figs/toy_w1_table_{mean,median}.{md,tex}
    python toy_w1_tables.py --data M5000   -> figs_M=5000/...  (the M=5000 runs present in results/toys_M=5000)
"""
import argparse, json, math, os
import paths

R = paths.res_dir("inf")
STATS = ["logdet", "logcond", "top_share", "ldv_top", "ldv_bot", "corr01"]
HEAD = ["log det", r"log $\kappa$", "top share", "dv top", "dv bot", r"$C_{01}$"]
ROWS = [("IW, RI", "iw_ri_d8_inf"), ("IW, non-RI", "iw_nonri_d8_inf"), ("Factor", "factor_d8_inf"),
        ("Iso-diag", "iso_diag_d8_inf"), ("AR(1)", "ar1_d8_inf"), ("Diagonal", "diag_d8_inf"),
        ("Heteroskedastic", "hetero_d8_inf")]
METH = [("scsi", "SC-SI", r"\scsi{}"), ("iw_ml", "IW-ML", "IW-ML")]


def table(agg, rows=ROWS):
    md = ["| Toy | Method | " + " | ".join(STATS) + " | avg |", "|" + "---|" * (len(STATS) + 3)]
    tex = []
    for i, (name, toy) in enumerate(rows):
        W = json.load(open(f"{R}/{toy}_W1.json"))[agg]
        fin = lambda x: x if x is not None and math.isfinite(x) else None
        v = {meth: [fin(W[meth]["detail"][s]["w1"]) for s in STATS] + [fin(W[meth]["value"])] for meth, *_ in METH}
        bold = [{meth for meth in v if v[meth][c] is not None and
                 round(v[meth][c], 2 if c < len(STATS) else 3) ==
                 min(round(v[x][c], 2 if c < len(STATS) else 3) for x in v if v[x][c] is not None)}
                for c in range(len(STATS) + 1)]
        if i:
            tex.append(r"    \midrule")
        for j, (meth, lab_md, lab_tex) in enumerate(METH):
            cells_md, cells_tex = [], []
            for c, x in enumerate(v[meth]):
                if x is None:
                    cells_md.append("--"); cells_tex.append("--"); continue
                s = f"{x:.2f}" if c < len(STATS) else f"{x:.3f}"
                cells_md.append(f"**{s}**" if meth in bold[c] else s)
                cells_tex.append(rf"\textbf{{{s}}}" if meth in bold[c] else s)
            md.append(f"| {name if j == 0 else ''} | {lab_md} | " + " | ".join(cells_md) + " |")
            tex.append(f"    {name if j == 0 else ''} & {lab_tex} & " + " & ".join(cells_tex) + r" \\")
    tex = [r"\begin{tabular}{ll" + "c" * (len(STATS) + 1) + "}", r"    \toprule",
           "    Toy & Method & " + " & ".join(HEAD) + r" & avg \\", r"    \midrule", *tex, r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    paths.add_data_arg(ap)
    a = ap.parse_args()
    R, out_dir = paths.res_dir(a.data), paths.figs_dir(a.data)
    rows = [(n, paths.toy_name(paths.base(t), a.data)) for n, t in ROWS]
    rows = [(n, t) for n, t in rows if os.path.exists(f"{R}/{t}_W1.json")]
    os.makedirs(out_dir, exist_ok=True)
    for agg in ("mean", "median"):
        md, tex = table(agg, rows)
        open(f"{out_dir}/toy_w1_table_{agg}.md", "w").write(md + "\n")
        open(f"{out_dir}/toy_w1_table_{agg}.tex", "w").write(tex + "\n")
        print(f"## W1 / ref sd, {agg} over test cases\n{md}\n")
