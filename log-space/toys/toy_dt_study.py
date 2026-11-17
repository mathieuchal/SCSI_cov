"""Effect of the SDE time step Delta t = 1/n_steps on SC-SI sampling, at evaluation only (no retraining).

The Follmer SDE of SCSI.sample_posterior is integrated by Euler-Maruyama on t in [0, 1] with n_steps steps
(default SCSIConfig.n_sde_steps = 64, i.e. Delta t = 1/64, used by every production result). For the production run
of each toy, at its best_k, and for each n_steps, this script recomputes with the production protocols:
  log-score : as toy_logscore_scsi.py -- J=2e6 posterior draws on the 10 validation pairs of toy_train.py
              (Generator(1000+seed)), seed 777, TF32 on; gap to the stored oracle of {toy}_logscore.json
  W1        : as toy_W1_scsi.py -- 128 test tasks, J=1024, the cached HMC reference {toy}_ref.pt,
              seed 99, TF32 during sampling; w1.w1_summary (mean and median over tasks)
With n_steps=64 it must reproduce {toy}_logscore.json["scsi"] and {toy}_W1.json[agg]["scsi"] (control).

Writes results/toys_M=-1/{toy}_dt.json = {"tag", "k", "n_steps": {n: {"logscore": {mean, se, per_task}, "W1": {mean, median}}}}
(resumable: n_steps already in the file are skipped), then the table figs/toy_dt_table.txt (n_steps=64 rows from the
production files).

    python toy_dt_study.py --toys iw_nonri_d8_inf,factor_d8_inf,diag_d8_inf --n_steps 32,128
"""
import argparse, json, os, time
import numpy as np, torch
from toys import make_toy
from toy_data import make_testset
from w1 import AGGS, stats_np, get_reference, w1_scsi
from logscore import log_score_scsi
from toy_data import val_pairs
import paths
from toy_paper_table import STATS, ROWS
from toy_logscore_w1_curve import paired

R = paths.res_dir("inf")                 # the time-step study is run on the M=inf production runs


def logscore(toy, path, n_steps, n_val, J, seed, chunk, threads, device):
    """{mean, se, per_task} of the checkpoint with n_steps SDE steps, on toy_train.py's validation pairs"""
    Ce_cal, Ce_val = val_pairs(toy, "report", n_val, seed, device)
    e = log_score_scsi(toy, path, Ce_cal, Ce_val, J, device, threads, chunk, n_steps=n_steps)
    return {x: e[x] for x in ("mean", "se", "per_task")}


def table(toys, out=f"{paths.figs_dir('inf')}/toy_dt_table.txt"):
    """Per toy, one row per n_steps. Log-score: gap to the oracle (oracle - SC-SI) and paired difference with the
    production n_steps=64 (SC-SI at Delta t - SC-SI at 1/64), both mean +- se over the validation pairs (same pairs for
    every Delta t, so the large pair-to-pair variability cancels). W1: average over the six statistics, median and mean
    over the test tasks."""
    name = dict((t, n) for n, t in ROWS)
    pm = lambda m, se, dg=2: rf"${m + 0.0:+.{dg}f} \pm {se:.{dg}f}$"
    tex = [r"% SC-SI at best_k vs SDE time step Delta t = 1/n_steps (Euler-Maruyama); n_steps=64 = production.",
           r"% log-score (J=2e6, 10 validation pairs): gap = oracle - SC-SI, diff = SC-SI(Delta t) - SC-SI(1/64), paired, mean +- se.",
           r"% W1 / oracle sd, average over the six statistics, median / mean over 128 test tasks (J=1024).",
           r"\begin{tabular}{@{}llcccc@{}}", r"    \toprule",
           r"    Prior & $\Delta t$ & gap to oracle & diff.\ vs $1/64$ & W1 (median) & W1 (mean) \\", r"    \midrule"]
    md = ["| Prior | Δt | gap to oracle | diff. vs 1/64 | W1 (median) | W1 (mean) |", "|---|---|---|---|---|---|"]
    for i, toy in enumerate(toys):
        L = json.load(open(f"{R}/{toy}_logscore.json")); W = json.load(open(f"{R}/{toy}_W1.json"))
        D = json.load(open(f"{R}/{toy}_dt.json"))
        orc = L["oracle_exact"] if "oracle_exact" in L else L["oracle"]
        rows = {int(n): v for n, v in D["n_steps"].items()}
        rows.setdefault(64, {"logscore": L["scsi"], "W1": {ag: W[ag]["scsi"] for ag in AGGS}})
        if i:
            tex.append(r"    \midrule")
        for j, n in enumerate(sorted(rows)):
            v = rows[n]
            g, gse = paired(orc, v["logscore"])
            diff = "--" if n == 64 else pm(*paired(v["logscore"], rows[64]["logscore"]), 3)
            cells = [pm(g, gse), diff, f"{v['W1']['median']['value']:.3f}", f"{v['W1']['mean']['value']:.3f}"]
            lab = name.get(toy, toy) if j == 0 else ""
            tex.append(f"    {lab} & $1/{n}$ & " + " & ".join(cells) + r" \\")
            md.append(f"| {lab} | 1/{n} | " + " | ".join(c.replace("$", "").replace(r"\pm", "±") for c in cells) + " |")
    txt = "\n".join(tex + [r"    \bottomrule", r"\end{tabular}"]) + "\n\n" + "\n".join(md) + "\n"
    open(out, "w").write(txt)
    print(txt); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toys", default="iw_ri_d8_inf,iw_nonri_d8_inf,factor_d8_inf,iso_diag_d8_inf,ar1_d8_inf,diag_d8_inf")
    ap.add_argument("--n_steps", default="32,128", help="comma list of SDE step counts (Delta t = 1/n_steps)")
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--J", type=int, default=2000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--table_only", action="store_true")
    a = ap.parse_args()
    toys = a.toys.split(",")
    if not a.table_only:
        torch.set_num_threads(a.threads)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for toy_name in toys:
            toy = make_toy(toy_name); tag = paths.tag(toy_name); stem = f"{toy_name}_{tag}"
            k = json.load(open(f"{R}/{stem}_train.json"))["best_k"]
            path = f"{R}/{stem}_k{k}.pt"
            out_path = f"{R}/{toy_name}_dt.json"
            D = json.load(open(out_path)) if os.path.exists(out_path) else {"tag": tag, "k": k, "n_steps": {}}
            assert (D["tag"], D["k"]) == (tag, k), f"{out_path} describes another checkpoint"
            _, Ce = make_testset(toy, 128); Ce = Ce.to(device)
            ref_stats = stats_np(get_reference(toy, Ce, 2048, f"{R}/{toy_name}_ref.pt")[0], Ce)
            for n in [int(x) for x in a.n_steps.split(",")]:
                if str(n) in D["n_steps"]:
                    continue
                t0 = time.time()
                torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # as toy_logscore_scsi.py
                ls = logscore(toy, path, n, a.n_val, a.J, a.seed, a.chunk, a.threads, device)
                w = w1_scsi(toy, path, Ce, 1024, ref_stats, a.threads, n_steps=n)
                D["n_steps"][str(n)] = {"logscore": ls, "W1": w}
                json.dump(D, open(out_path, "w"), indent=1)
                print(f"{toy_name} k={k} n_steps={n}: log-score {ls['mean']:.3f}  W1 median {w['median']['value']:.4f}  "
                      f"mean {w['mean']['value']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    table(toys)
