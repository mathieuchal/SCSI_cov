"""Predictive log-score of the references -- the oracle posterior and the ML-fitted IW baseline -- on the validation
pairs of toy_data.py, at large J. The SC-SI checkpoints are scored on the SAME pairs at the same J by
toy_logscore_scsi.py; the scoring of each method is in logscore.py.

  oracle       : direct posterior draws (IW conjugate, AR(1) grid) or HMC (2000 chains per pair, warmup 1500); the IW
                 toys also get the closed-form predictive "oracle_exact" (a direct check of the MC bias)
  iw_ml        : IW prior fitted by maximum likelihood (fit ensemble: toy_data.iwml_logscore_fit)

Writes, per toy:
  report pairs : {results folder}/{toy}_logscore.json = {toy, n_val, J, N, d, val_seed, oracle[, oracle_exact], iw_ml}
                 (then toy_logscore_scsi.py adds "scsi", the run at its best_k; read by toy_paper_table.py)
  select pairs : (finite data) into the run's {stem}_train.json "val_ref" {oracle[, oracle_exact], iw_ml},
                 the references of its trajectory "val_curve" (same val_curve_J / _n_val / _val_seed). Nothing to do
                 for M=inf, whose two roles share the replayed pairs: its references are those of {toy}_logscore.json.
Resumable per entry; a file computed with another (J, n_val, val_seed) raises.

    python toy_logscore_oracle.py                               # the M=inf toys, J=2e6, 10 replayed pairs
    python toy_logscore_oracle.py --data M5000                  # the M=5000 toys, report and selection pairs
    python toy_logscore_oracle.py --data M5000 --roles select --J 200000   # references of a reduced-budget trajectory
"""
import argparse, json, os
import torch
from toys import make_toy
from logscore import log_score_oracle, log_score_iwml
from toy_data import val_pairs, val_seed, iwml_logscore_fit
import paths
from runs import select_toys


def check_key(d, key, prefix, path):
    """d holds (J, n_val, val_seed) under prefix+J, ... (or nothing yet): another setting raises; then set it"""
    names = [prefix + x for x in ("J", "n_val", "val_seed")]
    if any(n in d for n in names[:2]) and tuple(d.get(n) for n in names) != key:
        raise ValueError(f"{path} holds (J, n_val, val_seed) = {tuple(d.get(n) for n in names)}, not {key}")
    d.update(zip(names, key))


def common_args(ap):
    ap.add_argument("--toys", default=None, help="comma-separated toy names (default: the runs of --data on disk)")
    paths.add_data_arg(ap)
    ap.add_argument("--J", type=int, default=2000000, help="posterior draws per validation pair, for every method")
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="toy_train.py's --seed")
    ap.add_argument("--threads", type=int, default=4)


def setup(a):
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    common_args(ap)
    ap.add_argument("--roles", default="report,select", help="pairs to score: report ({toy}_logscore.json) and/or select "
                    "(the trajectory's val_ref; skipped for M=inf, whose two roles share the same pairs)")
    ap.add_argument("--no_iw", action="store_true", help="skip the IW-ML baseline")
    ap.add_argument("--hmc_chains", type=int, default=2000, help="HMC chains per pair for the non-conjugate oracles")
    ap.add_argument("--warmup", type=int, default=1500)
    a = ap.parse_args()
    device = setup(a)
    for name in select_toys(a.toys, a.data):
        toy = make_toy(name); R = paths.res_dir(name); stem = f"{name}_{paths.tag(name)}"
        roles = [r for r in a.roles.split(",") if r == "report" or val_seed(toy, "select") != val_seed(toy, "report")]
        for role in roles:
            key = (a.J, a.n_val, val_seed(toy, role))
            if role == "report":
                path = f"{R}/{name}_logscore.json"
                doc = json.load(open(path)) if os.path.exists(path) else {}
                doc = {"toy": name, "n_val": a.n_val, "J": a.J, "N": toy.N, "d": toy.d, **doc}
                check_key(doc, key, "", path); ref = doc
            else:
                path = f"{R}/{stem}_train.json"
                doc = json.load(open(path))
                check_key(doc, key, "val_curve_", path); ref = doc.setdefault("val_ref", {})
            save = lambda: json.dump(doc, open(path, "w"), indent=1)
            print(f"--- {name}, {role} pairs (val_seed {key[2]}), J={a.J}", flush=True)
            Ce_cal, Ce_val = val_pairs(toy, role, a.n_val, a.seed, device)
            if "oracle" not in ref:
                ref |= log_score_oracle(toy, Ce_cal, Ce_val, a.J, device, a.hmc_chains, a.warmup); save()
                o = ref["oracle"]
                print(f"  oracle {o['mean']:.3f} +- {o['se']:.3f}  (ESS min {min(o['ess']):.0f}, {o['sec']:.0f}s)"
                      + (f"  exact {ref['oracle_exact']['mean']:.3f}" if "oracle_exact" in ref else ""), flush=True)
            if "iw_ml" not in ref and not a.no_iw:
                iw = iwml_logscore_fit(toy, a.n_val, a.seed, device)
                ref["iw_ml"] = log_score_iwml(iw, Ce_cal, Ce_val, a.J, device); save()
                print(f"  iw_ml {ref['iw_ml']['mean']:.3f}  ({ref['iw_ml']['sec']:.0f}s)", flush=True)
            save()
