"""Predictive log-score of the SC-SI checkpoints of a run, on the validation pairs of toy_data.py, at large J: the
trajectory along training, the choice of best_k with finite data, and the value at best_k. The references (oracle,
IW-ML) are scored on the SAME pairs at the same J by toy_logscore_oracle.py; the scoring is in logscore.py
(SC-SI posterior draws with seed 777, in blocks of 2e5, TF32 on).

  1. trajectory, selection pairs: every checkpoint of --ks -> the run's {stem}_train.json
       "val_curve" {k: mean log-score}, "val_curve_per_task" {k: per-pair log-scores} (checkpoints scored before
       per-pair values were kept have only the mean), "val_curve_J", "val_curve_n_val", "val_curve_val_seed"
  2. best_k: finite data, argmax of val_curve (the toy_train.py callback choice, J=64, is kept as "best_k_orig");
       M=inf, the run's own best_k (the last checkpoint, imposed by toy_train.py), unchanged
  3. report pairs: the run at best_k -> {toy}_logscore.json "scsi" {tag, k, mean, se, per_task, ess, sec}, next to
       the references. M=inf: the report pairs are the selection pairs, so best_k is scored once for both.
Resumable: checkpoints already in val_curve, and a "scsi" entry already at (tag, best_k), are skipped. A file computed
with another (J, n_val, val_seed) raises.

--ks: 'grid' (paths.GRID, the production grid, M=inf default), 'all' (every checkpoint on disk, finite
data default), a comma list, or 'best' (no trajectory: only the value at best_k); best_k is always scored.

    python toy_logscore_scsi.py                                 # the M=inf runs, J=2e6, 10 replayed pairs
    python toy_logscore_scsi.py --data M5000                    # the M=5000 runs: trajectory, best_k, report
    python toy_logscore_scsi.py --data M5000 --J 200000 --no_report   # a trajectory at reduced budget only
"""
import argparse, json, os, time
from toys import make_toy
from logscore import log_score_scsi
from toy_data import val_pairs, val_seed
from toy_logscore_oracle import check_key, common_args, setup
from runs import checkpoints, select_toys
import paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    common_args(ap)
    ap.add_argument("--ks", default=None, help="'grid' | 'all' | 'best' | comma list (default: grid for M=inf, all else)")
    ap.add_argument("--no_report", action="store_true", help="trajectory (and best_k) only, no {toy}_logscore.json entry")
    ap.add_argument("--chunk", type=int, default=131072, help="SC-SI SDE batch size")
    a = ap.parse_args()
    device = setup(a)
    for name in select_toys(a.toys, a.data):
        toy = make_toy(name); R = paths.res_dir(name); tag = paths.tag(name); stem = f"{name}_{tag}"
        finite = toy.M != -1
        ks = checkpoints(R, stem, a.ks or ("all" if finite else "grid"))
        tr_path, ls_path = f"{R}/{stem}_train.json", f"{R}/{name}_logscore.json"
        tr = json.load(open(tr_path))
        save_tr = lambda: json.dump(tr, open(tr_path, "w"), indent=1)
        shared = val_seed(toy, "select") == val_seed(toy, "report")
        t_toy = time.time()
        if not finite:
            ks = sorted({*ks, tr["best_k"]})                       # M=inf: best_k is known, on the trajectory too
        entries = {}                                               # entries scored now on the selection pairs

        # 1. trajectory, selection pairs
        if ks:
            check_key(tr, (a.J, a.n_val, val_seed(toy, "select")), "val_curve_", tr_path)
            vc = {int(k): v for k, v in tr.get("val_curve", {}).items()}
            pt = {int(k): v for k, v in tr.get("val_curve_per_task", {}).items()}
            todo = [k for k in ks if k not in vc]
            print(f"--- {stem}: trajectory, {len(ks)} checkpoints ({len(todo)} to score), selection pairs "
                  f"(val_seed {val_seed(toy, 'select')}), J={a.J}", flush=True)
            if todo:
                Ce_cal, Ce_val = val_pairs(toy, "select", a.n_val, a.seed, device)
            for k in todo:
                e = entries[k] = log_score_scsi(toy, f"{R}/{stem}_k{k}.pt", Ce_cal, Ce_val, a.J, device, a.threads, a.chunk)
                vc[k], pt[k] = e["mean"], e["per_task"]
                tr["val_curve"] = {str(x): vc[x] for x in sorted(vc)}
                tr["val_curve_per_task"] = {str(x): pt[x] for x in sorted(pt)}
                save_tr()
                ref = tr.get("val_ref", {}).get("oracle")
                print(f"  k={k:5d}  scsi {vc[k]:.3f}" + (f"  (- oracle {vc[k] - ref['mean']:+.3f})" if ref else "")
                      + f"  {e['sec']:.0f}s", flush=True)

        # 2. best_k
        if finite:
            vc = {int(k): v for k, v in tr["val_curve"].items()}
            tr.setdefault("best_k_orig", tr["best_k"])             # callback (J=64) selection, for reference
            tr["best_k"] = max(vc, key=vc.get)
            save_tr()
        best_k = tr["best_k"]
        print(f"  {stem}: best_k = {best_k}" + (f" (argmax of val_curve; callback J=64 picked {tr['best_k_orig']})"
                                                if finite else " (the run's own)"), flush=True)

        # 3. report pairs: the run at best_k
        if a.no_report:
            continue
        doc = json.load(open(ls_path)) if os.path.exists(ls_path) else {}
        doc = doc or {"toy": name, "N": toy.N, "d": toy.d}             # new file: J, n_val, val_seed set by check_key
        check_key(doc, (a.J, a.n_val, val_seed(toy, "report")), "", ls_path)
        if (doc.get("scsi", {}).get("tag"), doc.get("scsi", {}).get("k")) != (tag, best_k):
            if shared and best_k in entries:
                e = entries[best_k]
            else:
                Ce_cal, Ce_val = val_pairs(toy, "report", a.n_val, a.seed, device)
                e = log_score_scsi(toy, f"{R}/{stem}_k{best_k}.pt", Ce_cal, Ce_val, a.J, device, a.threads, a.chunk)
            doc["scsi"] = {"tag": tag, "k": best_k} | e
            print(f"  report: scsi k={best_k} {e['mean']:.3f}" + (f"  (- oracle {e['mean'] - doc['oracle']['mean']:+.3f})"
                  if "oracle" in doc else ""), flush=True)
        json.dump(doc, open(ls_path, "w"), indent=1)
        print(f"  done {stem} in {time.time() - t_toy:.0f}s", flush=True)
