"""W1 of the SC-SI checkpoints of a run on the test tasks of toy_data.py (128 tasks), against the cached oracle
reference: the trajectory along training and the value at best_k. The references (oracle floor, IW-ML) are scored on
the SAME tasks and reference by toy_W1_oracle.py; the scoring is in w1.py (J=1024 draws per task, seed 99, TF32
during sampling).

Writes
  the run's {stem}_train.json : "w1_curve_mean", "w1_curve_median" {k: value}, "w1_curve_n_test", "w1_curve_n_ref",
                                "w1_curve_J" -- every checkpoint of --ks, plus best_k
  {toy}_W1.json               : mean/median "scsi" = {tag, k=best_k, value, detail: {stat: {w1, std}}}
best_k is read from the run's _train.json (M=inf: the last checkpoint; finite data: set by toy_logscore_scsi.py, so run
that first). Resumable: checkpoints already in the curves, and a "scsi" entry already at (tag, best_k), are skipped.

--ks: 'grid' (paths.GRID, M=inf default), 'all' (every checkpoint on disk, finite-data default), a comma list, or
'best' (only best_k).

    python toy_W1_scsi.py                        # the M=inf runs
    python toy_W1_scsi.py --data M5000           # the M=5000 runs
"""
import argparse, gc, json, time
import torch
from toys import make_toy
from w1 import AGGS, w1_scsi
from runs import checkpoints, select_toys
from toy_W1_oracle import common_args, open_w1, reference
import paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    common_args(ap)
    ap.add_argument("--ks", default=None, help="'grid' | 'all' | 'best' | comma list (default: grid for M=inf, all else)")
    ap.add_argument("--chunk", type=int, default=131072, help="SC-SI sampler batch (draws per forward pass)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in select_toys(a.toys, a.data):
        toy = make_toy(name); R = paths.res_dir(name); tag = paths.tag(name); stem = f"{name}_{tag}"
        t_toy0 = time.time()
        tr_path = f"{R}/{stem}_train.json"; tr = json.load(open(tr_path))
        best_k = tr["best_k"]
        key = (a.n_test, a.n_ref, a.J)
        if tr.get("w1_curve_J") is not None and (tr["w1_curve_n_test"], tr["w1_curve_n_ref"], tr["w1_curve_J"]) != key:
            raise ValueError(f"{tr_path}: w1 curves computed with another (n_test, n_ref, J)")
        curves = {agg: {int(k): v for k, v in tr.get(f"w1_curve_{agg}", {}).items()} for agg in AGGS}
        W, w_path = open_w1(toy, a)
        need_best = any((W[agg].get("scsi", {}).get("tag"), W[agg].get("scsi", {}).get("k")) != (tag, best_k) for agg in AGGS)
        ks = sorted({*checkpoints(R, stem, a.ks or ("all" if toy.M != -1 else "grid")), best_k})
        todo = [k for k in ks if not all(k in curves[agg] for agg in AGGS) or (k == best_k and need_best)]
        print(f"--- {stem}: {len(ks)} checkpoints ({len(todo)} to score), best_k = {best_k}", flush=True)

        def save():
            for agg in AGGS:
                tr[f"w1_curve_{agg}"] = {str(k): curves[agg][k] for k in sorted(curves[agg])}
            tr["w1_curve_n_test"], tr["w1_curve_n_ref"], tr["w1_curve_J"] = key
            json.dump(tr, open(tr_path, "w"), indent=1)

        if todo:
            Ce, ref_stats = reference(toy, a, device)
        for k in todo:
            t0 = time.time()
            s = w1_scsi(toy, f"{R}/{stem}_k{k}.pt", Ce, a.J, ref_stats, a.threads, a.chunk)
            for agg in AGGS:
                curves[agg][k] = s[agg]["value"]
            save()
            if k == best_k:
                for agg in AGGS:
                    W[agg]["scsi"] = {"tag": tag, "k": k, **s[agg]}
                json.dump(W, open(w_path, "w"), indent=1)
            print(f"  k={k:5d}  mean {s['mean']['value']:.4f}  median {s['median']['value']:.4f}  ({time.time() - t0:.0f}s)"
                  + ("  -> _W1.json" if k == best_k else ""), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        print(f"  done {stem} in {time.time() - t_toy0:.0f}s", flush=True)
