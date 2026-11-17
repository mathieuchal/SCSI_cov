"""W1 of the references on the test tasks of toy_data.py (128 tasks): the oracle sampling floor (W1 between the two
halves of the reference draws) and the ML-fitted IW baseline. The SC-SI run is scored on the SAME tasks and reference
by toy_W1_scsi.py; the scoring is in w1.py.

  reference : oracle posterior draws, n_ref per task, built once and cached in {results folder}/{toy}_ref.pt (direct
              draws for the exact toys, HMC 8 chains, warmup 1500, seed 4242 otherwise)
  iw_ml     : IW prior fitted on toy_data.iwml_w1's ensemble (Generator(1000+seed); finite data: the training ensemble),
              cached in {toy}_iwfit_ref.pt, J draws per task, seed 99
Writes {results folder}/{toy}_W1.json = {toy, n_test, n_ref, J, N, d, mean: {oracle, iw_ml}, median: {oracle, iw_ml}}
(toy_W1_scsi.py adds "scsi", the run at its best_k; read by toy_paper_table.py and toy_w1_tables.py). Entries already
in the file are kept; another (n_test, n_ref, J) raises. --ref_only only builds the reference cache.

    python toy_W1_oracle.py                      # the M=inf toys
    python toy_W1_oracle.py --data M5000         # the M=5000 toys
"""
import argparse, json, os, time
import torch
from toys import make_toy
from toy_data import make_testset, iwml_w1
from w1 import AGGS, get_reference, stats_np, w1_oracle_floor, w1_iwml
from runs import select_toys
import paths


def common_args(ap):
    ap.add_argument("--toys", default=None, help="comma-separated toy names (default: the runs of --data on disk)")
    paths.add_data_arg(ap)
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_ref", type=int, default=2048, help="reference draws per task (the cache ignores it)")
    ap.add_argument("--J", type=int, default=1024, help="posterior draws per test task")
    ap.add_argument("--threads", type=int, default=26)


def open_w1(toy, a):
    """{toy}_W1.json (or a new header) and its path; another (n_test, n_ref, J) raises"""
    path = f"{paths.res_dir(toy.name)}/{toy.name}_W1.json"
    key = {"n_test": a.n_test, "n_ref": a.n_ref, "J": a.J}
    W = json.load(open(path)) if os.path.exists(path) else {"toy": toy.name, **key, "N": toy.N, "d": toy.d}
    if any(W[x] != v for x, v in key.items()):
        raise ValueError(f"{path} holds {[W[x] for x in key]} for {list(key)}, not {list(key.values())}")
    for agg in AGGS:
        W.setdefault(agg, {})
    return W, path


def reference(toy, a, device="cpu"):
    """(Ce, reference statistics) of the test tasks"""
    _, Ce = make_testset(toy, a.n_test)
    Ce = Ce.to(device)
    ref_C, diag = get_reference(toy, Ce, a.n_ref, f"{paths.res_dir(toy.name)}/{toy.name}_ref.pt")
    if "rhat" in diag:
        print("  reference R-hat (max over tasks):", {k: round(v["max"], 3) for k, v in diag["rhat"].items()}, flush=True)
    return Ce, stats_np(ref_C, Ce)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    common_args(ap)
    ap.add_argument("--seed", type=int, default=0, help="toy_train.py's --seed (IW-ML fit ensemble)")
    ap.add_argument("--ref_only", action="store_true", help="only build the reference cache {toy}_ref.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for name in select_toys(a.toys, a.data):
        toy = make_toy(name); t0 = time.time()
        print(f"--- {name}", flush=True)
        if a.ref_only:
            reference(toy, a); continue
        W, path = open_w1(toy, a)
        todo = [m for m in ("oracle", "iw_ml") if not all(m in W[agg] for agg in AGGS)]
        if not todo:
            continue
        Ce, ref_stats = reference(toy, a)
        for m in todo:
            s = w1_oracle_floor(ref_stats) if m == "oracle" else w1_iwml(iwml_w1(toy, a.seed), Ce, a.J, ref_stats)
            for agg in AGGS:
                W[agg][m] = s[agg]
            json.dump(W, open(path, "w"), indent=1)
            print(f"  {m}: mean {s['mean']['value']:.4f}  median {s['median']['value']:.4f}", flush=True)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
