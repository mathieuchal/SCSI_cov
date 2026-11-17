"""W1 trajectory of an SC-SI run: W1 distance to the oracle posterior at every checkpoint of the evaluation grid.

For each test task (toy_eval.make_testset: 128 tasks), the six posterior statistics STATS are computed on the oracle
reference draws (toy_eval.get_reference, cached in {toy}_ref.pt) and on SC-SI draws from checkpoint k. Per statistic:
W1(SC-SI, oracle) / sd(oracle) per task, aggregated over the tasks by the mean and by the median; the value at k is the
average over the statistics (toy_eval.w1_summary). Same test set, reference, seeds, TF32 setting and sampler chunk as
toy_eval.py, which evaluates a single checkpoint (the run's best_k) together with the oracle floor and IW-ML and
writes {toy}_W1.json.

Writes, in the run's results/toys_M=-1/{stem}_train.json:
  "w1_curve_mean": {k: value}, "w1_curve_median": {k: value}, "w1_curve_n_test", "w1_curve_n_ref", "w1_curve_J"
Re-running resumes: checkpoints already in the curves are skipped.

    python toy_W1_mean.py                                  # all d=8 production runs, the checkpoint grid
    python toy_W1_mean.py --toys hetero_d8_inf --ks 100,500,1000
"""
import argparse, gc, json, os, re, time
import torch
from toy_eval import make_testset, stats_np, get_reference, load_scsi, w1_summary, AGGS, TAGS, GRID
from toys import make_toy

R = "results/toys_M=-1"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toys", default=",".join(TAGS), help="comma-separated toy names (run = TAGS[toy])")
    ap.add_argument("--ks", default="grid", help="'grid' (evaluation grid, checkpoints on disk), 'all', or a comma list")
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_ref", type=int, default=2048)
    ap.add_argument("--J", type=int, default=1024, help="SC-SI posterior draws per test task")
    ap.add_argument("--threads", type=int, default=26)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in a.toys.split(","):
        tag = TAGS[name]; stem = f"{name}_{tag}"
        toy = make_toy(name)
        t_toy0 = time.time()
        print(f"--- {name} ({stem}) ---", flush=True)
        tr_path = f"{R}/{stem}_train.json"
        tr = json.load(open(tr_path))
        if tr.get("w1_curve_J") is not None and (tr["w1_curve_n_test"], tr["w1_curve_n_ref"], tr["w1_curve_J"]) != \
                (a.n_test, a.n_ref, a.J):
            raise ValueError(f"{tr_path}: w1 curves computed with another (n_test, n_ref, J)")
        curves = {agg: {int(k): v for k, v in tr.get(f"w1_curve_{agg}", {}).items()} for agg in AGGS}

        def save():
            for agg in AGGS:
                tr[f"w1_curve_{agg}"] = {str(k): curves[agg][k] for k in sorted(curves[agg])}
            tr["w1_curve_n_test"], tr["w1_curve_n_ref"], tr["w1_curve_J"] = a.n_test, a.n_ref, a.J
            json.dump(tr, open(tr_path, "w"), indent=1)

        on_disk = sorted(int(mm.group(1)) for f in os.listdir(R) if (mm := re.fullmatch(rf"{stem}_k(\d+)\.pt", f)))
        ks = [k for k in GRID if k in on_disk] if a.ks == "grid" else on_disk if a.ks == "all" else \
            [int(k) for k in a.ks.split(",")]
        ks = [k for k in ks if not all(k in curves[agg] for agg in AGGS)]
        if ks:
            _, Ce = make_testset(toy, a.n_test)
            Ce = Ce.to(device)
            ref_C, _ = get_reference(toy, Ce, a.n_ref, f"{R}/{name}_ref.pt")
            ref_stats = stats_np(ref_C, Ce)
        for k in ks:
            t0 = time.time()
            m, _ = load_scsi(toy, f"{R}/{stem}_k{k}.pt", a.threads)
            torch.backends.cuda.matmul.allow_tf32 = True
            Cs = m.sample_posterior(Ce, a.J, gen=torch.Generator(device=m.device).manual_seed(99), chunk=131072)
            torch.backends.cuda.matmul.allow_tf32 = False
            s = w1_summary(stats_np(Cs, Ce), ref_stats)
            for agg in AGGS:
                curves[agg][k] = s[agg]["value"]
            print(f"  k={k:5d}  mean {s['mean']['value']:.4f}  median {s['median']['value']:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            del m, Cs
            gc.collect(); torch.cuda.empty_cache()
            save()
        save()
        print(f"done {name} in {time.time() - t_toy0:.0f}s", flush=True)
