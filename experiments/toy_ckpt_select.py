"""Standalone sweep of covutils.worst_case_pit_ks (the ground-truth-free checkpoint-selection proxy now
wired into toy_train.py) across a toy's already-saved per-iteration checkpoints, printed alongside the
aggregate posterior-predictive log-score for comparison. Useful for post-hoc inspection of a completed run
without retraining; toy_train.py itself now selects on the worst-case criterion during training."""
import argparse
import torch

from scsi import wishart_channel
from covutils import posterior_predictive_logscore, worst_case_pit_ks
from toys import make_toy
from toy_eval import load_scsi

ap = argparse.ArgumentParser()
ap.add_argument("--toy", required=True)
ap.add_argument("--ks", required=True, help="comma-separated list of saved outer iterations to sweep")
ap.add_argument("--n_val", type=int, default=500)
ap.add_argument("--J", type=int, default=64)
ap.add_argument("--seed", type=int, default=5150)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
toy = make_toy(a.toy, device=dev)
N = toy.N

g = torch.Generator(device=dev).manual_seed(a.seed)               # independent of any training-time seed
C_va = toy.sample_prior(a.n_val, g)
Ce_cal = wishart_channel(C_va, N, g)
Ce_val = wishart_channel(C_va, N, g)

rows = []
for k in [int(x) for x in a.ks.split(",")]:
    model, kk = load_scsi(toy, f"results/toys/{toy.name}_k{k}.pt", a.threads, device=dev)
    assert kk == k
    gg = torch.Generator(device=dev).manual_seed(a.seed + 1)
    Cs = model.sample_posterior(Ce_cal, a.J, gen=gg)
    logscore = float(posterior_predictive_logscore(Ce_val, N, Cs).mean())
    worst, ks = worst_case_pit_ks(Ce_val, N, Cs, Ce_cal, gen=gg)
    rows.append((k, logscore, worst, ks))
    print(f"k={k:4d}  logscore={logscore:9.3f}  worst_ks={worst:.3f}  "
          + "  ".join(f"{n}={v:.3f}" for n, v in ks.items()), flush=True)

best_logscore = max(rows, key=lambda r: r[1])
best_worstks = min(rows, key=lambda r: r[2])
print(f"\nBy aggregate log-score: k={best_logscore[0]} (logscore={best_logscore[1]:.3f}, worst_ks={best_logscore[2]:.3f})")
print(f"By worst-case PIT-KS:   k={best_worstks[0]} (worst_ks={best_worstks[2]:.3f}, logscore={best_worstks[1]:.3f})")
