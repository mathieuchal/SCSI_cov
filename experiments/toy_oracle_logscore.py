"""Predictive log-score of the *oracle* posterior (exact IW conjugate / HMC for the factor model) and of the fitted IW baseline,
on the very validation pairs (Ce_cal, Ce_val) that toy_train.py scores SC-SI checkpoints on. Reproduces toy_train.py's data
generation exactly (same generator seed and draw order), so the numbers are directly comparable to results/toys/{toy}_train.json.

Scored like toy_train.py's callback: log (1/J) sum_j p(Ce_val | C_j) averaged over the pairs, with J = 64 posterior draws
(J = 256 also stored: the log-mean-exp of fewer draws is biased low, so J must match when comparing curves)."""
import argparse, json
import torch

from scsi import wishart_channel
from covutils import IWPrior, posterior_predictive_logscore, wishart_logscore
from toys import make_toy

ap = argparse.ArgumentParser()
ap.add_argument("--toy", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_val", type=int, default=300)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = torch.device(a.device)
toy = make_toy(a.toy, device=dev)
N = toy.N
g = torch.Generator(device=dev).manual_seed(1000 + a.seed)                                   # identical to toy_train.py
C_tr = toy.sample_prior(toy.M, g); Ce_tr = wishart_channel(C_tr, N, g)
C_va = toy.sample_prior(a.n_val, g); Ce_cal = wishart_channel(C_va, N, g); Ce_val = wishart_channel(C_va, N, g)

out = {"toy": toy.name, "n_val": a.n_val, "exact_reference": bool(toy.exact)}
gg = torch.Generator(device=dev).manual_seed(4321)
iw = IWPrior(Ce_tr, N, iters=800)
for J in (64, 256):
    out[f"oracle_J{J}"] = float(posterior_predictive_logscore(Ce_val, N, toy.ref_posterior(Ce_cal, J, gg)).mean())
    out[f"iw_ml_J{J}"] = float(posterior_predictive_logscore(Ce_val, N, iw.sample_posterior(Ce_cal, J, gen=gg)).mean())
out["plugin_scm"] = float(wishart_logscore(Ce_val, N, Ce_cal).mean())
json.dump(out, open(f"results/toys/{toy.name}_oracle_logscore.json", "w"), indent=1)
print(out)
