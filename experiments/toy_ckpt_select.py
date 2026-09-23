"""Ground-truth-free checkpoint-selection proxy: worst-case PIT-KS over a small set of scalar/directional
functionals, evaluated via posterior predictive checks on held-out (Ce_cal, Ce_val) pairs -- exactly the
split-half protocol available on real data (no access to the latent C or an exact reference posterior).

Motivation: toy_train.py's existing criterion (aggregate posterior-predictive log-score of Ce_val given
posterior draws from Ce_cal) picks checkpoints that are near-worst-possible on logcond/top_share/ldv_top
(see iw_ri_d20 and iw_ri_d20_M8000 sweeps: the log-score keeps improving well past the point where those
three statistics peak, trading them off against ldv_bot/logdet). This script tests whether a worst-case
per-statistic PIT-KS criterion -- computable with the same information the log-score already uses -- tracks
the *true* W1-to-reference distances (only available here because these are controlled toys) well enough to
recover the good early checkpoints instead of the badly-miscalibrated late ones.

Each functional's PIT is the rank of the observed value (on the real held-out Ce_val, or a fixed direction
from Ce_cal) within the posterior-predictive distribution obtained by pushing posterior draws C ~ model(.|
Ce_cal) through the *known* forward Wishart(N) channel -- a standard, ground-truth-free posterior predictive
check (Cook-Gelman-Rubin / SBC in spirit), using covutils.functional_pit / directional_pit."""
import argparse
import numpy as np
import torch

from scsi import DT, wishart_channel, safe_eigvalsh
from covutils import functional_pit, directional_pit, posterior_predictive_logscore
from toys import make_toy
from toy_eval import load_scsi, ks_unif

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


def f_logdet(Ce):
    return torch.linalg.slogdet(Ce)[1]


def f_logcond(Ce):
    w = safe_eigvalsh(Ce)
    return torch.log(w[..., -1] / w[..., 0].clamp_min(1e-300))


def f_top_share(Ce):
    w = safe_eigvalsh(Ce)
    return w[..., -1] / w.sum(-1)


def f_corr01(Ce):
    return Ce[..., 0, 1] / torch.sqrt(Ce[..., 0, 0] * Ce[..., 1, 1])


FUNCS = {"logdet": f_logdet, "logcond": f_logcond, "top_share": f_top_share, "corr01": f_corr01}

w_cal, V_cal = torch.linalg.eigh(Ce_cal)
v_top, v_bot = V_cal[..., -1].unsqueeze(1), V_cal[..., 0].unsqueeze(1)   # (M,1,d), from the OTHER observation

rows = []
for k in [int(x) for x in a.ks.split(",")]:
    model, kk = load_scsi(toy, f"results/toys/{toy.name}_k{k}.pt", a.threads, device=dev)
    assert kk == k
    gg = torch.Generator(device=dev).manual_seed(a.seed + 1)
    Cs = model.sample_posterior(Ce_cal, a.J, gen=gg)
    logscore = float(posterior_predictive_logscore(Ce_val, N, Cs).mean())
    ks = {}
    for name, fn in FUNCS.items():
        pit, _, _ = functional_pit(Ce_val, N, Cs, fn, gen=gg)
        ks[name] = ks_unif(pit.cpu().numpy())
    pit_top, _, _ = directional_pit(None, Ce_val, N, Cs, v_top, gen=gg)
    pit_bot, _, _ = directional_pit(None, Ce_val, N, Cs, v_bot, gen=gg)
    ks["ldv_top"] = ks_unif(pit_top.squeeze(-1).cpu().numpy())
    ks["ldv_bot"] = ks_unif(pit_bot.squeeze(-1).cpu().numpy())
    worst = max(ks.values())
    rows.append((k, logscore, worst, ks))
    print(f"k={k:4d}  logscore={logscore:9.3f}  worst_ks={worst:.3f}  "
          + "  ".join(f"{n}={v:.3f}" for n, v in ks.items()), flush=True)

best_logscore = max(rows, key=lambda r: r[1])
best_worstks = min(rows, key=lambda r: r[2])
print(f"\nBy aggregate log-score: k={best_logscore[0]} (logscore={best_logscore[1]:.3f}, worst_ks={best_logscore[2]:.3f})")
print(f"By worst-case PIT-KS:   k={best_worstks[0]} (worst_ks={best_worstks[2]:.3f}, logscore={best_worstks[1]:.3f})")
