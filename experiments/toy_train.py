"""Train SC-SI on a toy ensemble (noisy Ce only), select the outer iteration on validation pairs, save the model.

Validation pairs (Ce_cal, Ce_val) share a latent C (as the split-half pairs do in the real-data protocol); the criterion is the
posterior predictive log-score of Ce_val given Ce_cal.  No clean covariance is given to the learner."""
import argparse, json, os, time
import numpy as np, torch
from scsi import SCSI, SCSIConfig, Sym, wishart_channel, DT
from covutils import posterior_predictive_logscore
from toys import make_toy

ap = argparse.ArgumentParser()
ap.add_argument("--toy", required=True)
ap.add_argument("--n_outer", type=int, default=12)
ap.add_argument("--steps_first", type=int, default=4000)
ap.add_argument("--steps_outer", type=int, default=800)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_val", type=int, default=300)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
toy = make_toy(a.toy, device=dev)
d, N = toy.d, toy.N
g = torch.Generator(device=dev).manual_seed(1000 + a.seed)
C_tr = toy.sample_prior(toy.M, g); Ce_tr = wishart_channel(C_tr, N, g)                    # the ONLY training data: noisy Ce
C_va = toy.sample_prior(a.n_val, g); Ce_cal = wishart_channel(C_va, N, g); Ce_val = wishart_channel(C_va, N, g)
cfg = SCSIConfig(N=N, d=d, kappa=1.0, n_outer=a.n_outer, steps_first=a.steps_first, steps_outer=a.steps_outer, seed=a.seed,
                 init="deconv", log_prior_inflate=1.5, threads=a.threads, n_recon=8, device=str(dev))
model = SCSI(cfg)
print(f"toy={a.toy} device={dev} d={d} N={N} M={toy.M}", flush=True)
CK = [k for k in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30) if k <= a.n_outer]
os.makedirs("results/toys", exist_ok=True)
val, best = {}, (None, -1e30)


def _save(net, path):
    # checkpoints are stored on CPU regardless of training device, so they load anywhere
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()},
                "mu": net.mu.cpu().clone(), "sd": net.sd.cpu().clone(), "N": N, "d": d}, path)


def cb(m, k):
    global best
    if k in CK:
        Cs = m.sample_posterior(Ce_cal, 64)
        val[k] = float(posterior_predictive_logscore(Ce_val, N, Cs).mean())
        print(f"   [val] outer {k}: predictive log-score = {val[k]:.3f}", flush=True)
        _save(m.ema_net, f"results/toys/{toy.name}_k{k}.pt")   # every checkpoint (trajectory analysis)
        if val[k] > best[1]:
            best = (k, val[k])
            _save(m.ema_net, f"results/toys/{toy.name}_model.pt")

t0 = time.time()
model.fit(Ce_tr, callback=cb)
json.dump({"val_curve": val, "best_k": best[0], "fit_sec": time.time() - t0, "M": toy.M, "N": N, "d": d}, open(f"results/toys/{toy.name}_train.json", "w"), indent=1)
print(f"{toy.name}: selected outer iteration {best[0]}  (val curve {val})  fit {time.time()-t0:.0f}s")
