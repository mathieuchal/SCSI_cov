"""Train SC-SI on a toy ensemble (noisy Ce only), select the outer iteration on validation pairs, save the model.

Validation pairs (Ce_cal, Ce_val) share a latent C (as the split-half pairs do in the real-data protocol). The
checkpoint-selection criterion is the worst (max) PIT-KS-vs-uniform over a small set of scalar/directional
functionals (logdet, log condition number, top-eigenvalue share, corr01, directional variance along Ce_cal's
own top/bottom eigenvectors) -- see covutils.worst_case_pit_ks. The aggregate posterior-predictive log-score
is still recorded for comparison but no longer drives selection: on iw_ri_d20 it kept improving well past the
point where log condition number / top-eigenvalue share / top-directional-variance peaked, trading them off
against the bottom-directional-variance functional it happens to weight more heavily -- the worst-case
criterion instead refuses to sacrifice any one tracked functional for gains on another. No clean covariance is
given to the learner."""
import argparse, json, os, time
import numpy as np, torch
from scsi import SCSI, SCSIConfig, Sym, wishart_channel, DT
from covutils import posterior_predictive_logscore, worst_case_pit_ks
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
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--activation", default="silu", choices=["silu", "relu", "gelu", "tanh"])
ap.add_argument("--n_recon", type=int, default=8, help="reconstructions per training-set covariance per outer iteration (bank size = M * n_recon)")
ap.add_argument("--tag", default="", help="suffix for this run's output files (train/model/checkpoints), to keep several training variants of one toy")
ap.add_argument("--n_sde_steps", type=int, default=64, help="Follmer SDE integration steps, used both for E-step reconstruction during EM and (via the checkpoint) at eval time")
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
toy = make_toy(a.toy, device=dev)
stem = toy.name + (f"_{a.tag}" if a.tag else "")
d, N = toy.d, toy.N
g = torch.Generator(device=dev).manual_seed(1000 + a.seed)
C_tr = toy.sample_prior(toy.M, g); Ce_tr = wishart_channel(C_tr, N, g)                    # the ONLY training data: noisy Ce
C_va = toy.sample_prior(a.n_val, g); Ce_cal = wishart_channel(C_va, N, g); Ce_val = wishart_channel(C_va, N, g)
cfg = SCSIConfig(N=N, d=d, kappa=1.0, n_outer=a.n_outer, steps_first=a.steps_first, steps_outer=a.steps_outer, seed=a.seed,
                 init="deconv", log_prior_inflate=1.5, threads=a.threads, n_recon=a.n_recon, n_sde_steps=a.n_sde_steps, device=str(dev),
                 hidden=a.hidden, depth=a.depth, activation=a.activation)
model = SCSI(cfg)
print(f"toy={a.toy} device={dev} d={d} N={N} M={toy.M}  net: hidden={a.hidden} depth={a.depth} activation={a.activation}  "
      f"n_recon={a.n_recon} (bank={toy.M*a.n_recon})  n_sde_steps={a.n_sde_steps}", flush=True)
CK = [k for k in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300) if k <= a.n_outer]
os.makedirs("results/toys", exist_ok=True)
val, val_wc, best = {}, {}, (None, 1e30)


def _save(net, k, path):
    # checkpoints are stored on CPU regardless of training device, so they load anywhere; architecture
    # (hidden/depth/activation) travels with the checkpoint so toy_eval.py can reconstruct the right shape
    # regardless of what defaults it would otherwise assume.
    torch.save({"state": {kk: v.cpu() for kk, v in net.state_dict().items()},
                "mu": net.mu.cpu().clone(), "sd": net.sd.cpu().clone(), "k": k, "N": N, "d": d,
                "hidden": a.hidden, "depth": a.depth, "activation": a.activation, "n_sde_steps": a.n_sde_steps}, path)


def cb(m, k):
    global best
    if k in CK:
        Cs = m.sample_posterior(Ce_cal, 64)
        val[k] = float(posterior_predictive_logscore(Ce_val, N, Cs).mean())
        wc, ks = worst_case_pit_ks(Ce_val, N, Cs, Ce_cal, gen=m.gen)
        val_wc[k] = wc
        print(f"   [val] outer {k}: predictive log-score = {val[k]:.3f}  worst_ks = {wc:.3f}  ("
              + ", ".join(f"{n}={v:.3f}" for n, v in ks.items()) + ")", flush=True)
        _save(m.ema_net, k, f"results/toys/{stem}_k{k}.pt")   # every checkpoint (trajectory analysis)
        if wc < best[1]:
            best = (k, wc)
            _save(m.ema_net, k, f"results/toys/{stem}_model.pt")

t0 = time.time()
model.fit(Ce_tr, callback=cb)
json.dump({"val_curve": val, "val_curve_worst_ks": val_wc, "best_k": best[0], "fit_sec": time.time() - t0,
          "M": toy.M, "N": N, "d": d, "hidden": a.hidden, "depth": a.depth, "activation": a.activation,
          "n_recon": a.n_recon, "n_sde_steps": a.n_sde_steps},
          open(f"results/toys/{stem}_train.json", "w"), indent=1)
print(f"{toy.name}: selected outer iteration {best[0]} by worst-case PIT-KS={best[1]:.3f}  "
      f"(val curve {val})  fit {time.time()-t0:.0f}s")
