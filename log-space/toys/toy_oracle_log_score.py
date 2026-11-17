"""Predictive log-score, at large J, of the oracle posterior, of the ML-fitted IW baseline and of SC-SI checkpoints,
on validation pairs (Ce_cal, Ce_val) that share a latent C -- the same protocol as toy_train.py's callback, whose
data generation (generator seed and draw order) is reproduced exactly.

Score of a posterior sampler q:  mean over pairs of  log (1/J) sum_j p(Ce_val | C_j),  C_j ~ q(. | Ce_cal),
with p the Wishart channel up to C-independent terms (covutils.wishart_logscore, which takes the full covariance C).
The log-mean-exp of J draws is biased low, by an amount that depends on the sampler, so every method is scored
at the SAME (large) J on the SAME pairs.  For the conjugate IW toys (and for the IW-ML baseline) the predictive
density is also available in closed form (matrix-variate beta-II), stored as *_exact: a direct check of the MC bias.

Log-space version: SC-SI is a generative model of Y = log C (scsi.Sym), sample_posterior returns C directly,
toy.ref_posterior / toy.hmc return C, so no Cholesky factor appears anywhere.

    python toy_oracle_log_score.py                                 # all d=8 toys, production settings (J=2e6, 10 pairs,
                                                                   # 2000 HMC chains, the checkpoint grid)
    python toy_oracle_log_score.py --toys hetero_d8_inf --ks 100,500,1000
Writes
  results/toys/{toy}_logscore.json         the toy:  toy, n_val, J, N, d, and per method {mean, se, per_task, ess, sec}:
                                           oracle (+ diag of its sampler), oracle_exact (closed form, IW toys only),
                                           iw_ml, scsi (the production run, its "tag", at its best_k: "k")
  results/toys/{stem}_train.json           the SC-SI run:  "val_curve": {k: mean log-score} (the trajectory),
                                           "val_curve_n_val": n_val, "val_curve_J": J
best_k is the run's own (toy_train.py: the last checkpoint with --last_checkpoint, imposed in infinite-data mode; else the best callback validation).
Re-running resumes: checkpoints already in val_curve are skipped. best_k (the checkpoint selected by toy_train.py)
is always scored. One {toy}_logscore.json holds one (J, n_val): another setting raises.
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scsi import DT, wishart_channel
from covutils import IWPrior, wishart_logscore, iw_marginal_logpdf, log_mvgamma
from toys import make_toy, IWToy
from toy_eval import load_scsi

def load_train(stem):
    """{stem}_train.json; a file written before the val_curve -> val_curve_orig rename holds the toy_train callback
    curve under "val_curve": renamed here (that callback used toy_train's defaults n_val=300, J=64)."""
    tr = json.load(open(f"results/toys/{stem}_train.json"))
    if "val_curve" in tr and "val_curve_orig" not in tr:
        tr = {**{"val_curve_orig": tr.pop("val_curve"), "val_curve_orig_n_val": 300, "val_curve_orig_J": 64}, **tr}
    return tr


# d=8 runs: toy -> training stem tag (checkpoints results/toys/{toy}_{tag}_k{k}.pt)
TAGS = {"iw_ri_d8_inf": "mlp_w512_d4", "iw_nonri_d8_inf": "mlp_w512_d4", "factor_d8_inf": "mlp_w512_d4",
        "hetero_d8_inf": "mlp_w512_d4", "diag_d8_inf": "mlp_w512_d4", "iso_diag_d8_inf": "mlp_w1024_d4",
        "ar1_d8_inf": "resnet_w1024_d4"}


def mc_logscore(Ce_val, N, Cs, device, chunk=50000):
    """Per pair: log (1/J) sum_j p(Ce_val|C_j) and the importance ESS of the weights p(Ce_val|C_j).
    Ce_val (M,d,d); Cs (M,J,d,d) on any device.  Returns two (M,) CPU tensors."""
    M, J = Cs.shape[:2]
    ls, ess = torch.empty(M, dtype=DT), torch.empty(M, dtype=DT)
    for i in range(M):
        cv = Ce_val[i:i + 1].to(device, DT)
        s = torch.cat([wishart_logscore(cv, N, Cs[i:i + 1, j:j + chunk].to(device, DT))[0] for j in range(0, J, chunk)])
        lse = torch.logsumexp(s, 0)
        ls[i] = (lse - math.log(J)).cpu()
        ess[i] = torch.exp(2 * lse - torch.logsumexp(2 * s, 0)).cpu()
    return ls, ess


def iw_exact_logscore(Ce_val, Ce_cal, N, nu0, Psi0):
    """Closed-form log p(Ce_val | Ce_cal) under an IW(nu0, Psi0) prior, on wishart_logscore's scale (i.e. dropping
    the same C-independent terms (N-d-1)/2 log|S| - Nd/2 log 2 - log Gamma_d(N/2), S = N Ce_val)."""
    d = Ce_val.shape[-1]
    Ce_val, Ce_cal = Ce_val.to(DT), Ce_cal.to(DT)
    nu, Psi = nu0 + N, Psi0.to(Ce_cal.device, DT) + N * Ce_cal
    ld_S = torch.linalg.slogdet(N * Ce_val)[1]
    return (iw_marginal_logpdf(Ce_val, N, nu, Psi) - (N - d - 1) / 2 * ld_S + N * d / 2 * math.log(2.0)
            + log_mvgamma(N / 2, d).to(Ce_val.device)).cpu()


def summary(x):
    x = x.double()
    return {"mean": float(x.mean()), "se": float(x.std() / math.sqrt(len(x))), "per_task": [float(v) for v in x]}


ap = argparse.ArgumentParser()
ap.add_argument("--toys", default=",".join(TAGS), help="comma-separated toy names")
ap.add_argument("--J", type=int, default=2000000, help="posterior draws per validation pair, for every method")
ap.add_argument("--n_val", type=int, default=10)
ap.add_argument("--seed", type=int, default=0, help="toy_train.py's --seed (validation pairs from Generator(1000+seed))")
ap.add_argument("--m_inf", type=int, default=5000, help="IW-ML fit ensemble size (toy_train.py's infinite-mode batch)")
ap.add_argument("--ks", default="grid", help="checkpoints to score: 'grid' (default: the production grid "
                 "2..20, 24, 60, 100..980 by 40, 1000, 1100..2000 by 100, restricted to the checkpoints on disk), "
                 "'last' (final checkpoint of the run), 'all' (every checkpoint on disk), a comma list, or "
                 "'auto' = 20, 100..k_max by 100, + best_k")
ap.add_argument("--no_iw", action="store_true", help="skip the IW-ML baseline (oracle and SC-SI only)")
ap.add_argument("--hmc_chains", type=int, default=2000, help="HMC chains per pair for the non-conjugate oracles")
ap.add_argument("--warmup", type=int, default=1500)
ap.add_argument("--chunk", type=int, default=131072, help="SC-SI SDE batch size")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for name in a.toys.split(","):
    tag = TAGS[name]
    stem = f"{name}_{tag}"
    out_path = f"results/toys/{name}_logscore.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if out and (out["J"], out["n_val"]) != (a.J, a.n_val):
        raise ValueError(f"{out_path} holds J={out['J']}, n_val={out['n_val']}, not J={a.J}, n_val={a.n_val}")
    toy = make_toy(name)
    N, d = toy.N, toy.d
    t_toy0 = time.time()
    print(f"--- {name} ({stem}) ---", flush=True)

    # validation pairs exactly as toy_train.py draws them (infinite mode: nothing drawn before C_val)
    g = torch.Generator().manual_seed(1000 + a.seed)
    assert toy.M == -1, "finite-ensemble toys would need toy_train.py's C_tr draw first"
    C_va = toy.sample_prior(a.n_val, g)
    Ce_cal = wishart_channel(C_va, N, g).to(device)
    Ce_val = wishart_channel(C_va, N, g).to(device)
    Ce_tr = wishart_channel(toy.sample_prior(a.m_inf, g), N, g).to(device)       # IW-ML fit set, drawn after C_va
    del C_va
    out = {"toy": name, "n_val": a.n_val, "J": a.J, "N": N, "d": d, **out}

    if "oracle" not in out:
        gg = torch.Generator().manual_seed(4321)
        t0 = time.time()
        if toy.exact:                                                              # IW conjugate / AR(1) grid: direct draws,
            per = [mc_logscore(Ce_val[i:i + 1], N, toy.ref_posterior(Ce_cal[i:i + 1], a.J, gg), device)   # one pair at a time
                   for i in range(a.n_val)]                                        # (J up to ~1e7 fits in GPU memory)
            Cs = None
            diag = {"sampler": "direct"}
        else:                                                                      # HMC, many short chains per pair
            nk = int(math.ceil(a.J / a.hmc_chains))
            Cs, diag = toy.hmc(Ce_cal, n_chains=a.hmc_chains, n_keep=nk, thin=4, warmup=a.warmup, gen=gg, return_diag=True)
            if Cs.shape[1] > a.J:                                                  # all draws used otherwise: no shuffle copy
                Cs = Cs[:, torch.randperm(Cs.shape[1], generator=gg).to(Cs.device)][:, :a.J]
            diag.update(sampler="hmc", warmup=a.warmup)
        if Cs is None:
            ls, ess = torch.cat([p[0] for p in per]), torch.cat([p[1] for p in per])
        else:
            ls, ess = mc_logscore(Ce_val, N, Cs, device)
        del Cs
        out["oracle"] = summary(ls) | {"ess": [float(v) for v in ess], "sec": time.time() - t0, "diag": diag}
        if isinstance(toy, IWToy):                                                 # conjugate: exact predictive too
            out["oracle_exact"] = summary(iw_exact_logscore(Ce_val, Ce_cal, N, toy.nu0, toy.Psi0))
        print(f"oracle {out['oracle']['mean']:.3f} +- {out['oracle']['se']:.3f}  (ESS min {min(out['oracle']['ess']):.0f}, "
              f"{out['oracle']['sec']:.0f}s)" + (f"  exact {out['oracle_exact']['mean']:.3f}" if "oracle_exact" in out else ""),
              flush=True)
        json.dump(out, open(out_path, "w"), indent=1)

    if "iw_ml" not in out and not a.no_iw:
        t0 = time.time()
        iw = IWPrior(Ce_tr, N, iters=800)
        gi = torch.Generator().manual_seed(4322)
        per = [mc_logscore(Ce_val[i:i + 1], N, iw.sample_posterior(Ce_cal[i:i + 1], a.J, gen=gi), device)   # one pair at a time
               for i in range(a.n_val)]
        ls, ess = torch.cat([p[0] for p in per]), torch.cat([p[1] for p in per])
        out["iw_ml"] = summary(ls) | {"ess": [float(v) for v in ess], "sec": time.time() - t0}
        print(f"iw_ml {out['iw_ml']['mean']:.3f}", flush=True)
        json.dump(out, open(out_path, "w"), indent=1)

    tr = load_train(stem)
    if tr.get("val_curve_J", a.J) != a.J or tr.get("val_curve_n_val", a.n_val) != a.n_val:
        raise ValueError(f"{stem}_train.json: val_curve was computed with J={tr['val_curve_J']}, "
                         f"n_val={tr['val_curve_n_val']}, not J={a.J}, n_val={a.n_val}")
    vc = {int(k): v for k, v in tr.get("val_curve", {}).items()}

    def save_train():                                          # SC-SI trajectory, sorted by k
        tr["val_curve"] = {str(k): vc[k] for k in sorted(vc)}
        tr["val_curve_n_val"], tr["val_curve_J"] = a.n_val, a.J
        tr.pop("log_score_path", None)                         # older key: = val_curve[best_k]
        json.dump(tr, open(f"results/toys/{stem}_train.json", "w"), indent=1)

    if a.ks == "grid":
        GRID = [2, 4, 6, 8, 10, 12, 16, 20, 24, 60, *range(100, 1000, 40), 1000, *range(1100, 2001, 100)]
        ks = [k for k in GRID if os.path.exists(f"results/toys/{stem}_k{k}.pt")]
    elif a.ks == "all":                                                              # every checkpoint saved on disk
        import re
        ks = sorted(int(mm.group(1)) for f in os.listdir("results/toys") if (mm := re.fullmatch(rf"{stem}_k(\d+)\.pt", f)))
    elif a.ks == "last":
        ks = [max(int(k) for k in tr["val_curve_orig"])]
    elif a.ks == "auto":
        kmax = max(int(k) for k in tr["val_curve_orig"])
        ks = sorted({20, *range(100, kmax + 1, 100), tr["best_k"]})
    else:
        ks = [int(k) for k in a.ks.split(",")]
    ks = sorted({*ks, tr["best_k"]})                                               # best_k always scored (scsi entry)
    for k in ks:
        scsi_ok = (out.get("scsi", {}).get("tag"), out.get("scsi", {}).get("k")) == (tag, k)
        if k in vc and not (k == tr["best_k"] and not scsi_ok):                    # best_k: full scsi entry needed
            continue
        path = f"results/toys/{stem}_k{k}.pt"
        if not os.path.exists(path):
            print(f"  missing {path}", flush=True); continue
        t0 = time.time()
        m, _ = load_scsi(toy, path, a.threads)
        gm = torch.Generator(device=m.device).manual_seed(777)
        with torch.no_grad():                                                      # sample + score one pair at a time
            # draws in blocks of <= 200000: cusolver's batched eigh (Sym.decode) rejects ~2e6-matrix batches
            draw = lambda i: torch.cat([m.sample_posterior(Ce_cal[i:i + 1], min(200000, a.J - j), gen=gm, chunk=a.chunk)
                                        for j in range(0, a.J, 200000)], 1)
            per = [mc_logscore(Ce_val[i:i + 1], N, draw(i), device) for i in range(a.n_val)]
        ls, ess = torch.cat([p[0] for p in per]), torch.cat([p[1] for p in per])
        del m
        vc[k] = float(ls.mean())
        if k == tr["best_k"]:
            out["scsi"] = {"tag": tag, "k": k} | summary(ls) | {"ess": [float(v) for v in ess], "sec": time.time() - t0}
            json.dump(out, open(out_path, "w"), indent=1)
        print(f"  k={k:5d}  scsi {vc[k]:.3f}  (- oracle {vc[k] - out['oracle']['mean']:+.3f})  {time.time() - t0:.0f}s",
              flush=True)
        save_train()
    json.dump(out, open(out_path, "w"), indent=1)
    save_train()
    print(f"  {stem}_train.json: {len(vc)} checkpoints in val_curve; scsi at best_k={tr['best_k']}: "
          f"{out.get('scsi', {}).get('mean')}", flush=True)
    print(f"done {name} in {time.time() - t_toy0:.0f}s", flush=True)
