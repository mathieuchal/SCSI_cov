"""EEG split-half calibration test (common-C protocol).

Calibration half A and validation half B are made of interleaved 1-s chunks of the SAME 8-s epoch, so both are
independent-ish Wishart(N) draws around the same local covariance (no between-window drift).  We test the
posterior predictive law of B given A for directional variances and for nonlinear functionals of C.
Uses the SC-SI checkpoint / IW baseline / subject split from eeg_exp.py (test subjects were never seen).
"""
import argparse, json, math, sys
import numpy as np, torch

from scsi import SCSI, SCSIConfig, Sym, Drift, wishart_channel, DT, safe_eigvalsh
from covutils import (IWPrior, oas_shrink, nls_shrink, wishart_logscore, posterior_predictive_logscore, directional_pit,
                      coverage_from_pit, airm_distance, bootstrap_ci)

ap = argparse.ArgumentParser()
ap.add_argument("tag", nargs="?", default="eeg_w4_s0_k1.0_e1.0")
ap.add_argument("kbest", nargs="?", type=int, default=16)
ap.add_argument("neff_scale", nargs="?", type=float, default=1.0)
ap.add_argument("seed", nargs="?", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
TAG, KBEST, NEFF_SCALE, SEED = a.tag, a.kbest, a.neff_scale, a.seed
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
z = np.load("data/eeg_scm_4s.npz")
scm_win = torch.tensor(z["scm_win"], dtype=DT, device=dev); scm_il = torch.tensor(z["scm_il"], dtype=DT, device=dev)   # (S,2,7,2,d,d)
Sn, _, W, d, _ = scm_win.shape
N = int(round(float(np.median(z["neff_ratio"])) * int(z["n_win"]) * NEFF_SCALE))
rng = np.random.default_rng(SEED); perm = rng.permutation(Sn)
n_tr, n_va = int(0.65 * Sn), int(0.10 * Sn)
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
Ce_train = scm_win[tr].reshape(-1, d, d)
S_ = Sym(d, device=dev)
cfg = SCSIConfig(N=N, d=d, kappa=1.0, init="deconv", log_prior_inflate=1.5, threads=a.threads, device=str(dev))
model = SCSI(cfg)
Ye = S_.encode(Ce_train, 0.0)
model.ema_net = Drift(S_.p, Ye.mean(0), Ye.std(0).clamp_min(1e-3), cfg.hidden, cfg.depth).to(dev)
model.ema_net.load_state_dict(torch.load(f"results/{TAG}_ckpt/k{KBEST}.pt", map_location="cpu"))
iw = IWPrior(Ce_train, N, iters=800)
print(f"device={dev}  N={N} d={d}; test subjects={len(te)}; SC-SI checkpoint k={KBEST}")

J = 256
gen = torch.Generator(device=dev).manual_seed(4321)
# cases: (subject, cond, epoch, direction) with cal = half h, val = half 1-h
cal, val, meta = [], [], []
for s in te:
    for c in range(2):
        for e in range(7):
            for h in range(2):
                cal.append(scm_il[s, c, e, h]); val.append(scm_il[s, c, e, 1 - h]); meta.append((int(s), c, e, h))
Cal, Val, meta = torch.stack(cal), torch.stack(val), np.array(meta)
subj = meta[:, 0]
D_si = model.sample_posterior(Cal, J, gen=gen)
D_iw = iw.sample_posterior(Cal, J, gen=gen)
plug = {"Sample cov.": Cal, "Linear shrinkage (OAS)": oas_shrink(Cal, N), "Nonlinear shrinkage (LW)": nls_shrink(Cal, N)}
post = {"IW conjugate (ML-fitted)": D_iw, "SC-SI (ours)": D_si}
allm = {**{k: v[:, None].expand(-1, J, -1, -1) for k, v in plug.items()}, **post}

res = {"N": N, "n_cases": int(len(Cal)), "n_test_subjects": int(len(te))}
# ---- log-score and directional variances ---- #
Cref = Ce_train.mean(0)
ev, evec = torch.linalg.eigh(Cref)
dirs = torch.cat([torch.eye(d, dtype=DT, device=dev), evec[:, -3:].T], 0)
res["logscore"], res["dir_cov"], res["dir_cov_by_dir"] = {}, {}, {}
sc = {}
for name, Cs in allm.items():
    sc[name] = (wishart_logscore(Val, N, plug[name]) if name in plug else posterior_predictive_logscore(Val, N, Cs)).cpu().numpy()
    pit, _, _ = directional_pit(None, Val, N, Cs, dirs, gen=gen)
    res["dir_cov"][name] = coverage_from_pit(pit)
    p = pit.cpu().numpy()
    res["dir_cov_by_dir"][name] = {str(i): float(((p[:, i] > .1) & (p[:, i] < .9)).mean()) for i in range(p.shape[1])}
for name in sc:
    res["logscore"][name] = dict(mean=bootstrap_ci(sc[name], subj), gain=bootstrap_ci(sc[name] - sc["Sample cov."], subj))
    print(f"{name:28s} logscore={res['logscore'][name]['mean'][0]:.2f} gain={res['logscore'][name]['gain'][0]:+.2f} "
          f"[{res['logscore'][name]['gain'][1]:+.2f},{res['logscore'][name]['gain'][2]:+.2f}]  dir-cov(50/80/95)="
          + "/".join(f"{v:.3f}" for v in res["dir_cov"][name].values()))

res["logscore_paired"] = {}
for a_, b_ in (("SC-SI (ours)", "Nonlinear shrinkage (LW)"), ("SC-SI (ours)", "IW conjugate (ML-fitted)"), ("Nonlinear shrinkage (LW)", "Sample cov.")):
    res["logscore_paired"][f"{a_} - {b_}"] = bootstrap_ci(sc[a_] - sc[b_], subj)
    print(f"paired log-score  {a_} - {b_}: {res['logscore_paired'][f'{a_} - {b_}'][0]:+.2f} [{res['logscore_paired'][f'{a_} - {b_}'][1]:+.2f},{res['logscore_paired'][f'{a_} - {b_}'][2]:+.2f}]")

# ---- nonlinear functionals of C: posterior-predictive PIT ---- #
oz = 6
def f_logdet(C): return torch.linalg.slogdet(C)[1]
def f_logoz(C): return torch.log(C[..., oz, oz])
def f_top_share(C):
    w = safe_eigvalsh(C); return w[..., -1] / w.sum(-1)     # C is (n_cases,J,d,d): can exceed cuSOLVER's batch limit
def f_cond(C):
    w = safe_eigvalsh(C); return torch.log(w[..., -1] / w[..., 0])
FUN = {"log det C": f_logdet, "log C[Oz,Oz]": f_logoz, "top-eigenvalue share": f_top_share, "log cond. number": f_cond}
res["functional"] = {}
for fn, f in FUN.items():
    obs = f(Val)
    res["functional"][fn] = {}
    for name, Cs in allm.items():
        Ce_pred = wishart_channel(Cs.reshape(-1, d, d), N, gen).reshape(Cs.shape)
        pred = f(Ce_pred)
        pit = ((pred < obs[:, None]).to(DT).mean(1) + 0.5 * (pred == obs[:, None]).to(DT).mean(1)).cpu().numpy()
        cov = coverage_from_pit(torch.tensor(pit))
        # estimation of the functional itself: posterior/plug-in point value vs held-out half's noisy value
        pt = f(Cs).median(1).values if name in post else f(Cs[:, 0])
        res["functional"][fn][name] = dict(cov=cov, ks=float(np.max(np.abs(np.sort(pit) - (np.arange(len(pit)) + .5) / len(pit)))),
                                           abs_err_to_val_half=bootstrap_ci((pt - obs).abs().cpu().numpy(), subj))
    print(f"\n{fn}: coverage 50/80/95 by method")
    for name in allm:
        r = res["functional"][fn][name]
        print(f"   {name:28s} " + "/".join(f"{v:.3f}" for v in r["cov"].values()) + f"   KS={r['ks']:.3f}")

# ---- open-vs-closed Riemannian distance (pair the same epoch/half index of the two runs) ---- #
rows = [(int(s), e, h) for s in te for e in range(7) for h in range(2)]
Ao = torch.stack([scm_il[s, 0, e, h] for s, e, h in rows]); Ac = torch.stack([scm_il[s, 1, e, h] for s, e, h in rows])
Bo = torch.stack([scm_il[s, 0, e, 1 - h] for s, e, h in rows]); Bc = torch.stack([scm_il[s, 1, e, 1 - h] for s, e, h in rows])
sub_r = np.array([r[0] for r in rows])
obs = airm_distance(Bo, Bc)
res["distance"] = {}
def draws_for(A, kind):
    if kind == "Sample cov.": return A[:, None].expand(-1, J, -1, -1)
    if kind == "Linear shrinkage (OAS)": return oas_shrink(A, N)[:, None].expand(-1, J, -1, -1)
    if kind == "Nonlinear shrinkage (LW)": return nls_shrink(A, N)[:, None].expand(-1, J, -1, -1)
    if kind == "IW conjugate (ML-fitted)": return iw.sample_posterior(A, J, gen=gen)
    return model.sample_posterior(A, J, gen=gen)
print("\nopen-vs-closed Riemannian distance: posterior predictive of the held-out half")
for name in allm:
    Co, Cc = draws_for(Ao, name), draws_for(Ac, name)
    po = wishart_channel(Co.reshape(-1, d, d), N, gen).reshape(Co.shape); pc = wishart_channel(Cc.reshape(-1, d, d), N, gen).reshape(Cc.shape)
    pred = airm_distance(po, pc)
    pit = ((pred < obs[:, None]).to(DT).mean(1) + 0.5 * (pred == obs[:, None]).to(DT).mean(1))
    cov = coverage_from_pit(pit)
    dist_pt = airm_distance(Co, Cc).median(1).values if name in post else airm_distance(Co[:, 0], Cc[:, 0])
    res["distance"][name] = dict(cov=cov, abs_err=bootstrap_ci((dist_pt - obs).abs().cpu().numpy(), sub_r),
                                 bias_to_val=bootstrap_ci((dist_pt - obs).cpu().numpy(), sub_r))
    print(f"   {name:28s} cov 50/80/95 = " + "/".join(f"{v:.3f}" for v in cov.values()) +
          f"   plug-in distance bias vs held-out half = {res['distance'][name]['bias_to_val'][0]:+.3f}")
json.dump(res, open(f"results/{TAG}_splithalf.json", "w"), indent=1, default=float)
np.savez(f"results/{TAG}_splithalf_arrays.npz", meta=meta, score_names=np.array(list(sc)), scores=np.stack([sc[k] for k in sc]))
print("saved")
