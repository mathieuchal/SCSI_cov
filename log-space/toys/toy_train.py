"""Train SC-SI on a toy ensemble (noisy Ce only), select the outer iteration best_k, save the model.

best_k = the last checkpoint with --last_checkpoint, imposed in infinite-data mode (toy.M == -1, every d=8 production
run); the best-validation checkpoint otherwise.

Validation pairs (Ce_cal, Ce_val) share a latent C (as the split-half pairs do in the real-data protocol); the criterion is the
posterior predictive log-score of Ce_val given Ce_cal.  No clean covariance is given to the learner.

Defaults = the d=8 production runs (MLP-4 w512, seed 0, m_inf 5000, 26 threads), stage 1 of the two-stage protocol:
    stage 1: python toy_train.py --toy TOY --tag mlp_w512_d4                       # cosine outer lr 5e-4 -> 1e-5, k=0..700
    stage 2: python toy_train.py --toy TOY --tag mlp_w512_d4 --n_outer 1000 \
                 --resume results/toys_M=-1/TOY_mlp_w512_d4_k700.pt                       # constant lr 5e-5 (--resume_lr), k=700..1000
Exceptions: iso_diag_d8_inf uses --hidden 1024 (tag mlp_w1024_d4); ar1_d8_inf uses --arch resnet --hidden 1024
(tag resnet_w1024_d4) and a third stage, --n_outer 2000 --resume ..._k1000.pt (constant 5e-5, k=1000..2000).
On --resume the architecture is read from the checkpoint, so --hidden/--arch need not be repeated.

Outer lr schedule (SCSI._outer_lr0): the k-th outer refit runs its own inner cosine from a ceiling lr0_k down to
lr0_k*lr_end/lr. A fresh run anneals lr0_k by a cosine from lr*0.5 (k=1) to lr*0.5*lr_outer_floor_ratio (k=n_outer);
a resumed run holds lr0_k = resume_lr constant (warm restart: 5e-5 sits above the stage-1 floor 1e-5)."""
import argparse, json, os, time
import numpy as np, torch
from scsi import SCSI, SCSIConfig, Sym, wishart_channel, DT
from covutils import posterior_predictive_logscore
from toys import make_toy


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", required=True)
    ap.add_argument("--n_outer", type=int, default=700)
    ap.add_argument("--steps_first", type=int, default=4000)
    ap.add_argument("--steps_outer", type=int, default=800)
    ap.add_argument("--threads", type=int, default=26)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=300, help="validation pairs of the callback (val_curve_orig)")
    ap.add_argument("--m_inf", type=int, default=5000, help="ensemble size redrawn at every E-step when toy.M == -1 (infinite-data mode)")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "resnet", "transformer", "structured"], help="SCSIConfig.arch -- drift network architecture")
    ap.add_argument("--nhead", type=int, default=8, help="SCSIConfig.nhead -- attention heads, only used by --arch transformer/structured")
    ap.add_argument("--n_recon", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024, help="SCSIConfig.batch")
    ap.add_argument("--lr", type=float, default=1e-3, help="SCSIConfig.lr -- every outer refit starts its (fresh) "
                                                             "AdamW/cosine schedule at lr*0.5, regardless of k")
    ap.add_argument("--lr_end", type=float, default=1e-4, help="SCSIConfig.lr_end -- cosine floor for both the "
                                                                 "first fit and every outer refit")
    ap.add_argument("--lr_outer_floor_ratio", type=float, default=0.02,
                     help="SCSIConfig.lr_outer_floor_ratio -- fresh run: k=n_outer's outer lr ceiling as a "
                          "fraction of k=1's (lr*0.5)")
    ap.add_argument("--tag", default="", help="suffix for output files, to keep several training variants of one toy")
    ap.add_argument("--resume", default="", help="path to a checkpoint .pt (as saved by this script's callback) to resume "
                                                   "training from, instead of starting from the sampleable-prior init")
    ap.add_argument("--resume_lr", type=float, default=5e-5,
                     help="with --resume: constant outer lr ceiling over the resumed outer iterations (SCSIConfig.resume_lr)")
    ap.add_argument("--last_checkpoint", action="store_true",
                     help="best_k = the LAST checkpoint (saved as _model.pt) instead of the best-validation one. "
                          "Imposed in infinite-data mode (toy.M == -1: no overfitting, and the J=64 validation score is "
                          "too noisy to select on)")
    ap.add_argument("--out_dir", default="results/toys_M=-1", help="output folder of the checkpoints, _model.pt and _train.json "
                                                                "(results/toys_M=5000 for the finite-data M=5000 runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="SCSIConfig.device")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    toy = make_toy(a.toy)
    if toy.M == -1:                        # infinite data: best_k = the last checkpoint, imposed
        a.last_checkpoint = True
    stem = toy.name + (f"_{a.tag}" if a.tag else "")
    d, N = toy.d, toy.N
    g = torch.Generator().manual_seed(1000 + a.seed)
    infinite = (toy.M == -1)
    print(f"infinite is set to {infinite}")

    resume_ckpt = None
    if a.resume:
        resume_ckpt = torch.load(a.resume, map_location="cpu")
        # the architecture always comes from the checkpoint (it overrides --hidden/--depth/--arch/--nhead)
        a.hidden, a.depth = resume_ckpt.get("hidden", a.hidden), resume_ckpt.get("depth", a.depth)
        a.arch, a.nhead = resume_ckpt.get("arch", a.arch), resume_ckpt.get("nhead", a.nhead)
        if resume_ckpt.get("outer_gen_state") is not None:
            g.set_state(resume_ckpt["outer_gen_state"])                     # continue the SAME training-data stream, not a fresh replay
        print(f"resuming from {a.resume}: k={resume_ckpt['k']}  hidden={a.hidden}  depth={a.depth}", flush=True)

    if infinite: # fresh (C, Ce) ensemble every E-step
        def Ce_tr(B):
            return wishart_channel(toy.sample_prior(B, g), N, g)
    else:
        # the ONLY training data: one fixed ensemble of M noisy Ce. On --resume it must be the SAME ensemble as in the
        # original run, so it is redrawn from a fresh Generator(1000+seed) (g itself was restored to the checkpoint's state);
        # a fresh run draws it from g as before (identical stream).
        gd = torch.Generator().manual_seed(1000 + a.seed) if resume_ckpt is not None else g
        C_tr = toy.sample_prior(toy.M, gd); Ce_tr = wishart_channel(C_tr, N, gd); del C_tr
    # Validation dataset. Drawn from g after the training data: toy_oracle_log_score.py replays this exact sequence to
    # rebuild the same validation pairs, so keep the order and sizes of these draws unchanged.
    C_val = toy.sample_prior(a.n_val, g);
    Ce_cal = wishart_channel(C_val, N, g); Ce_val = wishart_channel(C_val, N, g)
    del C_val
    cfg = SCSIConfig(N=N, d=d, kappa=1.0, n_outer=a.n_outer, steps_first=a.steps_first, steps_outer=a.steps_outer, seed=a.seed,
                     init="deconv", log_prior_inflate=1.5, threads=a.threads, n_recon=a.n_recon, hidden=a.hidden, depth=a.depth,
                     arch=a.arch, nhead=a.nhead,
                     lr=a.lr, lr_end=a.lr_end, lr_outer_floor_ratio=a.lr_outer_floor_ratio,
                     batch=a.batch, resume_lr=a.resume_lr, device=a.device)
    model = SCSI(cfg)
    CK = [k for k in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                       320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700,
                       720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080,
                       1100, 1120, 1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300,
                       1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, *range(1620, 5001, 20)) if k <= a.n_outer]
    #Make results directory
    os.makedirs(a.out_dir, exist_ok=True)
    VAL_J = 64                       # posterior draws per validation pair in the callback (see the warning in cb)
    #Value and best -- pre-seed both with the resumed run's own history so best_k/val_curve_orig stay complete
    if resume_ckpt is not None and os.path.exists(f"{a.out_dir}/{stem}_train.json"):
        prev = json.load(open(f"{a.out_dir}/{stem}_train.json"))
        val = {int(k): v for k, v in prev.get("val_curve_orig", prev.get("val_curve", {})).items()}
        best = (prev["best_k"], val.get(prev["best_k"], -1e30))
    else:
        val, best = {}, (None, -1e30)
    S_ = Sym(d)

    #define callbacks
    def cb(m, k):
        global best
        if k in CK:
            # validation is inference: both nets in eval mode (no dropout), training net put back in train mode below
            m.ema_net.eval()
            m.net.eval()
            # DO NOT REMOVE (reproducibility): sample_posterior draws from m.dgen, the generator that also drives training
            # (minibatch indices, Wishart noise, t, z), so this validation pass shifts every later refit's random stream.
            # Dropping it, or changing n_val / the 64 draws / the CK grid, changes all checkpoints from the first one on.
            #Draw 64 posterior samples for each sample matrix Ce_cal
            Cs = m.sample_posterior(Ce_cal, VAL_J)
            #Calculate predictive log-score p(Ce_val|Cs) where Cs is generated from Ce_cal
            val[k] = float(posterior_predictive_logscore(Ce_val, N, Cs).mean())
            print(f"   [val] outer {k}: predictive log-score = {val[k]:.3f}", flush=True)
            e0, n0 = m.ema_net, m.net
            # every checkpoint (trajectory analysis, and a resume point): net + ema_net + generator states, so
            # training can be continued later (--resume) instead of restarted from the sampleable-prior init.
            torch.save({"state": e0.state_dict(), "net_state": n0.state_dict(), "mu": e0.mu.clone(), "sd": e0.sd.clone(),
                        "k": k, "N": N, "d": d, "hidden": a.hidden, "depth": a.depth, "arch": a.arch, "nhead": a.nhead,
                        "lr_outer": m.lr_outer,
                        "gen_state": m.gen.get_state(), "dgen_state": m.dgen.get_state(), "outer_gen_state": g.get_state()},
                       f"{a.out_dir}/{stem}_k{k}.pt")
            if a.last_checkpoint or val[k] > best[1]:
                best = (k, val[k])
            m.net.train()                                               # back to training for the next outer refit

    resume_kwargs = {}
    if resume_ckpt is not None:
        resume_kwargs["resume"] = {"net_state": resume_ckpt["net_state"], "ema_state": resume_ckpt["state"],
                                    "mu": resume_ckpt["mu"], "sd": resume_ckpt["sd"],
                                    "gen_state": resume_ckpt.get("gen_state"), "dgen_state": resume_ckpt.get("dgen_state"),
                                    "start_k": resume_ckpt["k"]}

    t0 = time.time()
    model.fit(Ce_tr, callback=cb, infinite_batch=a.m_inf if infinite else None, **resume_kwargs)
    # _model.pt is written once, at the end, from the selected checkpoint's EMA net (same content as the per-improvement
    # write it replaces). The selected _k{k}.pt may come from an earlier invocation when resuming; it is still on disk.
    zb = torch.load(f"{a.out_dir}/{stem}_k{best[0]}.pt", map_location="cpu")
    torch.save({"state": zb["state"], "mu": zb["mu"], "sd": zb["sd"], "k": zb["k"], "N": zb["N"], "d": zb["d"],
                "hidden": zb["hidden"], "depth": zb["depth"], "arch": zb["arch"], "nhead": zb["nhead"]},
               f"{a.out_dir}/{stem}_model.pt")
    # val_curve_orig: the callback's validation log-score per checkpoint ({k: mean over val_curve_orig_n_val pairs of
    # log (1/J) sum_j p(Ce_val | C_j), J = val_curve_orig_J posterior draws}), best_k, training time, M, N, d
    json.dump({"val_curve_orig": val, "val_curve_orig_n_val": a.n_val, "val_curve_orig_J": VAL_J, "best_k": best[0],
               "fit_sec": time.time() - t0, "M": toy.M, "N": N, "d": d}, open(f"{a.out_dir}/{stem}_train.json", "w"), indent=1)
    print(f"{stem}: selected outer iteration {best[0]}  (val curve {val})  fit {time.time()-t0:.0f}s")