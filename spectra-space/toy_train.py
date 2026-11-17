"""Train SC-SI on a toy ensemble (noisy Ce only), select the outer iteration on validation pairs, save the model.

Validation pairs (Ce_cal, Ce_val) share a latent C (as the split-half pairs do in the real-data protocol); the criterion is the
posterior predictive log-score of Ce_val given Ce_cal.  No clean covariance is given to the learner."""
import argparse, json, os, time
import numpy as np, torch
from scsi import SCSI, SCSIConfig, Chol, sample_covariance_from_factors, DT
from covutils import posterior_predictive_logscore   # now self-contained (no scsi.py dependency of its own)
from toys import make_toy                            # likewise -- both resolve directly against THIS directory's
                                                        # own scsi.py now, no cross-directory swap needed anymore
from run_paths import run_dir                         # per-encoding results/toys/{chol,spectral}/ routing


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", required=True)
    ap.add_argument("--n_outer", type=int, default=12)
    ap.add_argument("--steps_first", type=int, default=4000)
    ap.add_argument("--steps_outer", type=int, default=800)
    ap.add_argument("--threads", type=int, default=2, help="Number of parallel threads running on CPU for training.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--m_inf", type=int, default=5000, help="ensemble size redrawn at every E-step when toy.M == -1 (infinite-data mode)")
    ap.add_argument("--batch", type=int, default=1024, help="SCSIConfig.batch -- the notebook uses 192 for D<=8")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--last_checkpoint", action=argparse.BooleanOptionalAction, help="The last training checkpoint is saved as the model, regardless of past validation performance (instead of the usual best-validation-score selection). Advised when toy.M == -1, i.e. infinite data.")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer", "transformer_eig", "structured"], help="SCSIConfig.arch")
    ap.add_argument("--encoding", default="chol", choices=["chol", "spectral"], help="SCSIConfig.encoding -- "
                     "'spectral' (Spec: d-dim log-eigenvalue chart) is only valid for a rotationally-invariant "
                     "prior (e.g. iw_ri_*); 'chol' (Chol: d(d+1)/2-dim Cholesky chart) works for any prior")
    ap.add_argument("--nhead", type=int, default=8, help="SCSIConfig.nhead (transformer only)")
    ap.add_argument("--n_recon", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="SCSIConfig.lr -- every outer refit starts its (fresh) \
                                                             AdamW/cosine schedule at lr*0.5, regardless of k")
    ap.add_argument("--lr_end", type=float, default=1e-4, help="SCSIConfig.lr_end -- cosine floor for both the \
                                                                 first fit and every outer refit")
    ap.add_argument("--lr_outer_schedule", default="cosine", choices=["cosine", "const"],
                     help="SCSIConfig.lr_outer_schedule -- 'cosine' (default) anneals each outer refit's own "
                          "lr ceiling across outer iterations 1..n_outer instead of holding it fixed at lr*0.5; "
                          "'const' reproduces the original (pre-scheduler) behaviour")
    ap.add_argument("--lr_outer_floor_ratio", type=float, default=0.02,
                     help="SCSIConfig.lr_outer_floor_ratio -- k=n_outer's lr ceiling as a fraction of k=1's "
                          "(cosine schedule only)")
    ap.add_argument("--sampler", default="ode", choices=["ode", "follmer"], help="SCSIConfig.sampler -- 'ode' "
             "(default) is the deterministic flow-matching ODE from an independent base; 'follmer' is the "
             "stochastic Follmer bridge SDE anchored at Ye, matching spectra-space/covsi_pd_em_spectral.py")
    ap.add_argument("--follmer_epsilon", type=float, default=1.0, help="SCSIConfig.follmer_epsilon (sampler=follmer only)")
    ap.add_argument("--follmer_T", type=int, default=50, help="SCSIConfig.follmer_T (sampler=follmer only)")
    ap.add_argument("--n_ode_steps", type=int, default=50, help="SCSIConfig.n_ode_steps -- Euler/Ralston steps for "
                                                                   "sample_posterior (the notebook uses ODE_NFE=100 "
                                                                   "for D<=8, 50 for QUICK_RUN)")
    ap.add_argument("--ode_solver", default="euler", choices=["euler", "ralston"], help="SCSIConfig.ode_solver")
    ap.add_argument("--ref_sd", type=float, default=None, help="SCSIConfig.ref_sd -- per-coordinate std of the "
                                                                  "base/reference draw; default None -> 1.10/sqrt(d) "
                                                                  "(the notebook's REFERENCE_FACTOR_SD convention)")
    ap.add_argument("--refresh_every", type=int, default=0, help="SCSIConfig.refresh_every -- recompute mu_y/sd_y/"
                                                                    "mu_ce/sd_ce from that round's own data every N "
                                                                    "outer iterations (0 = never). The notebook "
                                                                    "recalibrates every single round (cheap: it's "
                                                                    "just mean/std of an already-computed batch) -- "
                                                                    "pass 1 to match that exactly.")
    ap.add_argument("--persist_optimizer", action="store_true", help="SCSIConfig.persist_optimizer -- keep the AdamW "
                     "momentum/variance state across outer EM iterations instead of a fresh optimizer every refit")
    ap.add_argument("--tag", default="", help="suffix for output files, to keep several training variants of one toy")
    ap.add_argument("--resume", default="", help="path to a checkpoint .pt (as saved by this script's callback) to resume "
                                                   "training from, instead of starting from the sampleable-prior init")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device on which to run calculations")
    ap.add_argument("--verbose", type=str, default='False', help="print logging messages")

    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    toy = make_toy(a.toy)
    stem = toy.name + (f"_{a.tag}" if a.tag else "")
    d, N = toy.d, toy.N
    g = torch.Generator().manual_seed(1000 + a.seed)
    infinite = (toy.M == -1)
    print(f"infinite is set to {infinite}")
    verbose=True if a.verbose=="True" else False

    Cho=Chol(d, a.device)

    resume_ckpt = None
    if a.resume:
        resume_ckpt = torch.load(a.resume, map_location="cpu")
        # match the architecture to whatever the checkpoint was trained with, unless overridden on the CLI
        a.hidden, a.depth = resume_ckpt.get("hidden", a.hidden), resume_ckpt.get("depth", a.depth)
        a.arch, a.nhead = resume_ckpt.get("arch", a.arch), resume_ckpt.get("nhead", a.nhead)
        a.encoding = resume_ckpt.get("encoding", a.encoding)      # old checkpoints predate this field -> "chol"
        a.sampler = resume_ckpt.get("sampler", a.sampler)         # old checkpoints predate this field -> "ode"
        if resume_ckpt.get("outer_gen_state") is not None:
            g.set_state(resume_ckpt["outer_gen_state"])                     # continue the SAME training-data stream, not a fresh replay
        print(f"resuming from {a.resume}: k={resume_ckpt['k']}  hidden={a.hidden}  depth={a.depth}", flush=True)

    rd = run_dir(a.encoding)   # results/toys/chol or results/toys/spectral -- a.encoding is final by now

    cfg = SCSIConfig(N=N, d=d, n_outer=a.n_outer, steps_first=a.steps_first, steps_outer=a.steps_outer, seed=a.seed,
             init="raw", log_prior_inflate=1.5, threads=a.threads, n_recon=a.n_recon, hidden=a.hidden, depth=a.depth,
             arch=a.arch, nhead=a.nhead, encoding=a.encoding, batch=a.batch,
             lr=a.lr, lr_end=a.lr_end, lr_outer_schedule=a.lr_outer_schedule, lr_outer_floor_ratio=a.lr_outer_floor_ratio,
             n_ode_steps=a.n_ode_steps, ode_solver=a.ode_solver, ref_sd=a.ref_sd, refresh_every=a.refresh_every,
             persist_optimizer=a.persist_optimizer, device=a.device,
             sampler=a.sampler, follmer_epsilon=a.follmer_epsilon, follmer_T=a.follmer_T)

    if infinite: # fresh (C, Ce) pair at every E-step loop
        def Ce_tr(B):
            C_tr=toy.sample_prior(B, g)
            Sigma_tr=Cho.encode(C_tr)
            return sample_covariance_from_factors(Sigma_tr, N, g)
    else:
        C_tr = toy.sample_prior(toy.M, g); Sigma_tr = Cho.encode(C_tr); Ce_tr = sample_covariance_from_factors(Sigma_tr, N, g); del C_tr
    #the ONLY training data: noisy Ce
    #Validation dataset
    C_val = toy.sample_prior(a.n_val, g);
    Sigma_val = Cho.encode(C_val);
    Ce_cal = sample_covariance_from_factors(Sigma_val, N, g); Ce_val = sample_covariance_from_factors(Sigma_val, N, g)

    #Create model
    model = SCSI(cfg)
    CK = [k for k in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
                       320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700,
                       720, 740, 760, 780, 800, 850, 900, 950, 1000) if k <= a.n_outer]
    #Make results directory
    os.makedirs("results/toys", exist_ok=True)
    #Value and best -- pre-seed both with the resumed run's own history so best_k/val_curve stay complete
    if resume_ckpt is not None and os.path.exists(f"{rd}/{stem}_train.json"):
        prev = json.load(open(f"{rd}/{stem}_train.json"))
        val = {int(k): v for k, v in prev["val_curve"].items()}
        best = (prev["best_k"], val.get(prev["best_k"], -1e30))
    else:
        val, best = {}, (None, -1e30)

    #define callbacks
    def cb(m, k):
        global best
        if k in CK:
            Sigma_s = m.sample_posterior_sigma(Ce_cal, 64, chunk=4000).to(DT)
            val[k] = float(posterior_predictive_logscore(Ce_val, N, Sigma_s).mean())
            print(f"   [val] outer {k}: predictive log-score = {val[k]:.3f}", flush=True)
            e0, n0 = m.ema_net, m.net
            torch.save({"state": e0.state_dict(), "net_state": n0.state_dict(),
                        "mu_y": e0.mu_y.clone(), "sd_y": e0.sd_y.clone(),
                        "mu_ce": e0.mu_ce.clone(), "sd_ce": e0.sd_ce.clone(),
                        "velocity_loc": e0.velocity_loc.clone(), "velocity_scale": e0.velocity_scale.clone(),
                        "k": k, "N": N, "d": d, "hidden": a.hidden, "depth": a.depth, "arch": a.arch, "nhead": a.nhead,
                        "encoding": a.encoding, "sampler": a.sampler,
                        "gen_state": m.gen.get_state(), "outer_gen_state": g.get_state()},
                       f"{rd}/{stem}_k{k}.pt")
            if a.last_checkpoint or val[k] > best[1]:
                best = (k, val[k])

    resume_kwargs = {}
    if resume_ckpt is not None:
        resume_kwargs["resume"] = {"net_state": resume_ckpt["net_state"], "ema_state": resume_ckpt["state"],
                                    "mu_y": resume_ckpt["mu_y"], "sd_y": resume_ckpt["sd_y"],
                                    "mu_ce": resume_ckpt["mu_ce"], "sd_ce": resume_ckpt["sd_ce"],
                                    "velocity_loc": resume_ckpt["velocity_loc"], "velocity_scale": resume_ckpt["velocity_scale"],
                                    "gen_state": resume_ckpt.get("gen_state"), "start_k": resume_ckpt["k"]}

    t0 = time.time()
    print("Start fit routine")
    model.fit(Ce_tr, callback=cb, infinite_batch=a.m_inf if infinite else None, verbose=verbose, **resume_kwargs)
    zb = torch.load(f"{rd}/{stem}_k{best[0]}.pt", map_location="cpu")
    torch.save({"state": zb["state"], "mu_y": zb["mu_y"], "sd_y": zb["sd_y"], "mu_ce": zb["mu_ce"], "sd_ce": zb["sd_ce"],
                "velocity_loc": zb["velocity_loc"], "velocity_scale": zb["velocity_scale"], "k": zb["k"],
                "N": zb["N"], "d": zb["d"], "hidden": zb["hidden"], "depth": zb["depth"], "arch": zb["arch"],
                "nhead": zb["nhead"], "encoding": zb["encoding"], "sampler": zb["sampler"]}, f"{rd}/{stem}_model.pt")
    ## Dump in a json file the val_curve (tagged with its own n_val/J, the only identifiers a log-score
    ## curve needs -- see cb()'s J=64 above), the best iteration, the training time, the amount of data
    ## per empirical matrices and the space dimension
    json.dump({"val_curve": val, "val_curve_n_val": a.n_val, "val_curve_J": 64, "best_k": best[0],
               "fit_sec": time.time() - t0, "M": toy.M, "N": N, "d": d}, open(f"{rd}/{stem}_train.json", "w"), indent=1)
    print(f"{stem}: selected outer iteration {best[0]}  (val curve {val})  fit {time.time()-t0:.0f}s")