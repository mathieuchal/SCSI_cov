"""Oracle + IW-ML log-score computation at a given (n_val, J), replacing log_score/toy_oracle_log_score.py
-- that script's small J (64/256/2048, at n_val=300) is not actually usable: the log-mean-exp estimator is
biased low, by a DIFFERENT amount for the oracle's HMC draws than for an approximate model's draws, and
that bias barely shrinks even at J in the hundreds of thousands for a heavy-tailed posterior-predictive
(effective sample size as low as ~1e-5 x J) -- so a small-J "SC-SI beats the oracle" reading can be pure
measurement artifact. J=200000 per validation task is only tractable with n_val cut down to 10 -- verified
(nval_sensitivity_factor.py, prior session segment) that this doesn't change the oracle-vs-SC-SI GAP
materially, since the same (Ce_cal, Ce_val) pair is used for every model compared.

The SC-SI-checkpoint-resampling side of this (eval_scsi_checkpoints_log_score/_full_curve) lives in the
separate log_score_scsi.py, which imports _draw_validation_pair from here to share the exact same draw
sequence -- that module needs a trained run's checkpoints on disk, this one doesn't, so they're driven
independently (call this module's eval_oracle_IW_log_score first, same n_val, so its "oracle" entry
exists for log_score_scsi.py's gap computation to read).

Result schema in {toy}_logscore.json: top-level "n_val"/"J" are SHARED across "oracle"/"iw_ml"/"scsi"
(all three are evaluated at the same (Ce_cal, Ce_val) validation pair, drawn once from n_val -- see
_draw_validation_pair), each a dict {"value", ...} -- not its own repeated {"n_val","J",...}. Top-level
"N"/"d" are the toy's own intrinsic (always-the-same-for-this-toy-name) Wishart dof/dimension, stored so
a reader of this file alone doesn't need make_toy(toy_name) -- unlike n_val/J, they never vary call to
call. This REPLACES the old flat-key scheme (oracle_J64/J256/J2048,
oracle_exact(+_n_val), iw_ml_J64/J256/exact, bigJ_n_val/bigJ_J/bigJ_oracle_pair/bigJ_iw_ml_pair/
bigJ_oracle_ess_ratio/bigJ_oracle_note, plugin_scm, scsi_{tag}_J64/J256/kbest) written by
toy_oracle_log_score.py and by earlier versions of this module -- those keys are NOT written by this
function anymore and will progressively stop existing in files this function touches. Any consumer still
reading them (as of this writing: log_score/toy_log_score.py, log_score/fig_log_score_w1_d16.py,
log_score/fig_log_score_d16.py, log_score/fig_log_score_d16_transformer.py) needs to be updated to read
"oracle"/"iw_ml" instead, or will silently keep working off whatever stale values toy_oracle_log_score.py
last wrote for toys this function hasn't touched yet.

For a non-conjugate toy with very low ESS/J (seen: ~1e-5 for factor/hetero/iso_diag_d16_inf), a single HMC
chain's log-mean-exp score is itself not reproducible run-to-run (iso_diag_d16_inf's original single-chain
value was a ~12-nat outlier against its own 5-chain spread) -- so low-ESS toys get re-verified with several
independent chains (different seeds, same (Ce_cal,Ce_val)) and the production value becomes their mean."""
import gc, json, os, time
import numpy as np
import torch

from scsi import DT, sample_covariance_from_factors
from covutils import IWPrior, posterior_predictive_logscore, wishart_logscore
from toys import make_toy


def wishart_channel(C, N, gen=None):
    return sample_covariance_from_factors(torch.linalg.cholesky(C.to(DT)), N, gen)


def as_factor(C):
    return torch.linalg.cholesky(C.to(DT))


def ess_ratio(Cs, Ce_val, N):
    """Kish's ESS of the J HMC draws' per-task Wishart log-likelihood, averaged over n_val tasks and
    normalized by J."""
    s = wishart_logscore(Ce_val, N, as_factor(Cs))
    smax = s.max(dim=1, keepdim=True).values
    w = torch.exp(s - smax)
    ess = (w.sum(dim=1) ** 2) / (w ** 2).sum(dim=1)
    return (ess / s.shape[1]).mean().item()


def _draw_validation_pair(toy, n_val, m_inf, device):
    """The single deterministic (seed=1000) draw sequence every oracle/IW-ML/checkpoint evaluation in this
    module is built from, for a given (toy, n_val) -- matches toy_oracle_log_score.py's original seed=0
    draw order exactly. Returns (Ce_tr, Ce_cal, Ce_val): Ce_tr is the IW-ML fitting ensemble (drawn before
    (Ce_cal, Ce_val) for a finite toy, after for an infinite one -- toy.M == -1); Ce_cal/Ce_val are the
    validation pair every caller needing the SAME pair (eval_oracle_IW_log_score,
    eval_scsi_checkpoints_log_score) must go through THIS function to get, rather than reimplementing the
    draw sequence, so the two can never silently drift out of sync."""
    N = toy.N
    g = torch.Generator().manual_seed(1000)
    infinite = (toy.M == -1)
    if not infinite:
        C_tr = toy.sample_prior(toy.M, g); Ce_tr = wishart_channel(C_tr, N, g)
    C_va = toy.sample_prior(n_val, g)
    Ce_cal = wishart_channel(C_va, N, g).to(device)
    Ce_val = wishart_channel(C_va, N, g).to(device)
    if infinite:
        C_tr = toy.sample_prior(m_inf, g); Ce_tr = wishart_channel(C_tr, N, g)        # drawn AFTER C_va
    return Ce_tr.to(device), Ce_cal, Ce_val


def eval_oracle_IW_log_score(toy_name, n_val=10, J=200000, warmup=1500, m_inf=5000,
                     seeds=(4321,), extra_seeds=(1111, 2222, 3333, 4444), low_ess_threshold=1e-3,
                     device=None, threads=2, out_dir="results/toys"):
    """seeds: HMC chain seed(s) to run for a non-exact toy's oracle (ignored if exact). If the resulting
    ESS/J (Kish's effective sample size, normalized by J -- see ess_ratio) of the FIRST chain comes back
    below `low_ess_threshold`, the chains in `extra_seeds` are automatically run TOO (not instead -- the
    already-computed `seeds` chains are kept, not redone) and the production oracle value becomes the
    mean of all of them. A single low-ESS chain's log-mean-exp score is itself not reproducible run-to-run
    (iso_diag_d16_inf's original single-chain value was a ~12-nat outlier against its own 5-chain spread),
    so this re-check is automatic rather than left to a calling script to remember to do. Pass
    extra_seeds=() to disable it (e.g. for a quick one-chain sanity check).

    Merges results into {out_dir}/{toy_name}_logscore.json in place (other fields are left
    untouched). Every call recomputes the oracle/IW-ML fresh (no caching) -- the (Ce_cal, Ce_val) pair is
    deterministic (seed=1000, via _draw_validation_pair) so repeated calls at the same n_val stay
    comparable regardless.

    Only computes the oracle/IW-ML log-score -- it does NOT touch any trained SC-SI checkpoint, so it
    takes no `tag`. For resampling a SC-SI run's checkpoints at this same (n_val, J) pair and comparing
    them to this call's oracle, see eval_scsi_checkpoints_log_score (call this function first, same
    n_val, so its "oracle" entry exists for that one to read)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(threads)
    toy = make_toy(toy_name)
    N = toy.N
    has_exact = hasattr(toy, "exact_logscore")

    out_path = f"{out_dir}/{toy_name}_logscore.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {"toy": toy_name, "exact_reference": has_exact}
    out["N"], out["d"] = N, toy.d  # intrinsic to the toy's own definition (unlike n_val/J below), so always
    # the same value every call -- stored anyway so a reader of this file alone doesn't need make_toy(toy_name)

    Ce_tr, Ce_cal, Ce_val = _draw_validation_pair(toy, n_val, m_inf, device)

    note, ess = None, None
    iw = IWPrior(Ce_tr, N, iters=800)
    iw_ml_val = float(iw.exact_logscore(Ce_val, Ce_cal).mean())
    if has_exact:
        oracle_val = float(toy.exact_logscore(Ce_val, Ce_cal).mean())
    else:
        def run_chain(seed):
            t0 = time.time()
            gg = torch.Generator().manual_seed(seed)
            Cs = toy.ref_posterior(Ce_cal, J, gg, warmup=warmup)
            sc = float(posterior_predictive_logscore(Ce_val, N, as_factor(Cs)).mean())
            e = ess_ratio(Cs, Ce_val, N)
            print(f"  [{toy_name}] chain seed={seed}: oracle={sc:.4f} ESS/J={e:.2e} ({time.time() - t0:.0f}s)", flush=True)
            del Cs; gc.collect(); torch.cuda.empty_cache()
            return sc, e

        scores = []
        for seed in seeds:
            sc, e = run_chain(seed)
            scores.append(sc)
            if ess is None:
                ess = e
        if ess < low_ess_threshold and extra_seeds:
            print(f"  [{toy_name}] ESS/J={ess:.2e} < {low_ess_threshold:.0e} -> extending with "
                  f"{len(extra_seeds)} more chains for reproducibility", flush=True)
            for seed in extra_seeds:
                sc, _ = run_chain(seed)
                scores.append(sc)
        oracle_val = float(np.mean(scores))
        if len(scores) > 1:
            all_seeds = list(seeds) + (list(extra_seeds) if ess < low_ess_threshold and extra_seeds else [])
            note = (f"mean of {len(scores)} independent chains (seeds {'/'.join(map(str, all_seeds))}): "
                     + "/".join(f"{s:.3f}" for s in scores)
                     + f", std={float(np.std(scores)):.3f}, range={max(scores) - min(scores):.3f}")

    # n_val/J are shared top-level fields (see module docstring) -- oracle/iw_ml/scsi are all evaluated
    # at the same pair, so a later call at a different (n_val, J) just overwrites these two once rather
    # than three separate, always-equal copies.
    out["n_val"] = n_val
    out["J"] = J
    out["oracle"] = {"value": oracle_val}
    out["iw_ml"] = {"value": iw_ml_val}
    if ess is not None:
        out["oracle"]["ess_ratio"] = ess
    if note is not None:
        out["oracle"]["note"] = note

    json.dump(out, open(out_path, "w"), indent=1)
    msg = f"{toy_name}: oracle={oracle_val:.4f} iw_ml={iw_ml_val:.4f}"
    if ess is not None:
        msg += f" ESS/J={ess:.2e}"
    print(msg, flush=True)

    return out
