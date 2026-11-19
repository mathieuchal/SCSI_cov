# Spectra-space

Spectral-encoding SC-SI (self-consistent stochastic-interpolant covariance posterior estimator) results.
Covers only the two actively-maintained spectral runs: `iw_ri_d8_inf` (rotationally-invariant prior --
the spectral encoding's valid regime) and `iso_diag_d8_inf` (not RI, an empirical robustness check), both
trained with the Föllmer bridge sampler (tag `spectral_follmer`).

Shared modules: `toys.py`, `scsi.py`, `model.py`, `covutils.py`, `toy_train.py`, `toy_eval.py`,
`toy_W1_mean.py`, `log_score_oracle.py`, `log_score_scsi.py`, `run_paths.py`, `plotstyle.py`,
`utils_spectra.py`. Campaign-specific: `train_iw_ri_d8.sh`, `train_iso_diag_d8.sh`,
`train_all_spectra.sh`, `toy_report_spectral_follmer.py`.

## Pipeline

For each toy, in order:
1. **Training** (`toy_train.py`) -- EM over outer iterations, checkpointed every few iterations to
   `results/toys/{toy}_spectral_follmer_k{k}.pt`, best one copied to `{toy}_spectral_follmer_model.pt`.
   A small-J (64) validation log-score curve is written to `{toy}_spectral_follmer_train.json`'s
   `val_curve` as training runs, and `best_k` picked from it.

   `iso_diag_d8_inf` was trained in **two separate `toy_train.py` calls**, not one (`train_iso_diag_d8.sh`
   runs `--n_outer 120` first, then `--resume .../k120.pt --n_outer 240`). This is NOT interchangeable
   with a single `--n_outer 240` call from scratch: each outer iteration's own (inner) cosine LR ceiling,
   `_outer_lr0(k)`, anneals across `1..cfg.n_outer` of *that specific call* -- stage 1 (`n_outer=120`)
   decays all the way to its floor by `k=120`, then stage 2 (`n_outer=240`) recomputes `frac` from `k=121`
   against its own, longer `n_outer=240`, landing roughly at that schedule's *midpoint* rather than
   continuing stage 1's decay. Concretely: `lr0` is `1e-5` at `k=120` (stage 1's floor) then jumps to
   `2.5e-4` at `k=121` (stage 2's own `k=121` position, not `k=1`'s `5e-4` ceiling -- `k` itself stays
   globally continuous across the resume, only each call's own `n_outer` denominator changes). A single
   continuous `--n_outer 240` run would never see this jump. `iw_ri_d8_inf` is single-stage, unaffected.
2. **Oracle / IW-ML log-score** (`log_score_oracle.py`'s `eval_oracle_IW_log_score`) -- the reference
   posterior's predictive log-score (exact formula if the toy has one, else an HMC chain) and the
   conjugate IW-ML baseline, at a much larger J (default 2e5, n_val=10) than training's own curve --
   merged into `results/toys/{toy}_logscore.json`.
3. **SC-SI checkpoint rescore** (`log_score_scsi.py`'s `eval_scsi_checkpoints_log_score`) -- resamples
   every saved checkpoint at that same big J, replacing `train.json`'s `val_curve`/`best_k` with the
   big-J version (the small-J one is kept alongside as `val_curve_orig`), and writes the `"scsi"` entry
   into `{toy}_logscore.json`.
4. **W1-mean eval** (`toy_W1_mean.py`'s `w1_mean_dict`) -- per-checkpoint Wasserstein-1 distance to the
   reference posterior (6 marginal statistics, `n_test=128`, `J=1024`), plus the IW-ML baseline and a
   sampling-noise floor. Merged into `results/toys/{toy}_W1_mean.json` (toy-level: floor/IW-ML baseline,
   and the SC-SI value **at the checkpoint actually selected** in step 3, i.e. `best_k`, not necessarily
   the W1-minimizing one) and into `train.json`'s `w1_curve` (the full per-checkpoint trajectory).
5. **Report** (`toy_report_spectral_follmer.py`) -- combined log-score/W1 curve figure
   (`figs/fig_spectral_follmer.png`) and a plain-text summary table (`figs/table_spectral_follmer.txt`)
   for both toys side by side.

Every result file is self-describing: log-score/W1 entries carry their own `n_val`/`J` (or
`n_test`/`n_ref`/`J`/`agg`), so there is never a need to infer which settings produced a given number.

## Reproducing the results

```bash
cd spectra-space
source .venv/bin/activate   # torch 2.2.2+cu121 -- needed for bit-identical reproduction

./train_all_spectra.sh              # both toys, steps 1-5 above, default J=2e5 for the log-score steps
./train_all_spectra.sh 50000         # override J (positional $1) if the default is too slow
```

Or one toy at a time: `./train_iw_ri_d8.sh [J]` / `./train_iso_diag_d8.sh [J]` (steps 1-4 only, no combined
figure/table -- that needs both toys done, see `toy_report_spectral_follmer.py`'s own docstring).

Starting from an EMPTY `results/toys/` (nothing checkpointed yet), the full `train_all_spectra.sh` run
takes several hours, dominated by step 3's big-J resampling (~1500-2400s per checkpoint, 16+22
checkpoints across both toys) -- steps 1, 4 and 5 are comparatively cheap (minutes).

Every step is resumable: training supports `--resume`, and both the log-score and W1 evaluations keep a
disposable per-run cache (`*_cache.json`, auto-deleted once merged) so a killed/restarted run picks back
up rather than redoing finished checkpoints. Re-running `train_all_spectra.sh` against an already-complete
`results/toys/` is cheap: every step detects its work is already done and just re-merges/re-plots.

## Current results (figs/table_spectral_follmer.txt)

```
iw_ri_d8_inf   (IW, RI, N=40, d=8):      best_k=20
  log-score   oracle=-130.6795  iw_ml=-130.6478  scsi=-130.8167  gap=-0.1372
  W1          floor=0.0537      iw_ml=0.0459      scsi=0.0903

iso_diag_d8_inf (Iso-diagonal, N=40, d=8): best_k=240
  log-score   oracle=-336.7046  iw_ml=-337.3245  scsi=-336.7330  gap=-0.0284
  W1          floor=0.0527      iw_ml=1.9085      scsi=0.1931
```

Verified bit-identical by direct recomputation from the saved checkpoints, both at the original torch
version (2.2.2+cu121, in `.venv`) and incidentally also under a newer one (2.7.0) when `.venv` was briefly
unavailable after a directory move.

## Known caveat

`iso_diag_d8_inf`'s oracle (non-exact, single HMC chain) is **not** bit-reproducible run to run even at a
fixed seed: two real runs differed by ~0.003 nats (`ESS/J≈0.024`, well above the `low_ess_threshold=1e-3`
that would auto-trigger extra chains in `eval_oracle_IW_log_score`). The value above is the originally
recorded one; re-deriving it from scratch may land a little off.
