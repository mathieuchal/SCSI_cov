# EEG: eyes open vs eyes closed

SC-SI on short-window EEG covariances (PhysioNet EEG Motor Movement/Imagery, 109 subjects, d = 8 channels), evaluated by subject-disjoint 5-fold cross-validation. This folder holds the code and the result files of the paper's single EEG run, `eeg_w4_s0_k1.0_e0.65_rv` (MLP 512, 4-s windows, N = 28).

Setup: the same Python 3.11 environment as `log-space` (`pip install -r ../log-space/requirements.txt`; it includes `mne`). A CUDA GPU is needed for steps 1–3, not for the figure. All commands are run from `eeg/`.

**What is versioned.** The code and the JSON result files of the paper run, which are enough to rebuild the figure (step 5) without a GPU. The data (`data/`), the checkpoints (`results/*_ckpt/`) and the per-case arrays (`results/*.npz`) are not versioned; steps 0–4 regenerate them.

## All steps at once: `run_cv.py`

After step 0, `run_cv.py` runs steps 1–5 for the five folds. For step 2, it reads each fold's selected checkpoint from `{tag}_results.json`.

```bash
python run_cv.py                                        # steps 1-5, all folds
python run_cv.py --folds 2 3 --stages exp splithalf     # steps 1-2 for some folds only (e.g. one process per GPU)
python run_cv.py --stages pool figs                     # steps 3-5 only, from the existing per-fold files (no GPU)
```

`eeg_exp.py` skips training when a fold's checkpoints already exist; `--retrain` forces it. The sections below detail each step.

## 0. Data: `eeg_data.py`

Download the baseline runs `S{001..109}R01.edf` (eyes open) and `S{001..109}R02.edf` (eyes closed) of [eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/) into `data/eeg/`, then:

```bash
python eeg_data.py 4        # -> data/eeg_scm_4s.npz
```

Each run is 1 min; its label is the run's condition. The script:
- keeps 8 channels (Fz, C3, Cz, C4, Pz, O1, Oz, O2), with a common-average reference over the 64 channels and an 8–13 Hz (alpha) band-pass;
- drops 2 s at each end of the run (56 s left);
- cuts the run into non-overlapping windows of 4 s (the argument), 14 per run.

It writes:

| array | shape | content |
|---|---|---|
| `scm_win` | (109, 2, 14, 8, 8) | sample covariance X Xᵀ / n of each 4-s window (zero-mean model, no centring) |
| `scm_full` | (109, 2, 8, 8) | covariance of the whole trimmed run |
| `scm_il` | (109, 2, 7, 2, 8, 8) | split-half covariances: 8-s epochs of eight 1-s chunks, half A = even chunks, half B = odd chunks |
| `neff_ratio` | per channel | N_eff / n from the autocorrelation; the Wishart N is median(N_eff/n) · n · `neff_scale` = 28 for 4 s and `neff_scale` = 0.65 |

## 1. Training and next-window evaluation: `eeg_exp.py`

```bash
for f in 0 1 2 3 4; do
  python eeg_exp.py --win 4 --neff_scale 0.65 --hidden 512 --n_outer 30 --cv_folds 5 --cv_fold $f --tag _rv_cv5f${f}_wide
done
```

**Split.** The 109 subjects are permuted with `seed` 0 and cut into 5 folds. Fold f is tested; among the other folds, taken in rotation from f+1, the first 10 subjects are the validation set and the rest (77) the training set. Every subject is tested exactly once. `eeg_exp2.py` rebuilds exactly the same split.

**Training.** SC-SI is trained on the training windows; checkpoints are saved at k ∈ {0, 2, 5, 8, 12, 16, 20, 25, 30} (`results/{tag}_ckpt/`).

**Checkpoint selection.** The checkpoint is chosen by the Gaussian (Wishart) predictive log-score on the validation subjects (`--select logscore`, the default). This gives k = [20, 5, 16, 12, 16] for the five folds. The validation curves are in `{tag}_val.json`; `{tag}_valwc.json` holds the worst-case PIT-KS curve, which is reported but not used.

**Evaluation on the test subjects**, at the selected checkpoint (next-window protocol: predict the next window of the same run):
- predictive log-score, with the gain over the sample covariance (95 % CI by bootstrap over subjects);
- coverage at 50 / 80 / 95 %;
- eyes-open/closed classification. A logistic regression on tangent-space features is trained on 4 posterior draws per training window; the test probability is the mean of its probabilities over 64 posterior draws.

Baselines: sample covariance, OAS, nonlinear shrinkage (LW-NLS), and the inverse-Wishart conjugate prior fitted by marginal likelihood.

Outputs: `{tag}_results.json` (summaries, `kbest`) and `{tag}_arrays.npz` (per-case scores, PIT values and class probabilities).

## 2. Split-half calibration: `eeg_exp2.py`

```bash
K=(20 5 16 12 16)
for f in 0 1 2 3 4; do
  python eeg_exp2.py eeg_w4_s0_k1.0_e0.65_rv_cv5f${f}_wide ${K[$f]} 0.65 --cv_folds 5 --cv_fold $f
done
```

This is the "common C" protocol. Half A and half B of each 8-s epoch are two nearly independent Wishart(N) draws around the same local covariance, so the test is not affected by drift between windows. The posterior predictive law of B is evaluated given A, and of A given B (J = 256 draws), on four families of quantities:
- the log-score;
- the coverage of directional variances uᵀBu, for 11 directions (the 8 canonical axes and the top 3 eigenvectors of the mean training covariance);
- the PIT coverage of four functionals: log det, log C[Oz,Oz], top-eigenvalue share, log condition number;
- the open-vs-closed Riemannian (AIRM) distance.

Outputs: `{tag}_splithalf_k{k}.json` and `_arrays.npz` (per-case PIT values, keys `dir|…`, `fun|…`, `dist|…`).

## 3. Classification pooled over the folds: `eeg_cv_classif.py`

```bash
python eeg_cv_classif.py --prefix eeg_w4_s0_k1.0_e0.65_rvlogscore --folds 5
cp results/eeg_w4_s0_k1.0_e0.65_rvlogscore_cv5_wide_classification.json results/eeg_w4_s0_k1.0_e0.65_cv5_wide_classification.json
```

This pools the held-out predictions of the five folds. It reports:
- accuracy, NLL, ECE and selective accuracy;
- per-fold accuracy;
- paired subject-level tests (bootstrap over subjects, Wilcoxon signed-rank test).

The `_rvlogscore_cv5f{f}_wide_*` files are copies of the `_rv_cv5f{f}_wide_*` files of step 1: they are byte-identical, since the log-score is the default selection rule. `paper_figs.py` reads the classification file under the name without suffix.

## 4. Calibration pooled over the folds: `eeg_cv_pool_calib.py`

```bash
python eeg_cv_pool_calib.py --results_prefix eeg_w4_s0_k1.0_e0.65_rvlogscore --splithalf_prefix eeg_w4_s0_k1.0_e0.65_rv --folds 5
# -> results/eeg_w4_s0_k1.0_e0.65_rvlogscore_cv5_{nextwindow,splithalf}.json
```

This pools steps 1 and 2 over the folds. Coverages are averaged with weights equal to the number of test subjects, which is identical to recomputing them on the pooled cases. Log-scores are pooled case by case, with a 95 % CI by bootstrap over subjects.

`results/eeg_w4_s0_k1.0_e0.65_rvlogscore_cv5_splithalf_coverage_ci.json` holds the split-half coverages with subject-bootstrap CIs. It was computed from the per-case PIT values of step 2 (`_splithalf_k{k}_arrays.npz`).

## 5. Figure: `paper_figs.py`

```bash
python paper_figs.py fig_eeg      # -> paper/fig_real/fig_eeg.png
python figs_classif.py 4          # panel (c) on its own -> paper/fig_real/fig_classif_w4.png
```

`fig_eeg` has three panels:
- (a) log-score gain over the sample covariance, for the next-window and split-half protocols;
- (b) 80 % coverage per metric;
- (c) selective classification.

It only reads the JSON files of steps 3–4.

Supplementary figures, from the same JSON files:

```bash
python eeg_figs.py     # -> figs/eeg_w4_s0_k1.0_e0.65_E1_calibration.png, figs/eeg_w4_s0_k1.0_e0.65_E3_decision.png
```

- **E1**: directional-variance coverage (split-half), log-score gains (split-half and next window) and 80 % coverage of the functionals and of the open-vs-closed distance.
- **E3**: selective classification curves and the pooled classification NLL.

## Files

| file | role |
|---|---|
| `run_cv.py` | runs steps 1–5 over the five folds |
| `scsi.py`, `model.py` | SC-SI (Föllmer SDE sampler, EM training) and drift networks |
| `covutils.py` | Wishart log-scores, PIT, coverage, shrinkage baselines, IW prior, bootstrap CIs |
| `toys.py`, `toy_eval.py` | imported by `paper_figs.py` (W1 helpers); not used by the EEG pipeline itself |
| `plotstyle.py` | matplotlib style |
| `paper_figs.py` | the paper figure `fig_eeg` (→ `paper/fig_real/`) |
| `figs_classif.py`, `eeg_figs.py` | panel (c) on its own and the supplementary figures E1, E3 (→ `paper/fig_real/`, `figs/`) |

`paper/` and `figs/` are generated and not versioned.
