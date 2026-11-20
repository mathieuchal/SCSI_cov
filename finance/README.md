# Finance experiment: covariance posterior of industry returns

This experiment uses Ken French's 48 industry portfolios (daily value-weighted returns, 1926–2026). Each task is a
63-day window and a random basket of d=12 industries. The SC-SI prior is learned from past windows only and refitted
on three expanding folds; the test months are 1995–2004, 2005–2014 and 2015–2026. The methods are scored over the
next 21 days, on:
- the realised variance of the min-variance portfolio;
- the coverage of the predictive intervals;
- the predictive log-scores.

The model, sampler and scores are in `../log-space/toys` (`scsi.py`, `model.py`, `covutils.py`, `plotstyle.py`),
which the scripts import. Setup: see `../log-space/toys/README.md` (venv in `../log-space/.venv`; a CUDA GPU is
needed for step 1, not for the others). All commands are run from `finance/`, and logs go to `../log-space/logs/`.

## Plotting the paper results without retraining

The result files of the paper run are in `results/` (see step 1). With the data of step 0, the figure and the tables
are rebuilt from them alone, without a GPU:

```bash
python fin_paper_fig.py        # figs/fig_finance.png
python fin_paper_tables.py     # figs/fin_paper_tables.txt
```

The trained checkpoints (`results/*_ckpt/`) are not needed for this; step 1 regenerates them.

## The paper run

`fin_d12_nw63_s0_mlp_w1024_d4_neff03_nfe6_o30_selT` (`RUN` below; `TAG` is the same name without `_selT`):
- an MLP-4 w1024 network, with 30 EM iterations per fold;
- N=19 and N_forward=6, from the coverage calibration (N_eff scale 0.3);
- the checkpoint of each fold maximises the Student-t daily log-score of the fold's validation pairs, which gives
  k = 10 / 4 / 6.

The whole chain, steps 1 to 3, takes about 10 minutes on one A100 GPU (step 1: about 9 minutes).

## 0. Data: `prepare_finance.py`

```bash
python prepare_finance.py      # ff48.zip -> 48_Industry_Portfolios_Daily.csv -> data/ff48_vw_daily.pkl
```

The script parses `ff48.zip`, French's file `48_Industry_Portfolios_daily_CSV.zip`, and downloads it from French's
data library only when it is missing. The paper uses the file downloaded on 2026-11-04: 26317 days, 1926-07-01 to
2026-08-31. French's library is refreshed periodically and past values can be revised, so a fresh download can
differ from this snapshot, and so can every number derived from it.

## 1. Training and evaluation: `fin_exp.py`

```bash
F="--hidden 1024 --depth 4 --neff_scale 0.3 --nfe_min 3 --n_outer 30 --tag _mlp_w1024_d4_neff03_nfe6_o30"
python -u fin_exp.py $F > ../log-space/logs/fin_mlp_w1024_d4_neff03_nfe6_o30.log 2>&1                        # trains the 3 folds
python -u fin_exp.py $F --select tscore > ../log-space/logs/fin_mlp_w1024_d4_neff03_nfe6_o30_selT.log 2>&1   # selection + test evaluation
```

The first command trains the three folds. It saves the checkpoints in `results/{TAG}_fold{i}_ckpt/` and the
validation curves in `results/{TAG}_fold{i}_val.json`. It also writes an evaluation at the PIT-KS-selected
checkpoints (`results/{TAG}_results.json`, `_arrays.npz`), which the paper does not use. If the checkpoints of a fold
are already on disk, it loads them instead of training.

The second command reuses these checkpoints without retraining. For each fold, it scores every checkpoint by the
Student-t daily log-score of the validation pairs (nu is fitted per fold on the same pairs), selects the best one and
evaluates it on the test months. It adds this validation curve to `{TAG}_fold{i}_val.json` (`tscore_nu7`) and writes:

| file | content |
|---|---|
| `results/{RUN}_results.json` | summary: N, N_forward, number of test cases, min-variance table (`mv`), log-scores, coverage |
| `results/{RUN}_arrays.npz` | per-test-case arrays: realised variances `var_*`, PITs `pit_*`, log-scores `ll_*` (tempered), `dl_*` (Gaussian daily), `dt_*` (Student-t daily), `nu_hat`, dates and folds |

## 2. OAS / LW with a tuned sample size: `fin_shrink_n_tune.py`

```bash
python fin_shrink_n_tune.py > ../log-space/logs/fin_shrink_n_tune_selT.log 2>&1   # -> results/{RUN}_shrinkn_tuned.npz
```

For each fold, this script chooses the sample size n given to OAS and LW-NLS on the validation pairs, then applies it
to the test cases. Panel (a) of the figure uses OAS with this tuned n as its reference.

## 3. Figure and tables

```bash
python fin_paper_fig.py        # figs/fig_finance.png
python fin_paper_tables.py     # figs/fin_paper_tables.txt (LaTeX: estimators, (a), (b))
```

Both scripts only read the files of steps 1–2 (and, for the tables, the data of step 0); nothing is recomputed except
bootstrap intervals with a fixed seed.

**The figure** has four panels:
- (a) min-variance portfolio, relative to OAS with tuned n;
- (b) 12-month rolling coverage of the 50/80/95% intervals;
- (c) tempered log-score gain;
- (d) daily log-score gains, Gaussian and Student-t.

**The tables:**
- the estimator table and table (b) use OAS with n = N = 19 as the reference, as stored by `fin_exp.py`;
- table (a) reports coverage overall and by tercile of market volatility, and the three log-score gains over the
  sample covariance.

## Reproducibility

Every step is seeded. Rerun from scratch on 2026-11-20 (A100, the versions of `../log-space/requirements.txt`), the
chain above reproduces the paper run bit for bit: the 33 checkpoints of the three folds, the validation curves, the
result files of steps 1–2, the figure and the tables. `prepare_finance.py` rebuilds `data/ff48_vw_daily.pkl`
exactly from the 2026-11-04 `ff48.zip`. On another GPU or with other library versions, floating-point reductions may
differ slightly, so the results can differ at the level of rounding errors.
