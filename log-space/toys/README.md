# d=8 toy experiments: train, evaluate, plot

Setup (Python 3.11; a CUDA GPU is needed for training and evaluation, not for plotting), once:

```bash
cd log-space
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from `log-space/toys` inside this venv (`source ../.venv/bin/activate`). Results go to `results/toys_M=-1/`, figures and tables to `figs/`, and logs to `../logs/`.

**What is versioned.** The repository holds the code and the result files of the seven production runs: `{STEM}_train.json`, `{toy}_logscore.json` and `{toy}_W1.json`. They are enough to rebuild the figure and the tables (step 4) without a GPU. The trained models, checkpoints, reference posteriors and IW-ML fits (`*.pt`) are not published. Steps 1–3 regenerate them, and the reference posteriors `{toy}_ref.pt` and the IW-ML fits `{toy}_iwfit_ref.pt` are rebuilt automatically when missing. On an already-versioned run, `toy_oracle_log_score.py` and `toy_W1_mean.py` skip the values already in its JSON files (delete them first to recompute from scratch); `toy_eval.py` always recomputes.

## Plotting the paper results without retraining

The result files of the seven production runs are included in `results/toys_M=-1/`: `{STEM}_train.json` (log-score and W1 trajectories along training), `{toy}_logscore.json` (log-scores at the last checkpoint) and `{toy}_W1.json` (W1 at the last checkpoint). The figure and the tables are built from these files alone, so they can be reproduced right after cloning, without training a model and without a GPU:

```bash
python toy_logscore_w1_curve.py     # figure: figs/logscore_w1_vs_k_d8.png (also --mean, --no_hetero)
python toy_paper_table.py           # paper table: figs/toy_paper_table.txt
python toy_w1_tables.py             # W1 tables: figs/toy_w1_table_{mean,median}.{md,tex}
```

Steps 1–3 below regenerate these JSON files from scratch (training, then evaluation).

## The seven production runs

The toys are defined in `toys.py` (`TOYS`). Each toy has one production run, `{toy}_{tag}`.

| toy | tag | training |
|---|---|---|
| `iw_ri_d8_inf`, `iw_nonri_d8_inf`, `factor_d8_inf`, `diag_d8_inf`, `hetero_d8_inf` | `mlp_w512_d4` | 2 stages, k = 0..1000 |
| `iso_diag_d8_inf` | `mlp_w1024_d4` | 2 stages, `--hidden 1024` |
| `ar1_d8_inf` | `resnet_w1024_d4` | 3 stages, k = 0..2000, `--arch resnet --hidden 1024` |

The scripts look up the tag of each toy in the `TAGS` dict of `toy_eval.py` and `toy_oracle_log_score.py`.

## 1. Train: `toy_train.py`

```bash
T=factor_d8_inf; TAG=mlp_w512_d4
python toy_train.py --toy $T --tag $TAG > ../logs/${T}_${TAG}_train.log 2>&1        # stage 1: k=0..700, cosine outer lr
python toy_train.py --toy $T --tag $TAG --n_outer 1000 \
       --resume results/toys_M=-1/${T}_${TAG}_k700.pt > ../logs/${T}_${TAG}_train_k700-1000.log 2>&1   # stage 2: constant lr 5e-5
```

The AR(1) toy has a third stage: `--n_outer 2000 --resume ..._k1000.pt`. On `--resume`, the architecture is read from the checkpoint.

The run is fully seeded. The weights, the normalisation, the three generator states and the outer-lr schedule (`lr_outer`) are stored in every checkpoint. A rerun with the same command reproduces the checkpoints bit for bit.

Outputs (`STEM={toy}_{tag}`):

| file | content |
|---|---|
| `{STEM}_k{k}.pt` | checkpoint at outer iteration k |
| `{STEM}_model.pt` | the `best_k` checkpoint |
| `{STEM}_train.json` | `val_curve_orig` (callback validation: 300 pairs, J=64), `best_k`, `fit_sec`, `M`, `N`, `d` |

`best_k` is the checkpoint the run reports. In infinite-data mode (`toy.M == -1`, the seven production runs), `best_k` is the last checkpoint (k=1000, k=2000 for AR(1)): there is no overfitting, and the J=64 callback score is too noisy to select on. This is imposed whatever the command line says. With finite data, `best_k` is the last checkpoint with `--last_checkpoint`, and the maximum of the callback validation curve otherwise.

## 2. Log-score: `toy_oracle_log_score.py`

```bash
python toy_oracle_log_score.py --toys $T > ../logs/logscore_d8_J2000000_$T.log 2>&1
```

This script computes the predictive log-score of `Ce_val` given `Ce_cal`. It uses J = 2·10⁶ posterior draws on the first 10 validation pairs of `toy_train.py` (`Generator(1000+seed)`). It scores three methods:
- the oracle: closed form for the IW toys, HMC Monte Carlo otherwise;
- IW-ML;
- SC-SI, on the checkpoint grid plus `best_k`.

The script is resumable. Scores already on disk are skipped.

| file | content |
|---|---|
| `{toy}_logscore.json` | `toy`, `n_val`, `J`, `N`, `d`, `oracle` (and `oracle_exact` for the IW toys), `iw_ml`, `scsi` = {`tag`, `k`=best_k, `mean`, `se`, `per_task`, `ess`, `sec`} |
| `{STEM}_train.json` | adds `val_curve` = {k: SC-SI log-score}, `val_curve_n_val`, `val_curve_J` |

## 3. W1: `toy_W1_mean.py` (trajectory) and `toy_eval.py` (best_k)

```bash
python toy_W1_mean.py --toys $T > ../logs/toy_W1_mean_$T.log 2>&1      # every checkpoint of the grid
python toy_eval.py --toy $T > ../logs/${T}_bestk_eval.log 2>&1          # best_k, IW-ML and the oracle floor
```

W1 is computed per test task (128 tasks) and per statistic, between the method's posterior and the oracle reference posterior (HMC, cached in `{toy}_ref.pt`), in units of the reference sd. There are six statistics: `logdet`, `logcond`, `top_share`, `ldv_top`, `ldv_bot`, `corr01`.

The per-task values are then aggregated by the mean or the median over the tasks. The value at a checkpoint is the average over the six statistics. A task where a statistic is degenerate is left out (`corr01` on the diagonal toy).

| script | writes |
|---|---|
| `toy_W1_mean.py` | in `{STEM}_train.json`: `w1_curve_mean`, `w1_curve_median` = {k: value}, `w1_curve_n_test`, `w1_curve_n_ref`, `w1_curve_J` |
| `toy_eval.py` | `{toy}_W1.json`: `mean` and `median`, each with `oracle` (sampling floor: two halves of the reference), `iw_ml` and `scsi` {`tag`, `k`}. Each entry holds `value` and `detail` = {stat: {`w1`, `std`}}. |

`toy_eval.py` also writes per-checkpoint diagnostics, `{STEM}_k{k}eval_metrics.json` and `{STEM}_k{k}eval_stats.npz`. It never modifies `_train.json`.

`--k K` evaluates another checkpoint. `{toy}_W1.json` is only written for `best_k`.

## 4. Figure and tables

```bash
python toy_logscore_w1_curve.py               # figs/logscore_w1_vs_k_d8.png          (W1 median)
python toy_logscore_w1_curve.py --mean        # figs/logscore_w1_vs_k_d8_meanW1.png   (W1 mean)
python toy_logscore_w1_curve.py --no_hetero   # figs/logscore_w1_vs_k_d8_nohetero.png
python toy_paper_table.py                     # figs/toy_paper_table.txt   (paper table, LaTeX rows)
python toy_w1_tables.py                       # figs/toy_w1_table_{mean,median}.{md,tex}   (W1 only)
```

**The figure** has one column per toy:
- top row: the SC-SI log-score minus the oracle along k (`val_curve`);
- bottom row: the SC-SI W1 along k (`w1_curve_*`);
- dashed line: IW-ML, from `{toy}_logscore.json` and `{toy}_W1.json`.

**The paper table** (`toy_paper_table.py`) only reads `{toy}_logscore.json` and `{toy}_W1.json`, so it reports SC-SI at `best_k`. Its columns are:
- the log-score gap, oracle − method, paired over the 10 pairs;
- the six W1 statistics and their average.

It writes two tables: W1 median and W1 mean. For the IW toys, the IW-ML gap is the closed-form expected gap `IW_EXACT_EXPECTED` (−0.001), defined in `toy_logscore_w1_curve.py`.

Nothing is recomputed in step 4. Steps 1–3 produce every number in the figure and the tables.

## Finite data: the M=5000 runs

`toys.py` also defines `{toy}_d8_M5000`, the finite-data counterparts of the production toys. They share the same prior and N=40, but train on one fixed ensemble of M=5000 noisy Ce instead of a fresh ensemble at every E-step. They use the same architectures and one training stage only (k=0..700, cosine outer lr), and write to a separate folder with `--out_dir`:

```bash
bash train_M5000.sh     # -> results/toys_M=5000/, logs in ../logs/M=5000/ (training only)
```

The script currently trains `iw_nonri`, `factor` and `diag`, one run at a time, in about 1 h each on one GPU. With finite data, `--resume` redraws the same fixed training ensemble as the original run (fresh `Generator(1000+seed)`), so a resumed run keeps training on the same M Ce. With finite data, `--last_checkpoint` is not imposed. The evaluation of these runs (same tables as M=inf) is still to be set up.

## Effect of the SDE time step: `toy_dt_study.py`

The sampler integrates the Föllmer SDE by Euler–Maruyama on t ∈ [0, 1] with Δt = 1/`n_steps`. Every production result uses 64 steps (`SCSIConfig.n_sde_steps`). `toy_dt_study.py` re-evaluates a run at its `best_k` with other step counts, at evaluation only. It reuses the production protocols: the J=2·10⁶ log-score of step 2 and the W1 of step 3.

```bash
python toy_dt_study.py --toys iw_ri_d8_inf,iw_nonri_d8_inf,factor_d8_inf,iso_diag_d8_inf,ar1_d8_inf,diag_d8_inf --n_steps 32,128
# -> results/toys_M=-1/{toy}_dt.json and figs/toy_dt_table.txt   (--table_only: rebuild the table from the JSON files)
```

With `--n_steps 64`, it reproduces the production values exactly. The table's Δt = 1/64 rows are therefore read from `{toy}_logscore.json` and `{toy}_W1.json`. The study was run for Δt = 1/32 and 1/128 on six toys (all but hetero), about 40 min on one GPU.

For each toy and Δt, the table gives four values:
- **Gap to the oracle**: oracle − SC-SI.
- **Difference with 1/64**: SC-SI at Δt − SC-SI at 1/64. The oracle cancels in this difference.
- **W1**: the average over the six statistics, median and mean over the test tasks.

Both log-score columns are paired over the 10 validation pairs, as mean ± se.

**Result**: every difference with 1/64 lies within one standard error, except iso-diag and AR(1) at 1/128. Those two are at about 2 se, but only 0.005 and 0.002 nat. The W1 values move by 0.001 to 0.005. Δt = 1/64 is converged.

On the two IW toys, the log-score estimator is noisy: the importance ESS is 3 to 12 on some pairs, even with J=2·10⁶. A change of seed alone, at fixed Δt, gives paired differences of ± 0.17.

## Full pipeline for one toy

`retrain_iso_d8.sh` runs steps 1 to 4 for the iso-diagonal toy, about 3 h on one GPU. The previous outputs, if any, are first moved to `results/toys_M=-1/archive_iso_oldrng/`. Use it as a template for the other toys.
