# d=8 toy experiments: train, evaluate, plot

Setup (Python 3.11; a CUDA GPU is needed for training and evaluation, not for plotting), once:

```bash
cd log-space
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from `log-space/toys` inside this venv (`source ../.venv/bin/activate`). Logs go to `../logs/`.

## Plotting the paper results without retraining

The result files of the production runs are versioned: `{STEM}_train.json` (log-score and W1 trajectories along training), `{toy}_logscore.json` (log-scores at `best_k`) and `{toy}_W1.json` (W1 at `best_k`), with `STEM={toy}_{tag}`. The figure and the tables are built from these files alone, so they can be reproduced right after cloning, without training a model and without a GPU:

```bash
python toy_logscore_w1_curve.py     # figure: figs/logscore_w1_vs_k_d8.png (also --mean, --no_hetero)
python toy_paper_table.py           # paper table: figs/toy_paper_table.txt
python toy_w1_tables.py             # W1 tables: figs/toy_w1_table_{mean,median}.{md,tex}
```

The trained models, checkpoints, reference posteriors and IW-ML fits (`*.pt`) are not versioned. Steps 1–3 below regenerate everything from scratch: training, then evaluation. The reference posteriors `{toy}_ref.pt` and the IW-ML fits `{toy}_iwfit_ref.pt` are rebuilt automatically when missing.

## Code layout

Each setting is defined in one place:

| file | defines |
|---|---|
| `paths.py` | the production runs: data regimes (`inf`, `M5000`), their folders, the architecture (`tag`) of each toy (`ARCH`), the evaluation grid `GRID` |
| `toys.py` | the toy priors (`TOYS`) and their oracle samplers (direct or HMC) |
| `toy_train.py`, `train_inf.sh`, `train_M5000.sh` | training (SC-SI, self-consistent EM) |
| `toy_data.py` | the evaluation data: validation pairs of the log-score, the 128 test tasks of W1, the IW-ML fit ensemble |
| `logscore.py`, `w1.py` | the scoring of each method (oracle, IW-ML, SC-SI), with its sampling seeds |
| `runs.py` | loading a checkpoint, listing the checkpoints of a run |
| `toy_logscore_{oracle,scsi}.py`, `toy_W1_{oracle,scsi}.py` | the evaluation scripts (steps 2 and 3) |
| `toy_logscore_w1_curve.py`, `toy_logscore_w1_curve_spectral.py`, `toy_paper_table.py`, `toy_w1_tables.py` | the figures and the tables (step 4) |

**Data regimes.** A toy name ends with its regime: `{base}_d8_inf` for infinite data (a fresh training ensemble at every E-step, `toy.M == -1`), `{base}_d8_M5000` for one fixed training ensemble of M=5000 noisy Ce. Results go to `results/toys_M=-1/` or `results/toys_M=5000/`, figures and tables to `figs/` or `figs_M=5000/`. The scripts that handle one toy derive the folder from its name. The evaluation and table scripts take `--data inf|M5000` (default `inf`). By default the evaluation scripts process every run of that regime present on disk; `--toys a,b` restricts them.

## The seven production runs (M=inf)

| toy | tag | training |
|---|---|---|
| `iw_ri_d8_inf`, `iw_nonri_d8_inf`, `factor_d8_inf`, `diag_d8_inf`, `hetero_d8_inf` | `mlp_w512_d4` | 2 stages, k = 0..1000 |
| `iso_diag_d8_inf` | `mlp_w1024_d4` | 2 stages, `--hidden 1024` |
| `ar1_d8_inf` | `resnet_w1024_d4` | 3 stages, k = 0..2000, `--arch resnet --hidden 1024` |

## 1. Train: `toy_train.py`

```bash
bash train_inf.sh       # the seven M=inf runs, every stage, one run at a time (DRY=1: print the commands only)
```

`train_inf.sh` runs, for each toy, the commands below. For one toy:

```bash
T=factor_d8_inf; TAG=mlp_w512_d4; R=results/toys_M=-1
python toy_train.py --toy $T --tag $TAG > ../logs/${T}_${TAG}_train.log 2>&1                     # stage 1: k=0..700, cosine outer lr
python toy_train.py --toy $T --tag $TAG --n_outer 1000 --resume $R/${T}_${TAG}_k700.pt \
       > ../logs/${T}_${TAG}_train_k700-1000.log 2>&1                                             # stage 2: constant lr 5e-5
```

Add `--hidden 1024` for `iso_diag_d8_inf`, and `--arch resnet --hidden 1024` for `ar1_d8_inf`, which has a third stage: `--n_outer 2000 --resume $R/${T}_${TAG}_k1000.pt`. On `--resume`, the architecture is read from the checkpoint.

The M=5000 runs are trained by `bash train_M5000.sh` (one stage, see below).

The run is fully seeded. The weights, the normalisation, the three generator states and the outer-lr schedule are stored in every checkpoint. A rerun with the same commands reproduces the checkpoints bit for bit on the same hardware and software versions (`requirements.txt`). On another GPU or with other library versions, floating-point reductions may differ slightly, so the checkpoints and the numbers derived from them can differ at the level of rounding errors. With finite data, `--resume` redraws the same fixed training ensemble as the original run.

| file | content |
|---|---|
| `{STEM}_k{k}.pt` | checkpoint at outer iteration k (about every 20 iterations) |
| `{STEM}_model.pt` | the `best_k` checkpoint |
| `{STEM}_train.json` | `best_k`, `val_curve_orig`, `fit_sec`, `M`, `N`, `d` |

`best_k` is the checkpoint the run reports. With infinite data, it is the last checkpoint (k=1000, k=2000 for AR(1)): there is no overfitting, and this is imposed whatever the command line says. With finite data, it is chosen at evaluation by the log-score (step 2).

`val_curve_orig` is the validation score of the training callback (300 pairs, J=64 draws). It is too noisy to select on and is not used by the evaluation. It is kept because its draws are part of the training random stream: removing it would change every checkpoint.

## 2. Log-score: `toy_logscore_oracle.py` and `toy_logscore_scsi.py`

```bash
python toy_logscore_oracle.py > ../logs/logscore_oracle.log 2>&1   # oracle and IW-ML
python toy_logscore_scsi.py   > ../logs/logscore_scsi.log 2>&1     # SC-SI: trajectory, best_k, value at best_k
```

For the M=5000 runs, the same commands with `--data M5000`. This also holds for steps 3 and 4.

The predictive log-score of a method is `log (1/J) Σ_j p(Ce_val | C_j)`, with `C_j` drawn from its posterior given `Ce_cal`, averaged over 10 validation pairs (`Ce_cal`, `Ce_val`) that share a latent C. Every method is scored on the same pairs with the same J = 2·10⁶ draws, so the gap to the oracle is paired pair by pair. The pairs are defined in `toy_data.py`:
- M=inf: the first validation pairs of `toy_train.py`, replayed from `Generator(1000+seed)`;
- finite data: independent pairs, `Generator(2000)` to select `best_k` and `Generator(2001)` to report it, so the reported value carries no selection bias.

`toy_logscore_oracle.py` scores the oracle (Monte Carlo, by direct draws or HMC, plus the closed form `oracle_exact` for the IW toys) and IW-ML (an inverse-Wishart prior fitted by maximum likelihood). `toy_logscore_scsi.py` then:
1. scores SC-SI on the checkpoints of `--ks`: `grid` (the default for M=inf, the checkpoints of `paths.GRID`), `all` (the default for finite data), `best`, or a list;
2. sets `best_k`: the run's own with M=inf, the argmax of this trajectory with finite data;
3. scores the run at `best_k` on the report pairs. With M=inf the two sets of pairs are the same, and `best_k` is scored once.

| file | content |
|---|---|
| `{toy}_logscore.json` | `toy`, `n_val`, `J`, `N`, `d`, `val_seed` (null = replayed pairs), `oracle` (and `oracle_exact`), `iw_ml`, `scsi` = {`tag`, `k`=best_k, `mean`, `se`, `per_task`, `ess`, `sec`} |
| `{STEM}_train.json` | adds `val_curve` = {k: SC-SI log-score}, `val_curve_per_task`, `val_curve_J`, `val_curve_n_val`, `val_curve_val_seed`; with finite data also `val_ref` (oracle and IW-ML on the selection pairs) and `best_k_orig` (the callback choice, for reference) |

A point of the trajectory costs about one GPU minute.

## 3. W1: `toy_W1_oracle.py` and `toy_W1_scsi.py`

```bash
python toy_W1_oracle.py > ../logs/W1_oracle.log 2>&1   # oracle floor and IW-ML (builds {toy}_ref.pt if missing)
python toy_W1_scsi.py   > ../logs/W1_scsi.log 2>&1     # SC-SI: trajectory and value at best_k
```

W1 is computed on 128 test tasks (`toy_data.make_testset`, `Generator(777)`), against the oracle reference posterior (direct draws or HMC, 2048 draws per task, cached in `{toy}_ref.pt`). For each task and each of six statistics of C (`logdet`, `logcond`, `top_share`, `ldv_top`, `ldv_bot`, `corr01`), W1(method, oracle) is divided by the reference sd. The per-task values are aggregated by the mean or the median over the tasks, and the value is the average over the statistics. A task where a statistic is degenerate is left out (`corr01` on the diagonal toy).

`toy_W1_oracle.py` scores the oracle sampling floor (W1 between the two halves of the reference draws) and IW-ML. `toy_W1_scsi.py` scores SC-SI on the checkpoints of `--ks` (same choices as in step 2) and writes the point at `best_k`, with its per-statistic detail, to `{toy}_W1.json`. It reads `best_k` from `_train.json`: with finite data, run step 2 first.

| file | content |
|---|---|
| `{toy}_W1.json` | `toy`, `n_test`, `n_ref`, `J`, `N`, `d`, then `mean` and `median`, each with `oracle`, `iw_ml` and `scsi` {`tag`, `k`=best_k}; each entry holds `value` and `detail` = {stat: {`w1`, `std`}} |
| `{STEM}_train.json` | adds `w1_curve_mean`, `w1_curve_median` = {k: value}, `w1_curve_n_test`, `w1_curve_n_ref`, `w1_curve_J` |

## 4. Figure and tables

```bash
python toy_logscore_w1_curve.py               # figs/logscore_w1_vs_k_d8.png          (W1 median)
python toy_logscore_w1_curve.py --mean        # figs/logscore_w1_vs_k_d8_meanW1.png   (W1 mean)
python toy_logscore_w1_curve.py --no_hetero   # figs/logscore_w1_vs_k_d8_nohetero.png
python toy_logscore_w1_curve_spectral.py      # figs/logscore_w1_vs_k_d8_spectral.png   (with the spectral runs)
python toy_paper_table.py                     # figs/toy_paper_table.txt   (paper table, LaTeX rows)
python toy_w1_tables.py                       # figs/toy_w1_table_{mean,median}.{md,tex}   (W1 only)
```

With `--data M5000`, the tables go to `figs_M=5000/`. The figure is drawn for M=inf only.

**The figure** (M=inf) has one column per toy:
- top row: the SC-SI log-score minus the oracle along k (`val_curve`);
- bottom row: the SC-SI W1 along k (`w1_curve_*`);
- dashed line: IW-ML, from `{toy}_logscore.json` and `{toy}_W1.json`.

**The paper table** reads `{toy}_logscore.json` and `{toy}_W1.json`, so it reports SC-SI at `best_k`. Its columns are the log-score gap, oracle − method, paired over the 10 pairs, then the six W1 statistics and their average. It writes two tables, W1 median and W1 mean. For the IW toys, the IW-ML gap is the closed-form expected gap `IW_EXACT_EXPECTED` (−0.001), defined in `toy_logscore_w1_curve.py`.

**The spectral figure** (`toy_logscore_w1_curve_spectral.py`) is the `--no_hetero` figure with, in green, the SC-SI runs of the spectral-Föllmer variant in the IW-RI and iso-diagonal columns. These runs were trained and evaluated in [`spectra-space`](../../spectra-space); only their curves are kept here, in `results/toys_M=-1/spectra/`. Their log-score gap is a difference of means at J=2·10⁵, not a paired difference at J=2·10⁶ like the blue curves, because their files hold no per-pair values.

Nothing is recomputed in step 4: steps 1–3 produce every number, except the spectral curves.

## Finite data: the M=5000 runs

`toys.py` also defines `{toy}_d8_M5000`, the finite-data counterparts of the production toys: same prior, N=40 and architecture, but one fixed training ensemble of M=5000 noisy Ce, and one training stage (k=0..700, cosine outer lr). `train_M5000.sh` trains six of them, all but AR(1), in about 35 min to 1 h each on one GPU.

They go through steps 1 to 4 with `--data M5000`. Two points differ from M=inf:
- `best_k` is chosen by `toy_logscore_scsi.py` (step 2), as the argmax of the log-score trajectory, so step 2 must run before `toy_W1_scsi.py` (step 3);
- by default, both trajectories cover every checkpoint (`--ks all`), not the grid.

The log-score trajectory of the versioned M=5000 runs was computed at a reduced budget, J=2·10⁵ (`toy_logscore_oracle.py --data M5000 --roles select --J 200000` and `toy_logscore_scsi.py --data M5000 --J 200000 --no_report`). The value at `best_k` is at J=2·10⁶ (`toy_logscore_oracle.py --data M5000 --roles report` and `toy_logscore_scsi.py --data M5000 --ks best`). Results and remarks: `M5000_W1_bestk.txt`.

## Effect of the SDE time step: `toy_dt_study.py`

The sampler integrates the Föllmer SDE by Euler–Maruyama on t ∈ [0, 1] with Δt = 1/`n_steps`. Every production result uses 64 steps (`SCSIConfig.n_sde_steps`). `toy_dt_study.py` re-evaluates the M=inf runs at their `best_k` with other step counts, at evaluation only, with the protocols of steps 2 and 3.

```bash
python toy_dt_study.py --n_steps 32,128     # six toys (all but hetero), about 40 min on one GPU
# -> results/toys_M=-1/{toy}_dt.json and figs/toy_dt_table.txt   (--table_only: rebuild the table from the JSON files)
```

With `--n_steps 64`, it reproduces the production values exactly, so the table's Δt = 1/64 rows are read from `{toy}_logscore.json` and `{toy}_W1.json`. For each toy and Δt, the table gives:
- **Gap to the oracle**: oracle − SC-SI.
- **Difference with 1/64**: SC-SI at Δt − SC-SI at 1/64. The oracle cancels in this difference.
- **W1**: the average over the six statistics, median and mean over the test tasks.

Both log-score columns are paired over the 10 validation pairs, as mean ± se.

**Result**: every difference with 1/64 lies within one standard error, except iso-diag and AR(1) at 1/128. Those two are at about 2 se, but only 0.005 and 0.002 nat. The W1 values move by 0.001 to 0.005. Δt = 1/64 is converged.

On the two IW toys, the log-score estimator is noisy: the importance ESS is 3 to 12 on some pairs, even with J=2·10⁶. A change of seed alone, at fixed Δt, gives paired differences of ± 0.17.
