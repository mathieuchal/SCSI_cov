"""Generic per-checkpoint mean-W1 (vs. the oracle reference posterior) computation for a tagged SC-SI
toy-training run. Extends toy_eval.py's final-checkpoint-only W1 metric (stats_of + w1_norm, averaged
over the six marginal statistics STATS) into a convergence curve: every saved EM checkpoint k of a
toy's run is evaluated, not just the one selected for toy_eval.py's report -- giving a curve analogous
to the per-checkpoint log-score curves built in log_score/. Uses the same n_test=128, J=1024 defaults
as toy_eval.py: W1 between empirical distributions doesn't suffer the log-mean-exp/heavy-tail bias that
motivated the huge J used for the oracle log-score, so it converges at the ordinary O(1/sqrt(J)) rate
of a plain distance-between-samples estimator.

Also absorbs what toy_traj.py used to compute separately (deleted; it duplicated this same
make_testset/get_reference/stats_of machinery at a smaller, non-canonical n_test/J): besides the single
mean-W1 number per checkpoint (`checkpoints`), each toy entry also carries `checkpoints_detail` (the W1
AND sd-ratio per individual statistic, per checkpoint -- reusing the very same samples, no extra
resampling) and `floor` (the W1 due to pure reference-sampling noise alone, i.e. between two halves of
the reference draws -- a lower bound below which W1 is just noise, not model error). The IW-ML baseline
gets the matching `iw_ml_w1_detail`.

Driven by w1_mean_dict(toys, tag, ...), which mirrors log_score_oracle.py/log_score_scsi.py's file split:
durable outputs are {toy_name}_W1_mean.json (toy-level, shared across tags -- floor/iw_ml_w1(+detail),
self-describing n_test/n_ref/J/agg/N/d, and a single "scsi" entry for the best checkpoint of THIS call's
tag, "last tag written wins" same as {toy}_logscore.json's own "scsi" slot) and {stem}_train.json's
"w1_curve" field (this tag's full per-checkpoint trajectory, mirroring val_curve/log_score_path there).
Progress during a sweep is kept in a disposable per-(toy,tag) RESUMABILITY CACHE
({stem}_W1_cache.json by default), safe to kill/restart, deleted once merged into the two files above --
but unlike log_score_scsi.py's equivalent cache, a fresh call first seeds its starting point from
whatever's ALREADY durably recorded in those two files (not from scratch), so a repeat call with no new
checkpoints to resample is cheap rather than redoing the whole sweep.
"""
import gc, json, os, time
import numpy as np, torch
from toy_eval import make_testset, stats_of, get_reference, load_scsi, sigma_to_cov, w1_norm, STATS
from covutils import IWPrior
from scsi import Chol, sample_covariance_from_factors
from toys import make_toy
from run_paths import list_run_dirs, find_run_file


def list_checkpoints(stem, results_dir="results/toys"):
    """All (k, full_path) pairs for {stem}_k*.pt, sorted by k -- searches results/toys/chol/,
    results/toys/spectral/, and the flat root (pre-split runs), in that order."""
    prefix = f"{stem}_k"
    out = []
    for d in list_run_dirs(results_dir):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith(prefix) and f.endswith(".pt"):
                out.append((int(f[len(prefix):-3]), f"{d}/{f}"))
    return sorted(out)


def w1_sd_per_stat(stats, ref_stats, agg="median"):
    """Per-statistic W1-vs-reference (aggregated over test tasks by `agg`: "median", default and
    outlier-robust, or "mean") and per-statistic sd-ratio (std(model)/std(reference), averaged over test
    tasks -- a dispersion-calibration check). stats/ref_stats: dict stat -> (M,J)/(M,J_ref) numpy arrays,
    already produced by stats_of(...).cpu().numpy() -- computed once per Cs and reused here rather than
    recomputed, since stats_of is not free. nanmean/nan{agg}: w1_norm returns NaN for a (stat, task) pair
    whose reference has zero variance (e.g. corr01 for a strictly-diagonal-covariance toy like diag) --
    skipped rather than letting one degenerate stat/task poison the aggregate.
    Returns (w1: {stat: float}, sd_ratio: {stat: float})."""
    f = np.nanmean if agg == "mean" else np.nanmedian
    w1 = {s: float(f(w1_norm(stats[s], ref_stats[s]))) for s in STATS}
    sdr = {s: float((stats[s].std(1) / ref_stats[s].std(1)).mean()) for s in STATS}
    return w1, sdr


def w1_mean_of(stats, ref_stats, agg="median"):
    """Mean, over the six marginal statistics STATS, of each statistic's W1-vs-reference distance (see
    w1_sd_per_stat) -- the single-number summary used for the `checkpoints`/`iw_ml_w1` curves."""
    w1, _ = w1_sd_per_stat(stats, ref_stats, agg)
    return float(np.nanmean(list(w1.values())))


def _seed_from_durable(toy_out_path, train_path):
    """Starting point for a fresh (no cache on disk) call: whatever's already durably recorded for this
    exact (toy, tag) in {toy}_W1_mean.json ("floor"/"iw_ml_w1"/"iw_ml_w1_detail", toy-level so valid
    regardless of which tag wrote them last) and in {stem}_train.json's "w1_curve" ("checkpoints"/
    "checkpoints_detail", specific to THIS tag). Lets a repeat call skip everything already computed
    instead of redoing the full sweep from scratch."""
    entry = {"checkpoints": {}, "checkpoints_detail": {}}
    if os.path.exists(toy_out_path):
        tw = json.load(open(toy_out_path))
        for k in ("floor", "iw_ml_w1", "iw_ml_w1_detail"):
            if k in tw:
                entry[k] = tw[k]
    if os.path.exists(train_path):
        tr = json.load(open(train_path))
        wc = tr.get("w1_curve")
        if wc:
            entry["checkpoints"] = wc.get("checkpoints", {})
            entry["checkpoints_detail"] = wc.get("checkpoints_detail", {})
    return entry


def w1_mean_dict(toys, tag, n_test=128, n_ref=2048, J=1024, m_fit=5000, threads=2,
                  device=None, results_dir="results/toys", agg="median", cache_path=None):
    """For each toy in `toys` (registry names, or already-built Toy instances), compute the
    per-EM-checkpoint mean-W1 of its `tag`-tagged SC-SI run, plus a checkpoint-independent IW-ML
    (conjugate, ML-fitted) baseline, and merge the durable result into {toy_name}_W1_mean.json and
    {stem}_train.json's "w1_curve" -- see module docstring for the exact split and the resumability cache
    (cache_path, default {results_dir}/{stem}_W1_cache.json).

    agg: forwarded to w1_sd_per_stat -- "median" (default) or "mean" over test tasks per statistic.
    Changing it changes the numbers' meaning: it is NOT encoded as a separate key in the two durable
    files, so switching agg for a (toy, tag) already merged under a different agg silently overwrites
    those entries with the new agg's values -- keep agg consistent across calls for the same (toy, tag)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_agg = np.nanmean if agg == "mean" else np.nanmedian

    for t in toys:
        toy = t if hasattr(t, "name") else make_toy(t)
        name = toy.name
        stem = f"{name}_{tag}" if tag else name
        cpath = cache_path or f"{results_dir}/{stem}_W1_cache.json"
        toy_out_path = f"{results_dir}/{name}_W1_mean.json"
        train_path = find_run_file(f"{stem}_train.json", results_dir)

        entry = json.load(open(cpath)) if os.path.exists(cpath) else _seed_from_durable(toy_out_path, train_path)

        ckpts = list_checkpoints(stem, results_dir)
        fully_done = ("floor" in entry and "iw_ml_w1_detail" in entry
                      and all(str(k) in entry["checkpoints_detail"] for k, _ in ckpts))
        if fully_done:
            print(f"{name}: already computed, merging", flush=True)
        else:
            print(f"=== {name} ===", flush=True)
            t0 = time.time()

            C_true, Ce = make_testset(toy, n_test)
            Ce = Ce.to(device)
            ref_C, ref_diag = get_reference(toy, Ce, n_ref, f"{results_dir}/{name}_ref.pt")
            ref_stats = stats_of(ref_C, Ce)
            ref_stats = {s: v.cpu().numpy() for s, v in ref_stats.items()}

            def save():
                json.dump(entry, open(cpath, "w"), indent=1)

            if "floor" not in entry:
                half = ref_stats[STATS[0]].shape[1] // 2
                entry["floor"] = float(np.mean([f_agg(w1_norm(ref_stats[s][:, :half], ref_stats[s][:, half:])) for s in STATS]))
                print(f"  sampling-noise floor = {entry['floor']:.4f}", flush=True)
                save()

            if "iw_ml_w1" not in entry or "iw_ml_w1_detail" not in entry:
                gg = torch.Generator().manual_seed(1000)
                Cho = Chol(toy.d)
                M = toy.M if toy.M > 0 else m_fit
                Sigma_fit = Cho.encode(toy.sample_prior(M, gg)).to(device)
                Ce_tr = sample_covariance_from_factors(Sigma_fit, toy.N, gg)
                iw = IWPrior(Ce_tr, toy.N, iters=800)
                g2 = torch.Generator().manual_seed(99)
                Cs_iw = iw.sample_posterior(Ce, J, gen=g2)
                stats_iw = {s: v.cpu().numpy() for s, v in stats_of(Cs_iw, Ce).items()}
                w1_iw, sdr_iw = w1_sd_per_stat(stats_iw, ref_stats, agg)
                entry["iw_ml_w1"] = float(np.nanmean(list(w1_iw.values())))
                entry["iw_ml_w1_detail"] = {"w1": w1_iw, "sd_ratio": sdr_iw}
                print(f"  IW-ML mean-W1 = {entry['iw_ml_w1']:.4f}", flush=True)
                del Cs_iw
                save()

            for k, path in ckpts:
                key = str(k)
                if key in entry["checkpoints"] and key in entry["checkpoints_detail"]:
                    continue
                t1 = time.time()
                m, _ = load_scsi(toy, path, threads)
                g3 = torch.Generator().manual_seed(99)
                Cs_si = sigma_to_cov(m.sample_posterior_sigma(Ce, J, gen=g3, chunk=4000).to(device))
                stats_si = {s: v.cpu().numpy() for s, v in stats_of(Cs_si, Ce).items()}
                w1_si, sdr_si = w1_sd_per_stat(stats_si, ref_stats, agg)
                entry["checkpoints"][key] = float(np.nanmean(list(w1_si.values())))
                entry["checkpoints_detail"][key] = {"w1": w1_si, "sd_ratio": sdr_si}
                print(f"  k={k:>3}: mean-W1={entry['checkpoints'][key]:.4f}  ({time.time()-t1:.0f}s)", flush=True)
                del m, Cs_si
                gc.collect(); torch.cuda.empty_cache()
                save()

            print(f"{name}: sweep done ({len(entry['checkpoints'])} checkpoints, {time.time() - t0:.0f}s)", flush=True)

        # ---- merge into the two durable files, then drop the disposable cache ----
        # best_k_w1 (this sweep's OWN argmin -- "what if we picked by W1 instead") is kept in w1_curve as
        # a diagnostic, but {toy}_W1_mean.json's "scsi" reports W1 at the checkpoint ACTUALLY SELECTED
        # (train.json's own best_k, chosen by the log-score curve -- the model really used), not best_k_w1.
        best_k_w1 = min(entry["checkpoints"], key=lambda kk: entry["checkpoints"][kk])
        best_k = None

        if os.path.exists(train_path):
            tr = json.load(open(train_path))
            tr["w1_curve"] = {"n_test": n_test, "n_ref": n_ref, "J": J, "agg": agg,
                               "checkpoints": entry["checkpoints"], "checkpoints_detail": entry["checkpoints_detail"],
                               "best_k_w1": int(best_k_w1)}
            json.dump(tr, open(train_path, "w"), indent=1)
            best_k = tr.get("best_k")
        else:
            print(f"  [{name}] WARNING: {train_path} not found, w1_curve left only in {cpath}", flush=True)

        if best_k is None or str(best_k) not in entry["checkpoints"]:
            print(f"  [{name}] WARNING: train.json's best_k ({best_k}) has no W1 entry -- "
                  f"falling back to this sweep's own argmin (best_k_w1={best_k_w1}) for scsi", flush=True)
            best_k = best_k_w1
        key = str(best_k)

        tw = json.load(open(toy_out_path)) if os.path.exists(toy_out_path) else {"toy": name}
        tw.update({"n_test": n_test, "n_ref": n_ref, "J": J, "agg": agg, "N": toy.N, "d": toy.d,
                   "floor": entry["floor"], "iw_ml_w1": entry["iw_ml_w1"], "iw_ml_w1_detail": entry["iw_ml_w1_detail"]})
        tw["scsi"] = {"tag": tag if tag else "model", "value": entry["checkpoints"][key],
                      "kbest": int(best_k), "detail": entry["checkpoints_detail"][key]}
        json.dump(tw, open(toy_out_path, "w"), indent=1)

        if os.path.exists(train_path) and os.path.exists(cpath):
            os.remove(cpath)  # durably merged above -- disposable, same convention as log_score_scsi.py
        print(f"{name}: merged into {toy_out_path}" + (f" and {train_path}" if os.path.exists(train_path) else ""), flush=True)
