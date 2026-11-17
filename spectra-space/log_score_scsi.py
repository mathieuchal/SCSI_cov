"""SC-SI checkpoint log-score resampling, split out of log_score_oracle.py (which now covers only the
oracle/IW-ML side) so each file is driven independently: this one needs a trained run's checkpoints,
the other doesn't. Both share the SAME deterministic (Ce_cal, Ce_val) draw (_draw_validation_pair,
imported from log_score_oracle.py rather than duplicated, so the two can never silently drift out of
sync) -- call log_score_oracle.eval_oracle_IW_log_score(toy_name, n_val=n_val, ...) FIRST, same n_val,
so that call's "oracle" entry already exists in {toy_name}_logscore.json for this module to read the
gap from; this module does not compute the oracle itself."""
import gc, json, os, time
import torch

from scsi import DT
from covutils import posterior_predictive_logscore
from toys import make_toy
from toy_eval import load_scsi
from toy_W1_mean import list_checkpoints
from run_paths import find_run_file
from log_score_oracle import _draw_validation_pair


def eval_scsi_checkpoints_log_score(toy_name, tag="", n_val=10, J=200000, m_inf=5000,
                                     device=None, threads=2, out_dir="results/toys", tmp_path=None):
    """Resamples every saved EM checkpoint of the {toy_name}_{tag} SC-SI run at J, scored against the
    SAME deterministic (Ce_cal, Ce_val) pair (see _draw_validation_pair) that
    eval_oracle_IW_log_score(toy_name, n_val=n_val, ...) used -- so the gap against that call's "oracle"
    entry (already written to {out_dir}/{toy_name}_logscore.json) is apples-to-apples. Call
    eval_oracle_IW_log_score FIRST, same n_val, so that file's "oracle" entry exists to read the gap from
    -- this function does NOT compute the oracle itself.

    Merges the full curve into {stem}_train.json (val_curve <- resampled curve, tagged with its own
    val_curve_n_val/val_curve_J so it's unambiguously identified by those two numbers rather than by a
    name like "J64"/"J256"/"bigJ" -- the old curve is kept alongside as val_curve_orig/
    val_curve_orig_n_val/val_curve_orig_J) and writes a single "scsi" entry into the shared
    {toy_name}_logscore.json: {"tag", "value", "kbest"} for the best (highest-scoring) checkpoint --
    n_val/J (this call's own -- expected to match the "oracle" entry's pair, since oracle_val is read
    straight from it below) are the file's shared top-level fields, not repeated here (see
    log_score_oracle.py's module docstring); this call (re)writes them too, so a resampling run at a
    different J than the oracle's own still ends up reflected at the top level. tmp_path (defaults to
    {out_dir}/{stem}_J{J}_cache.json) is a disposable RESUMABILITY CACHE, not a result file: safe to
    kill/restart mid-sweep (each checkpoint's score is written there
    as soon as it's computed), and deleted automatically once its content has been durably merged into
    {stem}_train.json -- the only lasting outputs of this function are {stem}_train.json and
    {toy_name}_logscore.json."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(threads)
    toy = make_toy(toy_name)
    N = toy.N

    out_path = f"{out_dir}/{toy_name}_logscore.json"
    out = json.load(open(out_path))
    oracle_val = out["oracle"]["value"]

    _, Ce_cal, Ce_val = _draw_validation_pair(toy, n_val, m_inf, device)

    stem = f"{toy_name}_{tag}" if tag else toy_name
    tmp_path = tmp_path or f"{out_dir}/{stem}_J{J}_cache.json"
    best_k, best_score = _full_curve(toy, stem, Ce_cal, Ce_val, N, J, n_val, oracle_val, threads, device, tmp_path, out_dir)
    if best_k is not None:
        out["n_val"] = n_val
        out["J"] = J
        out["scsi"] = {"tag": tag if tag else "model", "value": best_score, "kbest": best_k}
        json.dump(out, open(out_path, "w"), indent=1)
    return best_k, best_score


def _full_curve(toy, stem, Ce_cal, Ce_val, N, J, n_val, oracle_val, threads, device, tmp_path, out_dir):
    """Resamples every saved EM checkpoint of `stem` at J, scored against the same (Ce_cal, Ce_val) pair
    as the oracle. tmp_path is a disposable resumability CACHE (safe to kill/restart mid-sweep, each
    checkpoint's score written there as soon as it's computed) -- once every checkpoint's score has been
    merged into {stem}_train.json below, the cache file is deleted; it is not a result to keep around."""
    toy_name = toy.name
    results = json.load(open(tmp_path)) if os.path.exists(tmp_path) else {"toy": toy_name, "J": J, "checkpoints": {}}
    ckpts = list_checkpoints(stem, out_dir)
    for k, ckpt_path in ckpts:
        key = str(k)
        if key in results["checkpoints"]:
            continue
        t0 = time.time()
        m, kk = load_scsi(toy, ckpt_path, threads)
        gm = torch.Generator().manual_seed(777)
        Sigma_s = m.sample_posterior_sigma(Ce_cal, J, chunk=4000, gen=gm).to(DT).to(device)
        sc = float(posterior_predictive_logscore(Ce_val, N, Sigma_s).mean())
        dt = time.time() - t0
        results["checkpoints"][key] = sc
        json.dump(results, open(tmp_path, "w"), indent=1)
        print(f"  [{toy_name}] k={k:>3}: scsi={sc:9.3f}  gap(scsi-oracle)={sc - oracle_val:+8.3f}  ({dt:.0f}s)", flush=True)
        del m, Sigma_s; gc.collect(); torch.cuda.empty_cache()

    # merge into {stem}_train.json: val_curve <- full curve at this (n_val, J), tagged with those two
    # numbers (val_curve_n_val/val_curve_J) rather than a name -- old curve kept as val_curve_orig,
    # tagged the same way (val_curve_orig_n_val/val_curve_orig_J; defaults to toy_train.py's own
    # (300, 64) for a train.json written before this stamp existed). best_k <- argmax of the new curve.
    train_path = find_run_file(f"{stem}_train.json", out_dir)
    if not os.path.exists(train_path):
        print(f"  [{toy_name}] WARNING: {train_path} not found, full curve left only in {tmp_path}", flush=True)
        return None, None
    tr = json.load(open(train_path))
    if "val_curve_orig" not in tr:
        tr["val_curve_orig"] = tr["val_curve"]
        tr["val_curve_orig_n_val"] = tr.get("val_curve_n_val", 300)
        tr["val_curve_orig_J"] = tr.get("val_curve_J", 64)
    new_curve = {k: results["checkpoints"][k] for k in results["checkpoints"]}
    tr["val_curve"] = new_curve
    tr["val_curve_n_val"] = n_val
    tr["val_curve_J"] = J
    tr["best_k"] = int(max(new_curve, key=new_curve.get))
    # same self-describing {"n_val","J","value"} shape as {toy}_logscore.json's "oracle"/"iw_ml"/"scsi"
    # entries, for a reader of train.json alone (kbest/tag are implicit here: best_k above, stem is the
    # file); named log_score_path (not "scsi", which is {toy}_logscore.json's own, differently-shaped key)
    # since this is the log-score along the checkpoint path, at the one checkpoint (best_k) selected from it
    tr["log_score_path"] = {"n_val": n_val, "J": J, "value": new_curve[str(tr["best_k"])]}
    json.dump(tr, open(train_path, "w"), indent=1)
    print(f"  [{toy_name}] merged full curve into {train_path}, best_k={tr['best_k']}", flush=True)
    os.remove(tmp_path)  # durably merged above -- the cache has done its job, don't leave a stale duplicate on disk
    return tr["best_k"], new_curve[str(tr["best_k"])]
