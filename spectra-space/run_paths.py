"""Per-run artifact routing for results/toys/. Historically: checkpoints (_k*.pt), _model.pt and
_train.json were split by SCSIConfig.encoding into results/toys/chol/ or results/toys/spectral/. Roles
were later swapped (chol and spectral encodings are no longer symmetric in how actively they're worked
on): results/toys/chol/ is now a frozen, self-contained ARCHIVE of every chol-encoding run (848 files,
copied wholesale, no longer updated by training), while spectral-encoding work -- the two actively
maintained runs, iw_ri_d8_inf/iso_diag_d8_inf tag=spectral_follmer -- lives directly at the results/toys/
ROOT, exactly like it did before any chol/spectral split existed. results/toys/spectral/ itself is now
EMPTY and vestigial (kept only so find_run_file's fallback chain stays harmless if anything still looks
there). The HMC/exact reference cache ({toy}_ref.pt) and the merged per-toy {toy}_logscore.json/
{toy}_W1_mean.json also stay at the root since they are NOT per-encoding -- see toy_eval.py's
get_reference docstring and log_score_oracle.py/toy_W1_mean.py's module docstrings on why those are
merged across encodings rather than duplicated."""
import os

ROOT = "results/toys"
SUBDIRS = ("chol", "spectral")


def run_dir(encoding, results_dir=ROOT):
    """Directory a NEW per-run artifact for this encoding should be written to (created if needed).
    chol -> results/toys/chol/ (the frozen archive -- new chol training still lands there, consistent
    with every existing chol file already living there). spectral -> results/toys/ ROOT directly (the
    actively-maintained encoding's files live unprefixed at the root, matching the current two production
    runs -- results/toys/spectral/ is no longer written to)."""
    d = results_dir if encoding == "spectral" else f"{results_dir}/chol"
    os.makedirs(d, exist_ok=True)
    return d


def find_run_file(filename, results_dir=ROOT):
    """Locate an EXISTING per-run artifact by filename alone, checking both encoding subdirs then the
    flat root (runs saved before this split, or not yet migrated). Returns the root path if found
    nowhere, so the caller still gets an ordinary, clear FileNotFoundError instead of a routing one."""
    for sub in SUBDIRS:
        p = f"{results_dir}/{sub}/{filename}"
        if os.path.exists(p):
            return p
    return f"{results_dir}/{filename}"


def list_run_dirs(results_dir=ROOT):
    """Every directory that may hold per-run artifacts: both encoding subdirs plus the flat (legacy) root."""
    return [f"{results_dir}/{sub}" for sub in SUBDIRS] + [results_dir]
