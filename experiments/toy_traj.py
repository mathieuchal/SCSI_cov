"""Accuracy of SC-SI versus the reference posterior along the self-consistent (EM) iterations, next to the validation score
that the checkpoint-selection rule uses.  Needs the per-iteration checkpoints saved by toy_train.py."""
import glob, json, os, re, sys
import numpy as np, torch
from scipy import stats as sps
from toy_eval import make_testset, stats_of, load_scsi, w1_norm, STATS
from toys import make_toy
from scsi import DT

torch.set_num_threads(6)
out = {}
for name in sys.argv[1:]:
    files = sorted(glob.glob(f"results/toys/{name}_k*.pt"), key=lambda p: int(re.search(r"_k(\d+)\.pt", p).group(1)))
    if len(files) < 3:
        print("skip", name, len(files)); continue
    toy = make_toy(name)
    C, Ce = make_testset(toy, 128)
    n_obs, J = 64, 512
    Ce = Ce[:n_obs]
    ref = torch.load(f"results/toys/{name}_ref.pt")["Cs"][:n_obs, :1024].to(DT)
    Sref = {k: v.numpy() for k, v in stats_of(ref, Ce).items()}
    tr = json.load(open(f"results/toys/{name}_train.json"))
    res = {"val_curve": {int(k): v for k, v in tr["val_curve"].items()}, "best_k": tr["best_k"], "w1": {}, "sd_ratio": {}}
    g = torch.Generator().manual_seed(11)
    for f in files:
        k = int(re.search(r"_k(\d+)\.pt", f).group(1))
        m, _ = load_scsi(toy, f, 2)
        Cs = m.sample_posterior(Ce, J, gen=g)
        S = {kk: v.numpy() for kk, v in stats_of(Cs, Ce).items()}
        res["w1"][k] = {s: float(w1_norm(S[s], Sref[s]).mean()) for s in STATS}
        res["sd_ratio"][k] = {s: float((S[s].std(1) / Sref[s].std(1)).mean()) for s in STATS}
        print(name, "k", k, "mean W1/sd = %.3f" % np.mean(list(res["w1"][k].values())), "mean sd-ratio = %.3f" % np.mean(list(res["sd_ratio"][k].values())), flush=True)
    # sampling noise floor for W1 with these sample sizes (two halves of the reference)
    res["floor"] = float(np.mean([w1_norm(Sref[s][:, :512], Sref[s][:, 512:]).mean() for s in STATS]))
    out[name] = res
json.dump(out, open("results/toys/trajectory.json", "w"), indent=1, default=float)
