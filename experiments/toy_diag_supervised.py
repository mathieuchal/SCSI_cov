"""Diagnostic: is the d=20 discrepancy caused by EM self-consistency instability, or by the underlying
SI regression (network architecture/optimization) being unable to fit the conditional target at all?

Trains the SAME architecture directly on TRUE (C, Ce) pairs via SCSI.fit_supervised: every training pair
uses a genuinely clean C drawn straight from the toy's true prior, paired with one fresh Wishart
re-corruption -- this is exactly the paper's "supervised conditional sampling" control (bench_iw.py's
--supervised flag), which isolates the regression/SDE machinery from the self-consistent
reconstruct/re-corrupt/refit loop entirely. No reconstruction, no accumulated re-corruption error, no
outer-iteration dynamics -- if this control ALSO mis-fits the statistics that broke in the self-consistent
run, the SI regression itself (capacity or optimization) is the bottleneck, not the EM loop. If it fits
them well, the EM loop is the cause.

Runs several independent supervised fits at increasing step budgets (fresh network each time) to also
check whether more optimization closes the gap or whether it plateaus at a wrong answer (capacity
ceiling) -- distinguishing "needs more training" from "cannot represent this at all".
"""
import argparse, json, time
import numpy as np, torch

from scsi import SCSI, SCSIConfig, wishart_channel, DT
from toys import make_toy
from toy_eval import make_testset, get_reference, stats_of, w1_norm, pit_vs_ref, ks_unif, STATS

ap = argparse.ArgumentParser()
ap.add_argument("--toy", required=True)
ap.add_argument("--steps", default="4000,20000,52000", help="comma-separated step budgets, each an independent fresh-network run")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--n_test", type=int, default=128)
ap.add_argument("--n_ref", type=int, default=2048)
ap.add_argument("--J", type=int, default=1024)
ap.add_argument("--seed", type=int, default=2024)
ap.add_argument("--activation", default="silu", choices=["silu", "relu", "gelu", "tanh"])
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--depth", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
toy = make_toy(a.toy, device=dev)
print(f"toy={a.toy} device={dev} d={toy.d} N={toy.N} M={toy.M} activation={a.activation} hidden={a.hidden} depth={a.depth}", flush=True)

# same test set / reference as toy_eval.py (same seed, so directly comparable to the self-consistent run)
C_true, Ce = make_testset(toy, a.n_test)
ref_C, ref_diag = get_reference(toy, Ce, a.n_ref, f"results/toys/{toy.name}_ref.pt")
print("reference:", ref_C.shape, {k: v for k, v in ref_diag.items() if k != "rhat"}, flush=True)
Sref = {s: v.cpu().numpy() for s, v in stats_of(ref_C, Ce).items()}


def evaluate(model, gen):
    Cs = model.sample_posterior(Ce, a.J, gen=gen)
    S = {s: v.cpu().numpy() for s, v in stats_of(Cs, Ce).items()}
    row = {}
    for s in STATS:
        u = pit_vs_ref(S[s], Sref[s])
        row[s] = dict(w1=float(w1_norm(S[s], Sref[s]).mean()),
                      bias_z=float(((S[s].mean(1) - Sref[s].mean(1)) / Sref[s].std(1)).mean()),
                      sd_ratio=float((S[s].std(1) / Sref[s].std(1)).mean()),
                      pit_ks=ks_unif(u))
    return row


results = {}
for steps in [int(x) for x in a.steps.split(",")]:
    print(f"\n=== supervised control, {steps} gradient steps (fresh network, fresh {200_000}-covariance pool of TRUE draws) ===", flush=True)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    cfg = SCSIConfig(N=toy.N, d=toy.d, kappa=1.0, threads=a.threads, device=str(dev), seed=a.seed, pool_init=200_000,
                     activation=a.activation, hidden=a.hidden, depth=a.depth)
    model = SCSI(cfg)

    def sampler(n, gen=gen):
        return toy.sample_prior(n, gen)

    Ce_norm = wishart_channel(toy.sample_prior(2000, gen), toy.N, gen)   # only used to set the network's input normalisation
    t0 = time.time()
    model.fit_supervised(sampler, Ce_norm, steps=steps)
    if steps == [int(x) for x in a.steps.split(",")][0]:
        nparam = sum(p.numel() for p in model.net.parameters())
        print(f"  net: hidden={a.hidden} depth={a.depth} activation={a.activation} -> {nparam:,} parameters", flush=True)
    row = evaluate(model, torch.Generator(device=dev).manual_seed(99))
    results[steps] = row
    print(f"  ({time.time()-t0:.0f}s)  " + "  ".join(f"{s}: W1={row[s]['w1']:.2f}(bias{row[s]['bias_z']:+.2f},PITks={row[s]['pit_ks']:.2f})" for s in STATS), flush=True)

print(f"\n{'steps':>8s} | " + " | ".join(f"{s:>10s}" for s in STATS))
for steps, row in results.items():
    print(f"{steps:8d} | " + " | ".join(f"{row[s]['w1']:6.2f}({row[s]['bias_z']:+.2f})" for s in STATS))

suffix = ""
if a.activation != "silu":
    suffix += f"_{a.activation}"
if a.hidden != 256 or a.depth != 4:
    suffix += f"_h{a.hidden}d{a.depth}"
json.dump({"toy": toy.name, "N": toy.N, "d": toy.d, "activation": a.activation, "hidden": a.hidden, "depth": a.depth,
          "results": results}, open(f"results/toys/{toy.name}_diag_supervised{suffix}.json", "w"), indent=1, default=float)
print("saved")
