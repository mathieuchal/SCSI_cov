"""OAS / LW-NLS with a tuned sample size n: on each fold, n is chosen on the fold's in-fold validation pairs (minimum
mean log realised min-variance-portfolio variance over the grid NS), then the plug-in is applied with that n to the
test cases of the run. SC-SI is untouched (its stored per-case realised variances are reused). Used by fin_paper_fig.py
(panel a: OAS with tuned n is the reference).

The data set-up (folds, test cases, validation pairs, RNG consumption) is taken from fin_exp.py itself: the part of
fin_exp.py before its main loop is executed with the run's flags, so the test cases are exactly those of the run
(checked against the stored sample-covariance variances).

    python fin_shrink_n_tune.py                    # the paper run -> results/{RUN}_shrinkn_tuned.npz
    FIN_FLAGS="..." FIN_RUN=results/... python fin_shrink_n_tune.py     # another run (flags = its fin_exp.py flags)
"""
import os, sys, json, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PAPER_FLAGS = "--hidden 1024 --depth 4 --neff_scale 0.3 --nfe_min 3 --n_outer 30 --tag _mlp_w1024_d4_neff03_nfe6_o30 --select tscore"
PAPER_RUN = "results/fin_d12_nw63_s0_mlp_w1024_d4_neff03_nfe6_o30_selT"
sys.argv = ["fin_exp.py"] + os.environ.get("FIN_FLAGS", PAPER_FLAGS).split()
src = open("fin_exp.py").read()
g = {"__name__": "tune", "__file__": os.path.join(HERE, "fin_exp.py")}
exec(compile(src[:src.index("rng = np.random.default_rng(args.seed)")], "fin_exp.py", "exec"), g)
for k in ("args", "first_idx", "T", "NW", "train_tasks", "test_paths", "make_cases", "fold_val_pairs", "mv_weights", "oas_shrink", "nls_shrink"):
    globals()[k] = g[k]
from covutils import bootstrap_ci

RUN = os.environ.get("FIN_RUN", PAPER_RUN)
res = json.load(open(RUN + "_results.json")); z = np.load(RUN + "_arrays.npz")
names = list(res["mv"]); V = {n: z[f"var_{i}"] for i, n in enumerate(names)}
NS = [14, 19, 25, 30, 38, 50, 63, 80, 100, 126, 200, 400]
fold_years = [int(y) for y in args.folds.split(",")]
bounds = [(fold_years[i], fold_years[i + 1] if i + 1 < len(fold_years) else 2027) for i in range(len(fold_years))]
mvvar = lambda C, F: torch.einsum("mi,mij,mj->m", mv_weights(C), F, mv_weights(C)).cpu().numpy()
test = {m: {n: [] for n in NS} for m in ("OAS", "LW")}; tuned = {m: [] for m in ("OAS", "LW")}; scm_chk = []; choice = {}
for fi, (y0, y1) in enumerate(bounds):
    tr_end, te_end = first_idx(y0), min(first_idx(y1), T)
    frng = np.random.default_rng(args.seed + 10 * fi)
    val_start = max(tr_end - int(round(args.val_years * 252)), NW)
    train_tasks(val_start, frng)                                   # consume the RNG exactly as fin_exp does
    baskets, ends = test_paths(tr_end, te_end, frng)
    Cal, Fwd, meta = make_cases(baskets, ends)
    Cv, Fv, _ = fold_val_pairs(val_start, tr_end, np.random.default_rng(args.seed + 10 * fi + 5))[:3]
    scm_chk.append(mvvar(Cal, Fwd))
    for m, f in (("OAS", oas_shrink), ("LW", nls_shrink)):
        val = {n: float(np.mean(np.log(mvvar(f(Cv, n), Fv)))) for n in NS}
        nb = min(val, key=val.get); choice[(m, fi)] = (nb, val)
        for n in NS:
            test[m][n].append(mvvar(f(Cal, n), Fwd))
        tuned[m].append(test[m][nb][-1])
        print(f"fold {fi} {m}: val mean log var by n: " + " ".join(f"{n}:{val[n]:.4f}" for n in NS) + f"  -> n*={nb}", flush=True)
scm = np.concatenate(scm_chk); assert np.allclose(scm, V["SCM"]), "test cases do not match the stored run"
print("test cases reproduce the stored run:", len(scm))
mid = z["meta"][:, 1]
for m in ("OAS", "LW"):
    ref = np.concatenate(tuned[m])
    print(f"\n== reference {m} with n tuned on validation (per fold: {[choice[(m, f)][0] for f in range(3)]}), vol {np.sqrt(252 * ref.mean()):.2f}")
    for lab in ("SC-SI Stein-opt.", "SC-SI post. mean", "OAS shrinkage", "IW post. mean"):
        a, lo, hi = bootstrap_ci(np.log(V[lab] / ref), mid); print(f"   {lab:18s} log(var/ref) {a:+.4f} [{lo:+.4f},{hi:+.4f}]")

out = RUN + "_shrinkn_tuned.npz"
np.savez(out, oas_tuned=np.concatenate(tuned["OAS"]), lw_tuned=np.concatenate(tuned["LW"]),
         n_oas=np.array([choice[("OAS", f)][0] for f in range(3)]), n_lw=np.array([choice[("LW", f)][0] for f in range(3)]), grid=np.array(NS),
         oas_n63=np.concatenate(test["OAS"][63]), lw_n63=np.concatenate(test["LW"][63]),
         oas_n19=np.concatenate(test["OAS"][19]), lw_n19=np.concatenate(test["LW"][19]))
print("saved", out)
