import time, torch, numpy as np
from toys import make_toy
from scsi import wishart_channel, DT
torch.set_num_threads(2)
toy = make_toy("factor_d8")
g = torch.Generator().manual_seed(5)
C = toy.sample_prior(16, g); Ce = wishart_channel(C, toy.N, g)
print("prior draw: eigenvalues of first task:", np.round(torch.linalg.eigvalsh(C[0]).flip(0).numpy(), 2), " cond numbers:", np.round((torch.linalg.eigvalsh(C)[:, -1] / torch.linalg.eigvalsh(C)[:, 0]).numpy()[:6], 1))
t0 = time.time()
Cs, diag = toy.hmc(Ce, n_chains=8, n_keep=128, thin=4, warmup=1000, gen=g, return_diag=True)
print(f"HMC time {time.time()-t0:.0f}s  diag {diag}")
nc, nk = 8, 128
def rhat(x):                      # x: (M, nc, nk)
    m = x.mean(-1); W = x.var(-1, unbiased=True).mean(-1); B = nk * m.var(-1, unbiased=True)
    return torch.sqrt(((nk - 1) / nk * W + B / nk) / W)
def ess_prox(x):                  # crude: 1 / (1 + 2 * sum of positive lag autocorr) using lag-1..5 on chain-averaged
    x = x - x.mean(-1, keepdim=True); ac = [(x[..., :-l] * x[..., l:]).mean(-1) / x.var(-1) for l in range(1, 6)]
    tau = 1 + 2 * torch.stack(ac).clamp_min(0).sum(0).mean(-1); return nk * nc / tau
def stat(Cs):
    ld = torch.linalg.slogdet(Cs)[1]; w = torch.linalg.eigvalsh(Cs)
    return {"logdet": ld, "logcond": torch.log(w[..., -1] / w[..., 0]), "top_share": w[..., -1] / w.sum(-1)}
S = stat(Cs)
for k, v in S.items():
    v = v.reshape(16, nc, nk)
    print(f"  {k:10s} R-hat max={float(rhat(v).max()):.3f}  median={float(rhat(v).median()):.3f}   ESS(median)~{float(ess_prox(v).median()):.0f}")
