import torch

def orientation_mcmc(Lambda: torch.Tensor, Lambda_e: torch.Tensor, N: int,
                      n_steps: int = 3000, step_size: float = 0.3,
                      burn_in: int = 1000, thin: int = 10, seed: int = None):
    '''Metropolis-Hastings sampler for the relative orientation R in O(p),
    targeting p(R) propto exp(-N/2 * Tr(Lambda^-1 R^T Lambda_e R)) dH(R).

    Proposition kernel is based on Givens rotation.

    Lambda, Lambda_e: (B, p) -- Lambda from the trained model's own E-step
    estimate (calling the model on Lambda_e), Lambda_e the empirical
    eigenvalues of the real Ce. One independent chain per batch row.

    Returns R samples, shape (n_kept, B, p, p), plus the acceptance rate.
    '''
    if seed is not None:
        torch.manual_seed(seed)
    device = Lambda.device
    B, p = Lambda.shape
    inv_Lambda = 1.0 / Lambda

    def energy(R):
        # Tr(Lambda^-1 R^T Lambda_e R) = sum_i Lambda_e[i] * sum_k R[i,k]^2 * inv_Lambda[k]
        W = R ** 2
        Winv = torch.einsum('bik,bk->bi', W, inv_Lambda)
        return 0.5 * N * torch.einsum('bi,bi->b', Lambda_e, Winv)

    R = torch.eye(p, device=device).unsqueeze(0).repeat(B, 1, 1)
    E = energy(R)

    samples = []
    n_accept = 0
    for step in range(n_steps):
        i, j = torch.randperm(p)[:2].tolist()
        theta = torch.randn(B, device=device) * step_size
        c, s = torch.cos(theta), torch.sin(theta)
        G = torch.eye(p, device=device).unsqueeze(0).repeat(B, 1, 1)
        G[:, i, i] = c; G[:, i, j] = -s
        G[:, j, i] = s; G[:, j, j] = c
        R_prop = torch.matmul(G, R)
        E_prop = energy(R_prop)
        dE = E_prop - E

        log_accept_prob = torch.clamp(-dE, max=0.0)   # min(0, -dE), numerically stable
        accept = torch.log(torch.rand(B, device=device)) < log_accept_prob
        R = torch.where(accept.view(B, 1, 1), R_prop, R)
        E = torch.where(accept, E_prop, E)
        n_accept += accept.float().mean().item()

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(R.clone())

    acc_rate = n_accept / n_steps
    return torch.stack(samples), acc_rate