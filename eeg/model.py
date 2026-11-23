"""Drift (velocity) networks for scsi.py. Several architectures, same (t, I, Ye) -> velocity interface,
same (p, mu, sd, hidden, depth, ...) constructor convention (single normalization pair, shared between
I and Ye, exactly as the Drift already used by scsi.py)."""
import math
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# drift network
# --------------------------------------------------------------------------- #
ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class Drift(nn.Module):
    def __init__(self, p: int, mu: torch.Tensor, sd: torch.Tensor, hidden=256, depth=4, nfreq=8, activation="silu"):
        super().__init__()
        self.p = p
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        din = 2 * p + 2 * nfreq + 1
        act = ACTIVATIONS[activation]                    # parameter-free, so checkpoints load whatever the activation
        layers = [nn.Linear(din, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, p)]
        self.net = nn.Sequential(*layers)
        # zero-init last layer: start from b = 0 (scaled)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.out_scale = float(sd.mean())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        # t: (B,1)
        #Normalize input data. Same coefficients are applied for I and Ye
        zi = (I - self.mu) / self.sd
        ze = (Ye - self.mu) / self.sd
        tf = t * self.freq
        h = torch.cat([zi, ze, t, torch.sin(tf), torch.cos(tf)], dim=-1)
        return t * (I - Ye) + self.net(h) * self.out_scale  # exact limit b_1(y,Ye) = y - Ye


class ResBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))

class ResNetDrift(nn.Module):
    """Same (t, I, Ye) -> velocity interface and (p, mu, sd, hidden, depth, nfreq) constructor convention
    as Drift, but the flat MLP body is replaced by in_proj -> depth x ResBlock -> out_norm -> zero-init
    out_proj: residual connections should keep gradients healthy as hidden/depth grow, where a plain deep
    Sequential (as in Drift) starts to struggle."""

    def __init__(self, p: int, mu: torch.Tensor, sd: torch.Tensor, hidden=256, depth=4, nfreq=8, **_):
        super().__init__()
        self.p = p
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        din = 2 * p + 2 * nfreq + 1
        self.in_proj = nn.Linear(din, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, p)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.out_scale = float(sd.mean())

    def _net(self, h: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(h)
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.out_norm(x))

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        zi = (I - self.mu) / self.sd
        ze = (Ye - self.mu) / self.sd
        tf = t * self.freq
        h = torch.cat([zi, ze, t, torch.sin(tf), torch.cos(tf)], dim=-1)
        return t * (I - Ye) + self._net(h) * self.out_scale  # exact limit b_1(y,Ye) = y - Ye


class TransformerDrift(nn.Module):
    """Row-tokenized self-attention drift: d tokens (one per original variable), token i = [Sigma_row_i,
    Ce_row_i, time features]. A learned per-index position embedding is added to each token: without it,
    attention is exactly equivariant under permuting the tokens, which is only correct when the prior is
    itself exchangeable over the d coordinates (the "ri" toys) -- a "nonri" prior (fixed, non-symmetric C0)
    needs the network to tell variable i from variable j, which plain attention over unordered tokens cannot
    represent at all. The embedding is zero-init so it costs nothing when position truly doesn't matter."""

    def __init__(self, p: int, mu: torch.Tensor, sd: torch.Tensor, hidden=256, depth=4, nfreq=8,
                 d: int = None, nhead: int = 8, **_):
        super().__init__()
        assert d is not None, "TransformerDrift needs d (p = d(d+1)/2)"
        self.p, self.d = p, d
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        il = torch.tril_indices(d, d)
        self.register_buffer("i0", il[0])
        self.register_buffer("i1", il[1])
        self.tok_embed = nn.Linear(2 * d + 2 * nfreq + 1, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(d, hidden))
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.out_proj = nn.Linear(hidden, d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.out_scale = float(sd.mean())

    def _unflatten(self, y: torch.Tensor, sym: bool) -> torch.Tensor:
        B = y.shape[0]
        M = torch.zeros(B, self.d, self.d, dtype=y.dtype, device=y.device)
        M[:, self.i0, self.i1] = y
        if sym:
            M[:, self.i1, self.i0] = y
        return M

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        zi = (I - self.mu) / self.sd
        ze = (Ye - self.mu) / self.sd
        Si = self._unflatten(zi, sym=False)
        Se = self._unflatten(ze, sym=True)
        B = t.shape[0]

        ##Time embedding
        tf = t * self.freq
        tfeat = torch.cat([t, torch.sin(tf), torch.cos(tf)], dim=-1).unsqueeze(1).expand(B, self.d, -1)

        ##Concatenate + Add positional embedding
        tokens = torch.cat([Si, Se, tfeat], dim=-1)
        h = self.encoder(self.tok_embed(tokens) + self.pos_embed)
        out = self.out_proj(h)
        vel = out[:, self.i0, self.i1]
        return t * (I - Ye) + self.out_scale * vel  # same exact-limit convention as Drift


def make_drift(arch: str, p: int, mu, sd, hidden=256, depth=4, d=None, nhead=8, N=1, activation="silu"):
    """activation: hidden-layer nonlinearity of the plain MLP Drift only (the other architectures keep theirs)."""
    if arch == "transformer":
        return TransformerDrift(p, mu, sd, hidden=hidden, depth=depth, d=d, nhead=nhead)
    if arch == "structured":
        return StructuredFactorDrift(p, mu, sd, hidden=hidden, depth=depth, d=d, nhead=nhead, N=N)
    if arch == "resnet":
        return ResNetDrift(p, mu, sd, hidden=hidden, depth=depth)
    return Drift(p, mu, sd, hidden=hidden, depth=depth, activation=activation)

#==================================================================================================


class EntryRowNetwork(nn.Module):
    def __init__(self, width=48, heads=4, layers=2, n_features=12):
        super().__init__()
        self.entry = nn.Sequential(nn.Linear(n_features, width), nn.SiLU(), nn.Linear(width, width))
        self.query = nn.Parameter(torch.randn(width)/math.sqrt(width))
        layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0.,
                    activation="gelu", batch_first=True, norm_first=False)
        self.global_attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.decoder = nn.Sequential(nn.Linear(3*width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, features):
        entries = self.entry(features)
        weights = torch.softmax((entries*self.query).sum(-1)/math.sqrt(entries.shape[-1]), dim=2)
        rows = (weights[..., None]*entries).sum(2)
        rows = self.global_attention(rows)
        b,d,_,q = entries.shape
        decoded = self.decoder(torch.cat([entries, rows[:,:,None,:].expand(b,d,d,q),
                                                   rows[:,None,:,:].expand(b,d,d,q)], -1)).squeeze(-1)
        return torch.tril(decoded)


class StructuredFactorDrift(nn.Module):
    """Entry-level (d^2 tokens, one per matrix entry) drift via EntryRowNetwork -- axial-ish attention
    (column pooling then row self-attention, row-i and row-j summaries both fed back into each entry's
    decode) plus hand-engineered per-entry features (row/col index, row-col index gap, N, diag_i, diag_j,
    diag/tril masks). Same (p, mu, sd, hidden, depth, d, nhead) constructor convention as Drift/
    TransformerDrift; hidden/depth/nhead map onto EntryRowNetwork's width/layers/heads."""

    def __init__(self, p: int, mu: torch.Tensor, sd: torch.Tensor, hidden=48, depth=2,
                 d: int = None, nhead: int = 4, N: int = 1, nfreq: int = 8, **_):
        super().__init__()
        assert d is not None, "StructuredFactorDrift needs d (p = d(d+1)/2)"
        self.p, self.d, self.N, self.nfreq = p, d, N, nfreq
        self.register_buffer("mu", mu.float())
        self.register_buffer("sd", sd.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        il = torch.tril_indices(d, d)
        self.register_buffer("i0", il[0])
        self.register_buffer("i1", il[1])
        self.net = EntryRowNetwork(hidden, nhead, depth, n_features=12 + 2 * nfreq)  # EntryRowNetwork already zero-inits its own decoder
        self.out_scale = float(sd.mean())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        b, d = I.shape[0], self.d
        i, j = self.i0, self.i1
        #Normalize I and Ye
        state = torch.zeros(b,d,d,device=I.device,dtype=I.dtype)
        state[:,i,j] = (I-self.mu)/self.sd
        lower = torch.zeros_like(state)
        lower[:,i,j] = (Ye-self.mu)/self.sd
        cond = lower+lower.transpose(1,2)-torch.diag_embed(lower.diagonal(dim1=1,dim2=2))
        diag = cond.diagonal(dim1=1,dim2=2)
        pos = torch.arange(d,device=I.device,dtype=I.dtype)/max(d-1,1)
        ri = pos[:,None].expand(d,d); cj = pos[None,:].expand(d,d)
        def expand(x): return x.expand(b,d,d)
        #Stack time
        scalar_features = torch.stack([state,cond,diag[:,:,None].expand(b,d,d),diag[:,None,:].expand(b,d,d),
            expand(t[:,:,None]), torch.full_like(state,self.N**-.5),
            expand(torch.eye(d,device=state.device)),expand(torch.tril(torch.ones(d,d,device=state.device))),
            expand(ri),expand(cj),expand(ri-cj),torch.full_like(state,math.log(d)/5)],-1)
        tf = t * self.freq                                                  # (b, nfreq)
        time_embed = torch.cat([torch.sin(tf), torch.cos(tf)], dim=-1)      # (b, 2*nfreq)
        time_embed = time_embed[:, None, None, :].expand(b, d, d, 2 * self.nfreq)
        features = torch.cat([scalar_features, time_embed], dim=-1)
        result = self.net(features)
        return t * (I - Ye) + self.out_scale * result[:, i, j]  # same exact-limit convention as Drift
