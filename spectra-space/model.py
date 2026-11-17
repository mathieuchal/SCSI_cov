"""Drift (velocity) networks for scsi.py. Two architectures, same (t, I, Ye) -> velocity interface."""
import math
import torch
import torch.nn as nn


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
    """Flat-vector MLP: in_proj -> depth x ResBlock -> out_norm -> zero-init out_proj."""

    def __init__(self, p: int, mu_y: torch.Tensor, sd_y: torch.Tensor, mu_ce: torch.Tensor, sd_ce: torch.Tensor,
                 velocity_loc: torch.Tensor, velocity_scale: torch.Tensor, hidden=256, depth=4, nfreq=8, **_):
        super().__init__()
        self.p = p
        self.register_buffer("mu_y", mu_y.float())
        self.register_buffer("sd_y", sd_y.float())
        self.register_buffer("mu_ce", mu_ce.float())
        self.register_buffer("sd_ce", sd_ce.float())
        self.register_buffer("velocity_loc", velocity_loc.float())
        self.register_buffer("velocity_scale", velocity_scale.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        din = 2 * p + 2 * nfreq + 1
        self.in_proj = nn.Linear(din, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, p)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        print(f"The model has {sum(p.numel() for p in self.parameters())} parameters")

    def _net(self, h: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(h)
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.out_norm(x))

    @torch.no_grad()
    def refresh_norm(self, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale):
        self.mu_y.copy_(mu_y.float()); self.sd_y.copy_(sd_y.float())
        self.mu_ce.copy_(mu_ce.float()); self.sd_ce.copy_(sd_ce.float())
        self.velocity_loc.copy_(velocity_loc.float()); self.velocity_scale.copy_(velocity_scale.float())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        zi = (I - self.mu_y) / self.sd_y
        ze = (Ye - self.mu_ce) / self.sd_ce
        tf = t * self.freq
        h = torch.cat([zi, ze, t, torch.sin(tf), torch.cos(tf)], dim=-1)
        return self.velocity_loc + self.velocity_scale * self._net(h)


class TransformerDrift(nn.Module):
    """Row-tokenized self-attention drift: d tokens (one per original variable), token i = [Sigma_row_i,
    Ce_row_i, time features]. A learned per-index position embedding is added to each token: without it,
    attention is exactly equivariant under permuting the tokens, which is only correct when the prior is
    itself exchangeable over the d coordinates (the "ri" toys) -- a "nonri" prior (fixed, non-symmetric C0)
    needs the network to tell variable i from variable j, which plain attention over unordered tokens cannot
    represent at all. The embedding is zero-init so it costs nothing when position truly doesn't matter."""

    def __init__(self, p: int, mu_y: torch.Tensor, sd_y: torch.Tensor, mu_ce: torch.Tensor, sd_ce: torch.Tensor,
                 velocity_loc: torch.Tensor, velocity_scale: torch.Tensor, hidden=256, depth=4, nfreq=8,
                 d: int = None, nhead: int = 8, **_):
        super().__init__()
        assert d is not None, "TransformerDrift needs d (p = d(d+1)/2)"
        self.p, self.d = p, d
        self.register_buffer("mu_y", mu_y.float())
        self.register_buffer("sd_y", sd_y.float())
        self.register_buffer("mu_ce", mu_ce.float())
        self.register_buffer("sd_ce", sd_ce.float())
        self.register_buffer("velocity_loc", velocity_loc.float())
        self.register_buffer("velocity_scale", velocity_scale.float())
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

        print(f"The model has {sum(p.numel() for p in self.parameters())} parameters")

    def _unflatten(self, y: torch.Tensor, sym: bool) -> torch.Tensor:
        B = y.shape[0]
        M = torch.zeros(B, self.d, self.d, dtype=y.dtype, device=y.device)
        M[:, self.i0, self.i1] = y
        if sym:
            M[:, self.i1, self.i0] = y
        return M

    @torch.no_grad()
    def refresh_norm(self, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale):
        self.mu_y.copy_(mu_y.float()); self.sd_y.copy_(sd_y.float())
        self.mu_ce.copy_(mu_ce.float()); self.sd_ce.copy_(sd_ce.float())
        self.velocity_loc.copy_(velocity_loc.float()); self.velocity_scale.copy_(velocity_scale.float())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        zi = (I - self.mu_y) / self.sd_y
        ze = (Ye - self.mu_ce) / self.sd_ce
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
        return self.velocity_loc + self.velocity_scale * vel


class TransformerEigDrift(nn.Module):
    """Same row-tokenized self-attention drift as TransformerDrift, plus 2 extra GLOBAL tokens appended to
    the row-token sequence: one encoding Ce's bottom eigenvector (its full d components + its eigengap),
    one encoding the top eigenvector. Row tokens and eigvec tokens then share the SAME self-attention stack,
    so each row learns, via genuine (content-adaptive, learned) attention weights, how much to draw on the
    bottom-eigvec token vs the top-eigvec token -- rather than a fixed per-row scalar spliced into every row
    unconditionally. Eigenvectors come from an explicit eigh of the raw (unnormalized) conditioning Ce (see
    StructuredFactorDrift's docstring for the rationale: an explicit, if imperfect, extraction beats forcing
    the network to rediscover the same global, basis-rotating operation implicitly). The eigvec tokens'
    own encoder outputs are discarded -- they only ever serve as extra keys/values for the row tokens."""

    def __init__(self, p: int, mu_y: torch.Tensor, sd_y: torch.Tensor, mu_ce: torch.Tensor, sd_ce: torch.Tensor,
                 velocity_loc: torch.Tensor, velocity_scale: torch.Tensor, hidden=256, depth=4, nfreq=8,
                 d: int = None, nhead: int = 8, **_):
        super().__init__()
        assert d is not None, "TransformerEigDrift needs d (p = d(d+1)/2)"
        self.p, self.d = p, d
        self.register_buffer("mu_y", mu_y.float())
        self.register_buffer("sd_y", sd_y.float())
        self.register_buffer("mu_ce", mu_ce.float())
        self.register_buffer("sd_ce", sd_ce.float())
        self.register_buffer("velocity_loc", velocity_loc.float())
        self.register_buffer("velocity_scale", velocity_scale.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        il = torch.tril_indices(d, d)
        self.register_buffer("i0", il[0])
        self.register_buffer("i1", il[1])
        self.tok_embed = nn.Linear(2 * d + 2 * nfreq + 1, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(d, hidden))
        self.eig_embed = nn.Linear(d + 1, hidden)              # full eigenvector (d) + its eigengap (1)
        self.eig_type_embed = nn.Parameter(torch.zeros(2, hidden))  # distinguishes the bot/top eigvec token
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.out_proj = nn.Linear(hidden, d)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        print(f"The model has {sum(p.numel() for p in self.parameters())} parameters")

    def _unflatten(self, y: torch.Tensor, sym: bool) -> torch.Tensor:
        B = y.shape[0]
        M = torch.zeros(B, self.d, self.d, dtype=y.dtype, device=y.device)
        M[:, self.i0, self.i1] = y
        if sym:
            M[:, self.i1, self.i0] = y
        return M

    @torch.no_grad()
    def refresh_norm(self, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale):
        self.mu_y.copy_(mu_y.float()); self.sd_y.copy_(sd_y.float())
        self.mu_ce.copy_(mu_ce.float()); self.sd_ce.copy_(sd_ce.float())
        self.velocity_loc.copy_(velocity_loc.float()); self.velocity_scale.copy_(velocity_scale.float())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        zi = (I - self.mu_y) / self.sd_y
        ze = (Ye - self.mu_ce) / self.sd_ce
        Si = self._unflatten(zi, sym=False)
        Se = self._unflatten(ze, sym=True)
        B, d, i, j = t.shape[0], self.d, self.i0, self.i1

        # -- explicit bottom/top eigenvector conditioning of the RAW (unnormalized) Ce -- no grad needed:
        # Ye has no parameter dependency, so this just avoids an unused, near-degenerate-eigenvalue-fragile
        # autograd graph through eigh (see StructuredFactorDrift's forward).
        with torch.no_grad():
            raw = torch.zeros(B, d, d, device=Ye.device, dtype=Ye.dtype)
            raw[:, i, j] = Ye.detach()
            Ce_mat = (raw + raw.transpose(1, 2) - torch.diag_embed(raw.diagonal(dim1=1, dim2=2))).double()
            evals, evecs = torch.linalg.eigh(Ce_mat)                           # ascending eigenvalues
            scale = Ce_mat.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-8)    # (B,) -- dimensionless gaps
            v_bot, v_top = evecs[..., 0].float(), evecs[..., -1].float()       # (B,d) each
            gap_bot = ((evals[..., 1] - evals[..., 0]) / scale).float()        # (B,) Davis-Kahan reliability
            gap_top = ((evals[..., -1] - evals[..., -2]) / scale).float()

        ##Time embedding
        tf = t * self.freq
        tfeat = torch.cat([t, torch.sin(tf), torch.cos(tf)], dim=-1).unsqueeze(1).expand(B, d, -1)

        ##Row tokens (same as TransformerDrift) + 2 global eigvec tokens, sharing one self-attention stack
        row_tok = self.tok_embed(torch.cat([Si, Se, tfeat], dim=-1)) + self.pos_embed
        eig_raw = torch.stack([torch.cat([v_bot, gap_bot[:, None]], dim=-1),
                                torch.cat([v_top, gap_top[:, None]], dim=-1)], dim=1)   # (B,2,d+1)
        eig_tok = self.eig_embed(eig_raw) + self.eig_type_embed                        # (B,2,hidden)
        h = self.encoder(torch.cat([row_tok, eig_tok], dim=1))[:, :d, :]               # drop the 2 eigvec tokens
        out = self.out_proj(h)
        vel = out[:, i, j]
        return self.velocity_loc + self.velocity_scale * vel


class Drift(nn.Module):
    """Pre-ResBlock flat-vector MLP: plain nn.Sequential body (Linear+SiLU stacked, no LayerNorm/residual),
    identical (t, I, Ye) -> velocity interface and normalization/time-embedding as Drift. Exists only to
    load the d=8 toy checkpoints saved before Drift's body was refactored from a flat Sequential into
    in_proj -> depth x ResBlock -> out_norm -> out_proj (their state_dict keys are net.0/2/4/6/8.*, with
    no "arch" field at all, predating that field being recorded)."""

    def __init__(self, p: int, mu_y: torch.Tensor, sd_y: torch.Tensor, mu_ce: torch.Tensor, sd_ce: torch.Tensor,
                 velocity_loc: torch.Tensor, velocity_scale: torch.Tensor, hidden=256, depth=4, nfreq=8, **_):
        super().__init__()
        self.p = p
        self.register_buffer("mu_y", mu_y.float())
        self.register_buffer("sd_y", sd_y.float())
        self.register_buffer("mu_ce", mu_ce.float())
        self.register_buffer("sd_ce", sd_ce.float())
        self.register_buffer("velocity_loc", velocity_loc.float())
        self.register_buffer("velocity_scale", velocity_scale.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        din = 2 * p + 2 * nfreq + 1
        layers = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, p)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        zi = (I - self.mu_y) / self.sd_y
        ze = (Ye - self.mu_ce) / self.sd_ce
        tf = t * self.freq
        h = torch.cat([zi, ze, t, torch.sin(tf), torch.cos(tf)], dim=-1)
        return self.velocity_loc + self.velocity_scale * self.net(h)


def make_drift(arch: str, p: int, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale,
                hidden=256, depth=4, d=None, nhead=8, N=1):
    if arch == "mlp":
        return Drift(p, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale, hidden=hidden, depth=depth)
    if arch == "transformer":
        return TransformerDrift(p, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale,
                                 hidden=hidden, depth=depth, d=d, nhead=nhead)
    if arch == "transformer_eig":
        return TransformerEigDrift(p, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale,
                                    hidden=hidden, depth=depth, d=d, nhead=nhead)
    if arch == "structured":
        return StructuredFactorDrift(p, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale,
                                      hidden=hidden, depth=depth, d=d, nhead=nhead, N=N)
    return ResNetDrift(p, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale, hidden=hidden, depth=depth)

#==================================================================================================


class EntryRowNetwork(nn.Module):
    """Column-pooling step uses per-row, content-adaptive multi-head attention (query = projection of that
    row's own mean-pooled entries) instead of a single fixed learned query shared by every row -- the latter
    can't adapt how it aggregates columns to what a particular row actually looks like, which is a likely
    bottleneck for resolving near-degenerate eigenvector directions (e.g. ldv_bot), where the right column
    weighting is row-specific and content-dependent, not a fixed global pattern."""

    def __init__(self, width=48, heads=4, layers=2, n_features=12):
        super().__init__()
        assert width % heads == 0, "width must be divisible by heads"
        self.heads, self.head_dim = heads, width // heads
        #Convert n_features into width parameter
        self.entry = nn.Sequential(nn.Linear(n_features, width), nn.SiLU(), nn.Linear(width, width))
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=2*width, dropout=0.,
                    activation="gelu", batch_first=True, norm_first=False)
        self.global_attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.decoder = nn.Sequential(nn.Linear(3*width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, features):
        entries = self.entry(features)                                  # (b,d,d,w) -- entries[i,j]
        b, d, _, w = entries.shape
        h, hd = self.heads, self.head_dim
        row_summary = entries.mean(dim=2)                                # (b,d,w) cheap row content, to derive query from
        Q = self.q_proj(row_summary).view(b, d, h, hd)                   # (b,d,h,hd) -- per-row, content-adaptive query
        K = self.k_proj(entries).view(b, d, d, h, hd)                    # (b,d,d,h,hd)
        V = self.v_proj(entries).view(b, d, d, h, hd)
        logits = torch.einsum('bihd,bijhd->bijh', Q, K) / math.sqrt(hd)  # (b,d,d,h) -- row i's query . col j's key
        weights = torch.softmax(logits, dim=2)                           # normalize over columns j, per row/head
        rows = torch.einsum('bijh,bijhd->bihd', weights, V).reshape(b, d, w)
        rows = self.global_attention(rows)
        q = w
        decoded = self.decoder(torch.cat([entries, rows[:,:,None,:].expand(b,d,d,q),
                                                   rows[:,None,:,:].expand(b,d,d,q)], -1)).squeeze(-1)
        return torch.tril(decoded)


class StructuredFactorDrift(nn.Module):
    """Entry-level (d^2 tokens, one per matrix entry) drift via EntryRowNetwork -- axial-ish attention
    (column pooling then row self-attention, row-i and row-j summaries both fed back into each entry's
    decode) plus hand-engineered per-entry features (row/col index, row-col index gap, N, diag_i, diag_j,
    diag/tril masks), PLUS an explicit eigendecomposition of the raw (unnormalized) conditioning Ce:
    the bottom/top eigenvector outer products v_i*v_j (exactly the per-entry weights that appear in
    v^T C v, the quantity ldv_top/ldv_bot score) and the bottom/top eigengaps (the Davis-Kahan factor
    governing how reliable each eigenvector estimate even is), scale-normalized by mean(diag(Ce)).
    Ce's own eigenvectors are themselves a noisy estimate of C's -- particularly in the small-eigengap
    regime that makes ldv_bot hard in the first place -- so this doesn't bypass the problem, but it
    hands the network an explicit (if imperfect) extraction instead of forcing it to rediscover the same
    global, basis-rotating operation implicitly through axial attention on raw matrix entries alone.
    Same (p, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale, hidden, depth, d, nhead)
    constructor convention as Drift/TransformerDrift; hidden/depth/nhead map onto EntryRowNetwork's
    width/layers/heads."""

    def __init__(self, p: int, mu_y: torch.Tensor, sd_y: torch.Tensor, mu_ce: torch.Tensor, sd_ce: torch.Tensor,
                 velocity_loc: torch.Tensor, velocity_scale: torch.Tensor, hidden=48, depth=2,
                 d: int = None, nhead: int = 4, N: int = 1, nfreq: int = 8, **_):
        super().__init__()
        assert d is not None, "StructuredFactorDrift needs d (p = d(d+1)/2)"
        self.p, self.d, self.N, self.nfreq = p, d, N, nfreq
        self.register_buffer("mu_y", mu_y.float())
        self.register_buffer("sd_y", sd_y.float())
        self.register_buffer("mu_ce", mu_ce.float())
        self.register_buffer("sd_ce", sd_ce.float())
        self.register_buffer("velocity_loc", velocity_loc.float())
        self.register_buffer("velocity_scale", velocity_scale.float())
        self.register_buffer("freq", (2.0 ** torch.arange(nfreq)) * math.pi)
        il = torch.tril_indices(d, d)
        self.register_buffer("i0", il[0])
        self.register_buffer("i1", il[1])
        self.net = EntryRowNetwork(hidden, nhead, depth, n_features=16 + 2 * nfreq)

        print(f"The model has {sum(p.numel() for p in self.parameters())} parameters")

    @torch.no_grad()
    def refresh_norm(self, mu_y, sd_y, mu_ce, sd_ce, velocity_loc, velocity_scale):
        self.mu_y.copy_(mu_y.float()); self.sd_y.copy_(sd_y.float())
        self.mu_ce.copy_(mu_ce.float()); self.sd_ce.copy_(sd_ce.float())
        self.velocity_loc.copy_(velocity_loc.float()); self.velocity_scale.copy_(velocity_scale.float())

    def forward(self, t: torch.Tensor, I: torch.Tensor, Ye: torch.Tensor) -> torch.Tensor:
        b, d = I.shape[0], self.d
        i, j = self.i0, self.i1
        #Normalize I and Ye
        state = torch.zeros(b,d,d,device=I.device,dtype=I.dtype)
        state[:,i,j] = (I-self.mu_y)/self.sd_y
        lower = torch.zeros_like(state)
        lower[:,i,j] = (Ye-self.mu_ce)/self.sd_ce
        cond = lower+lower.transpose(1,2)-torch.diag_embed(lower.diagonal(dim1=1,dim2=2))
        diag = cond.diagonal(dim1=1,dim2=2)
        pos = torch.arange(d,device=I.device,dtype=I.dtype)/max(d-1,1)
        ri = pos[:,None].expand(d,d); cj = pos[None,:].expand(d,d)
        def expand(x): return x.expand(b,d,d)

        # -- explicit bottom/top eigenvector conditioning of the RAW (unnormalized) Ce -- no grad needed:
        # Ye has no parameter dependency, so this just avoids building an (unused, and near-degenerate-
        # eigenvalue-fragile) autograd graph through eigh.
        with torch.no_grad():
            raw = torch.zeros_like(state)
            raw[:,i,j] = Ye.detach()
            Ce_mat = (raw+raw.transpose(1,2)-torch.diag_embed(raw.diagonal(dim1=1,dim2=2))).double()
            evals, evecs = torch.linalg.eigh(Ce_mat)                           # ascending eigenvalues
            scale = Ce_mat.diagonal(dim1=1,dim2=2).mean(-1).clamp_min(1e-8)     # (b,) -- dimensionless gaps/entries
            v_bot, v_top = evecs[...,0].float(), evecs[...,-1].float()         # (b,d) each
            gap_bot = ((evals[...,1]-evals[...,0])/scale).float()              # (b,) Davis-Kahan reliability factor
            gap_top = ((evals[...,-1]-evals[...,-2])/scale).float()
        vb_outer = v_bot[:,:,None]*v_bot[:,None,:]                              # (b,d,d) == per-entry weight in v^T C v
        vt_outer = v_top[:,:,None]*v_top[:,None,:]
        def expand_scalar(x): return x[:,None,None].expand(b,d,d)

        #Stack time
        scalar_features = torch.stack([state,cond,diag[:,:,None].expand(b,d,d),diag[:,None,:].expand(b,d,d),
            expand(t[:,:,None]), torch.full_like(state,self.N**-.5),
            expand(torch.eye(d,device=state.device)),expand(torch.tril(torch.ones(d,d,device=state.device))),
            expand(ri),expand(cj),expand(ri-cj),torch.full_like(state,math.log(d)/5),
            vb_outer, vt_outer, expand_scalar(gap_bot), expand_scalar(gap_top)],-1)
        tf = t * self.freq                                                  # (b, nfreq)
        time_embed = torch.cat([torch.sin(tf), torch.cos(tf)], dim=-1)      # (b, 2*nfreq)
        time_embed = time_embed[:, None, None, :].expand(b, d, d, 2 * self.nfreq)
        features = torch.cat([scalar_features, time_embed], dim=-1)
        result = self.net(features)
        return self.velocity_loc+self.velocity_scale*result[:,i,j]
