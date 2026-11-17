"""SC-SI runs on disk: the checkpoints of a run and how to load one. Shared by the log-score and W1 scripts."""
import os, re
import torch
import paths
from scsi import SCSI, SCSIConfig, Sym
from model import make_drift



def checkpoints(R, stem, ks):
    """checkpoints {R}/{stem}_k{k}.pt on disk among ks: 'grid' (paths.GRID), 'all', 'best' (none: only best_k, which the
    callers add), or a comma list"""
    on_disk = {int(m.group(1)) for f in os.listdir(R) if (m := re.fullmatch(rf"{re.escape(stem)}_k(\d+)\.pt", f))}
    if ks == "best":
        return []
    wanted = set(paths.GRID) if ks == "grid" else on_disk if ks == "all" else {int(k) for k in ks.split(",")}
    return sorted(wanted & on_disk)


def select_toys(toys, data):
    """the comma list toys, or (None) the toys of the data regime with a production run on disk"""
    if toys:
        return toys.split(",")
    R = paths.res_dir(data)
    return [t for t, g in paths.tags(data).items() if os.path.exists(f"{R}/{t}_{g}_train.json")]


def load_scsi(toy, path, threads=2):
    """SC-SI model of a checkpoint (EMA drift net; hidden/depth/arch from the checkpoint) -> (model, k)"""
    z = torch.load(path, map_location="cpu")
    arch, nhead = z.get("arch", "mlp"), z.get("nhead", 8)
    cfg = SCSIConfig(N=toy.N, d=toy.d, kappa=1.0, threads=threads, hidden=z.get("hidden", 256), depth=z.get("depth", 4),
                     arch=arch, nhead=nhead)
    m = SCSI(cfg)
    m.ema_net = make_drift(arch, Sym(toy.d).p, z["mu"], z["sd"], hidden=cfg.hidden, depth=cfg.depth, d=toy.d, nhead=nhead,
                           N=toy.N).to(m.device)
    m.ema_net.load_state_dict(z["state"])
    m.ema_net.eval()
    return m, z["k"]
