"""Inference: whole-image (B0) and sliding-window patch -> centre with overlap averaging (B1/B2)."""
import numpy as np
import torch
import torch.nn.functional as F

from .models import CENTRE, PATCH, norm_cond


def _pad_to(x, mult):
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    return F.pad(x, (0, pw, 0, ph)), (h, w)


@torch.no_grad()
def predict_whole(model, x, device, cond=None):
    """x: (B, Z, H, W) tensor. Returns (B, 8, H, W) with heat as probability."""
    model.eval()
    xp, (h, w) = _pad_to(x.to(device), 2 ** model.depth)
    out = model(xp, None if cond is None else cond.to(device))[..., :h, :w]
    out[:, 0] = torch.sigmoid(out[:, 0])
    return out.cpu()


@torch.no_grad()
def predict_patches(model, x, device, cond=None, stride=8, batch=256):
    """Slide PATCH windows with `stride`; keep each window's centre CENTRE px; average overlaps."""
    model.eval()
    B, Z, H, W = x.shape
    m = (PATCH - CENTRE) // 2
    # Pad so every image pixel is covered by window centres; zero = per-slice mean after z-scoring.
    Hp = int(np.ceil(H / stride)) * stride + 2 * m + (CENTRE - stride)
    Wp = int(np.ceil(W / stride)) * stride + 2 * m + (CENTRE - stride)
    xp = F.pad(x, (m, Wp - W - m, m, Hp - H - m))
    acc = torch.zeros(B, 8, Hp, Wp)
    cnt = torch.zeros(1, 1, Hp, Wp)
    ys = list(range(0, Hp - PATCH + 1, stride))
    xs = list(range(0, Wp - PATCH + 1, stride))
    coords = [(y, x_) for y in ys for x_ in xs]
    for b in range(B):
        c_b = None if cond is None else cond[b:b + 1]
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            tiles = torch.stack([xp[b, :, y:y + PATCH, x_:x_ + PATCH] for y, x_ in chunk]).to(device)
            cc = None if c_b is None else c_b.expand(len(chunk), -1).to(device)
            out = model(tiles, cc)[:, :, m:m + CENTRE, m:m + CENTRE].cpu()
            for (y, x_), o in zip(chunk, out):
                acc[b, :, y + m:y + m + CENTRE, x_ + m:x_ + m + CENTRE] += o
                if b == 0:
                    cnt[0, 0, y + m:y + m + CENTRE, x_ + m:x_ + m + CENTRE] += 1
    out = (acc / cnt.clamp(min=1))[:, :, m:m + H, m:m + W]
    out[:, 0] = torch.sigmoid(out[:, 0])
    return out


def predict(kind, model, x, device, amp=None, stiffness=None):
    if kind == "B0":
        return predict_whole(model, x, device)
    cond = None
    if kind == "B2":
        cond = norm_cond(torch.as_tensor(amp, dtype=torch.float32), torch.as_tensor(stiffness, dtype=torch.float32))
        cond = cond.reshape(-1, 2)
    return predict_patches(model, x, device, cond)
