"""Train B0 / B1 / B2 on Track A for a fixed wall-clock budget on MPS.

Loss: (1 - SSIM) + 10*MSE on the atom heatmap; class-weighted CE on element map; L1 height on
atom pixels. For B1/B2 the loss is computed on the centre CENTRE x CENTRE px of each PATCH window
(patch -> centre, as in PtyRANNOSAUR). Keeps the checkpoint with the lowest validation loss.

Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/train.py B0|B1|B2 <minutes>
"""
import json
import math
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from afmport.common import RESULTS
from afmport.data import TB, TrackA
from afmport.models import CENTRE, PATCH, make_model, norm_cond


def gauss_win(dev, size=7, sigma=1.5):
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] * (g / g.sum())[None]
    return g[None, None].to(dev)


def ssim(a, b, win, C1=0.01**2, C2=0.03**2):
    mu_a, mu_b = F.conv2d(a, win, padding=3), F.conv2d(b, win, padding=3)
    saa = F.conv2d(a * a, win, padding=3) - mu_a**2
    sbb = F.conv2d(b * b, win, padding=3) - mu_b**2
    sab = F.conv2d(a * b, win, padding=3) - mu_a * mu_b
    return (((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (saa + sbb + C2))).mean()


def main():
    kind, minutes = sys.argv[1], float(sys.argv[2])
    track = sys.argv[3] if len(sys.argv) > 3 else "a"
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    out_dir = RESULTS / "pilot" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "whole" if kind == "B0" else "patch"
    bs = 16 if mode == "whole" else 128
    if track == "b":  # same patch -> centre recipe (B1 architecture) on AFM-ice sims
        train = TrackA("train", mode, n_items=10**7, root=TB, min_slices=7)
        val = TrackA("val", mode, augment=False, noise=(0.05, 0.05), n_items=2048, seed=7, root=TB)
    else:
        train = TrackA("train", mode, n_items=10**7)
        val = TrackA("val", mode, augment=False, noise=(0.05, 0.05), n_items=350 if mode == "whole" else 2048, seed=7)
    tag = kind if track == "a" else f"{kind}_ice"
    tl = DataLoader(train, batch_size=bs, num_workers=2, persistent_workers=True, prefetch_factor=2)
    vl = DataLoader(val, batch_size=bs, num_workers=2)

    model = make_model(kind).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_w = torch.tensor([0.05, 1.0, 1.0, 2.0, 2.0, 3.0], device=dev)
    win = gauss_win(dev)

    def loss_fn(batch):
        x, heat, cls, hgt, ak = [t.to(dev) for t in batch]
        cond = norm_cond(ak[:, 0], ak[:, 1]) if kind == "B2" else None
        out = model(x, cond)
        if mode == "patch":
            m = (PATCH - CENTRE) // 2
            crop = lambda t: t[..., m:m + CENTRE, m:m + CENTRE]
            out, heat, cls, hgt = crop(out), crop(heat), crop(cls), crop(hgt)
        p = torch.sigmoid(out[:, :1])
        l_heat = (1 - ssim(p, heat, win)) + 10 * F.mse_loss(p, heat)
        l_cls = F.cross_entropy(out[:, 1:7], cls, weight=class_w)
        mask = (cls > 0).float()[:, None]
        l_h = (torch.abs(out[:, 7:8] - hgt) * mask).sum() / mask.sum().clamp(min=1)
        return l_heat + 0.5 * l_cls + 0.5 * l_h, (l_heat.item(), l_cls.item(), l_h.item())

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for b in vl:
            l, _ = loss_fn(b)
            tot += l.item() * len(b[0])
            n += len(b[0])
        model.train()
        return tot / n

    t0, step, best, log = time.time(), 0, math.inf, []
    budget = minutes * 60
    eval_every = 200
    model.train()
    for batch in tl:
        el = time.time() - t0
        if el > budget:
            break
        for g in opt.param_groups:
            g["lr"] = 1e-3 if el < 0.7 * budget else 3e-4
        loss, parts = loss_fn(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % eval_every == 0 or step == 20:
            v = validate()
            rec = {"step": step, "min": round(el / 60, 1), "train": round(loss.item(), 4),
                   "heat/cls/h": [round(p, 4) for p in parts], "val": round(v, 4)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if v < best:
                best = v
                torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    v = validate()
    if v < best:
        best = v
        torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    (out_dir / f"{tag}_log.json").write_text(json.dumps({"steps": step, "minutes": minutes, "best_val": best,
                                                          "log": log}, indent=1))
    print(f"{kind}: {step} steps in {(time.time() - t0) / 60:.1f} min, best val {best:.4f}")


if __name__ == "__main__":
    main()
