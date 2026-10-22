"""Shared evaluation helpers: load models, predict stacks, per-sample metrics."""
import json
import os

import numpy as np
import torch

from .common import RESULTS
from .infer import predict
from .metrics import evaluate_sample
from .models import make_model
from .sim import zscore_slices

DEV = torch.device(os.environ.get("AFM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
MODELS = RESULTS / "pilot" / "models"


def load_model(kind):
    m = make_model(kind)
    m.load_state_dict(torch.load(MODELS / f"{kind}.pt", map_location="cpu", weights_only=True))
    return m.to(DEV).eval()


def prep(stack, noise, rng):
    """stack (H, W, Z) raw -> z-scored tensor (Z, H, W) with relative Gaussian noise."""
    s = np.asarray(stack, np.float32)
    if noise > 0:
        s = s + rng.normal(0, 1, s.shape).astype(np.float32) * noise * s.std(axis=(0, 1), keepdims=True)
    return torch.from_numpy(zscore_slices(s)).permute(2, 0, 1)


def predict_many(kind, model, stacks, amps, ks, noise=0.05, seed=0, batch=8):
    """Returns list of (8, H, W) numpy predictions."""
    rng = np.random.default_rng(seed)
    outs = []
    for i in range(0, len(stacks), batch):
        x = torch.stack([prep(stacks[j], noise, rng) for j in range(i, min(i + batch, len(stacks)))])
        o = predict(kind, model, x, DEV, amps[i:i + len(x)], ks[i:i + len(x)])
        outs.extend(o.numpy())
    return outs


def score(outs, atoms_xy, atoms_cls, threshold, tol=0.5):
    return [evaluate_sample(o, xy, c, threshold, tol) for o, xy, c in zip(outs, atoms_xy, atoms_cls)]


def tune_threshold(outs, atoms_xy, atoms_cls, grid=np.arange(0.1, 0.86, 0.05)):
    best = max(grid, key=lambda t: np.mean([r["f1"] for r in score(outs, atoms_xy, atoms_cls, t)]))
    return float(best)


def thresholds_path():
    return MODELS / "thresholds.json"


def get_threshold(kind):
    return json.loads(thresholds_path().read_text())[kind]


def split_atoms(meta):
    o = meta["offsets"]
    xy = [meta["atoms_xyz"][o[i]:o[i + 1], :2] for i in range(len(o) - 1)]
    cl = [meta["atoms_cls"][o[i]:o[i + 1]].astype(int) for i in range(len(o) - 1)]
    return xy, cl
