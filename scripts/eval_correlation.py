"""Does the model's output track the truth? Correlation view (added 2026-10-16, after seeing results).

Per model (B0, B1, B2) and test set:
  - heat r: per-sample Pearson r between predicted atom map and true atom map
  - chance r: same, but predicted map paired with a different sample's true map (shuffled control)
  - height r: Pearson r between predicted and true atom height over matched atoms (pooled)
Sets: T1 (held-out molecules), T2 (composites), FDBM (QUAM's own images, K100_Amp100 folder).
Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/eval_correlation.py
"""
import json

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from afmport.common import DERIVED, QUAM_CELL, QUAM_DIR, RESULTS, quam_xyz_path, read_xyz, z_to_class
from afmport.data import TA
from afmport.evaluation import get_threshold, load_model, predict_many, split_atoms
from afmport.metrics import detect, match
from afmport.targets import render


def pearson(a, b):
    a, b = np.ravel(a) - np.mean(a), np.ravel(b) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def quam_stack(cid, tag="K100_Amp100", n_px=120):
    folder = next(QUAM_DIR.glob(f"K-*/Conformer3D_CID_{cid}_{tag}"))
    q = np.stack([np.asarray(Image.open(f), np.float32) for f in sorted(folder.glob("*_df_*.jpg"))], -1)
    ours = np.transpose(np.rot90(q[..., ::-1], -1, axes=(0, 1)), (1, 0, 2))
    g = np.arange(n_px) * 0.2 * 255 / QUAM_CELL
    yy, xx = np.meshgrid(g, g, indexing="ij")
    return np.stack([map_coordinates(ours[:, :, z], [yy, xx], order=1) for z in range(ours.shape[2])], -1)


sets = {}
for name, xf, mf in (("T1", "test_X.npy", "test_meta.npz"), ("T2", "t2_X.npy", "t2_meta.npz")):
    meta = np.load(TA / mf)
    X = np.load(TA / xf, mmap_mode="r")
    o = meta["offsets"]
    atoms = [meta["atoms_xyz"][o[i]:o[i + 1]] for i in range(len(o) - 1)]
    sets[name] = (X, meta["params"][:, 0], meta["params"][:, 1], atoms)
test = json.loads((DERIVED / "scaffold_split.json").read_text())["test"]
fx, fa = [], []
for cid in test:
    _, Zs, xyz, _ = read_xyz(quam_xyz_path(cid))
    p = xyz.copy()
    p[:, 2] -= xyz[:, 2].mean()
    fx.append(quam_stack(cid))
    fa.append(p.astype(np.float32))
sets["FDBM"] = (fx, np.full(len(fx), 1.0), np.full(len(fx), 1.0), fa)

res = {}
rng = np.random.default_rng(0)
for kind in ("B0", "B1", "B2"):
    model, thr = load_model(kind), get_threshold(kind)
    res[kind] = {}
    for name, (X, amps, ks, atoms) in sets.items():
        outs = predict_many(kind, model, X, amps, ks, seed=900)
        n_px = outs[0].shape[-1]
        truth = [render(a, np.ones(len(a), int), (0.0, 0.0), n_px)[0] for a in atoms]
        r = [pearson(o[0], t) for o, t in zip(outs, truth)]
        perm = rng.permutation(len(outs))
        perm = np.where(perm == np.arange(len(outs)), (perm + 1) % len(outs), perm)
        r_chance = [pearson(outs[i][0], truth[j]) for i, j in enumerate(perm)]
        hp, ht = [], []
        for o, a in zip(outs, atoms):
            xy, pk = detect(o[0], thr)
            for i, j in match(xy, a[:, :2], 0.5):
                hp.append(o[7, pk[i, 0], pk[i, 1]])
                ht.append(a[j, 2])
        res[kind][name] = {
            "n": len(outs), "heat_r_mean": float(np.nanmean(r)), "heat_r_median": float(np.nanmedian(r)),
            "frac_samples_r_pos": float(np.mean(np.array(r) > 0)), "chance_r_mean": float(np.nanmean(r_chance)),
            "height_r": pearson(hp, ht), "n_height_atoms": len(hp),
        }
        print(kind, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[kind][name].items()}, flush=True)
(RESULTS / "pilot" / "correlation.json").write_text(json.dumps(res, indent=1))
