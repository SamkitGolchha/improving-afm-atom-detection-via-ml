"""S3: recovery of injected linear scan distortions on T2 composite scenes (B2 model, stitched).

Locked protocol: distortion D = [[1+sx, sh], [0, 1+sy]], sx, sy, sh ~ U(-5%, 5%), one per scene.
A scan pixel at nominal position u probes true position D u, so atoms appear at D^-1 p.
Correction C (upper triangular, 3 params) is fitted so nearest-neighbour bond lengths of the
detected atoms match per-pair-type reference lengths (bond-length prior). Reference lengths are
calibrated on UNDISTORTED predictions of the Track A val split (not the test scenes).
Metric: RMS over the scene's true atoms of |(S - I)(p - centroid)|, where M = C D^-1 = R S (polar
decomposition: rotation and translation removed). Pass if median <= 0.125 A.

Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/eval_s3_distortion.py
"""
import json

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from afmport.common import RESULTS
from afmport.data import TA
from afmport.evaluation import get_threshold, load_model, predict_many, split_atoms
from afmport.metrics import detect

KIND = "B2"
rng = np.random.default_rng(31)
model, thr = load_model(KIND), get_threshold(KIND)


def nn_pairs(xy, cls, rmax=2.0):
    """Unique nearest-neighbour pairs; type 0 = H-heavy, 1 = heavy-heavy (H-H skipped)."""
    if len(xy) < 2:
        return np.zeros((0, 2), int), np.zeros(0, int)
    d, j = cKDTree(xy).query(xy, k=2)
    pairs = sorted({tuple(sorted((i, int(j[i, 1])))) for i in range(len(xy)) if d[i, 1] <= rmax})
    keep, types = [], []
    for a, b in pairs:
        h = int(cls[a] == 1) + int(cls[b] == 1)
        if h == 2:
            continue
        keep.append((a, b))
        types.append(0 if h == 1 else 1)
    return np.array(keep, int).reshape(-1, 2), np.array(types, int)


def detections(out):
    xy, pk = detect(out[0], thr)
    if len(pk) == 0:
        return xy, np.zeros(0, int)
    return xy, np.argmax(out[1:7, pk[:, 0], pk[:, 1]], axis=0)  # class id: 0 bg, 1 H, 2 C, ...


# --- calibrate reference bond lengths on undistorted val predictions ---
val = np.load(TA / "val_meta.npz")
Xv = np.load(TA / "val_X.npy", mmap_mode="r")
lens = {0: [], 1: []}
for o in predict_many(KIND, model, Xv, val["params"][:, 0], val["params"][:, 1], seed=800):
    xy, cls = detections(o)
    pr, ty = nn_pairs(xy, cls)
    for (a, b), t in zip(pr, ty):
        lens[t].append(np.linalg.norm(xy[a] - xy[b]))
L_REF = {t: float(np.median(v)) for t, v in lens.items()}
print("reference NN lengths (val, predicted):", L_REF, "counts", {t: len(v) for t, v in lens.items()}, flush=True)

# --- distort T2 scenes, predict, correct ---
meta = np.load(TA / "t2_meta.npz")
X2 = np.load(TA / "t2_X.npy", mmap_mode="r")
true_xy, _ = split_atoms(meta)
n, H = len(X2), X2.shape[1]
c0 = (H - 1) / 2
uu, vv = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")  # (row=y, col=x)
ux, uy = vv - c0, uu - c0
dist_stacks, Ds = [], []
for s in range(n):
    sx, sy, sh = rng.uniform(-0.05, 0.05, 3)
    D = np.array([[1 + sx, sh], [0, 1 + sy]])
    tx = D[0, 0] * ux + D[0, 1] * uy + c0
    ty = D[1, 0] * ux + D[1, 1] * uy + c0
    st = np.asarray(X2[s], np.float32)
    dist_stacks.append(np.stack([map_coordinates(st[:, :, z], [ty, tx], order=1, mode="nearest")
                                 for z in range(st.shape[2])], -1))
    Ds.append(D)
outs = predict_many(KIND, model, dist_stacks, meta["params"][:, 0], meta["params"][:, 1], seed=900)

rms_corr, rms_unc, n_bonds = [], [], []
for s in range(n):
    xy, cls = detections(outs[s])
    pr, ty = nn_pairs(xy, cls)
    n_bonds.append(len(pr))
    if len(pr) >= 6:
        dvec = xy[pr[:, 1]] - xy[pr[:, 0]]
        ref = np.array([L_REF[t] for t in ty])
        resid = lambda p: np.linalg.norm(dvec @ np.array([[p[0], p[1]], [0, p[2]]]).T, axis=1) - ref
        fit = least_squares(resid, [1.0, 0.0, 1.0], loss="soft_l1", f_scale=0.1)
        C = np.array([[fit.x[0], fit.x[1]], [0, fit.x[2]]])
    else:
        C = np.eye(2)
    p = true_xy[s] - true_xy[s].mean(0)
    Dinv = np.linalg.inv(Ds[s])
    for M, lst in ((C @ Dinv, rms_corr), (Dinv, rms_unc)):
        _, S = polar(M)
        lst.append(float(np.sqrt(np.mean(np.sum((p @ (S - np.eye(2)).T) ** 2, axis=1)))))

res = {"model": KIND, "n_scenes": n, "L_ref": L_REF,
       "rms_corrected_median": float(np.median(rms_corr)), "rms_corrected_p90": float(np.percentile(rms_corr, 90)),
       "rms_uncorrected_median": float(np.median(rms_unc)), "rms_uncorrected_p90": float(np.percentile(rms_unc, 90)),
       "frac_scenes_improved": float(np.mean(np.array(rms_corr) < np.array(rms_unc))),
       "median_bonds_used": float(np.median(n_bonds)), "target_A": 0.125}
res["S3_pass"] = res["rms_corrected_median"] <= 0.125
(RESULTS / "pilot" / "s3_distortion.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
