"""S5: per-cluster experimental ice check. Our Track B model (B1 recipe, trained on AFM-ice sims)
vs the authors' PosNet, against the authors' relaxed structures (weak labels, see plan Section 5).

1. Frame convention: PosNet O predictions vs registered relaxed O, over 8 axis conventions; the
   best one (by PosNet O recall at 1.0 A, summed over A-H) is adopted for both models. This uses
   only the authors' model and labels, never ours.
2. Our model: threshold tuned on Track B val sims only. Experimental stacks resampled to 0.125 A,
   closest 10 slices kept (shorter stacks zero-filled at the far end, as in training).
3. Metrics per cluster: O recall / precision at 0.5 A (primary) and 1.0 A, O height MAE (heights
   relative to the highest O) over 0.5 A matches. No aggregate claim (n = 9, weak labels).

Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/eval_s5_ice.py
"""
import json

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from afmport.common import RESULTS
from afmport.data import TB
from afmport.evaluation import DEV
from afmport.ice import LABELS, load_exp, registered_atoms
from afmport.infer import predict_patches
from afmport.metrics import detect, f1_scores, match
from afmport.models import make_model
from afmport.sim import zscore_slices

PXB = 0.125
OUT = RESULTS / "pilot" / "track_b"
posnet = json.loads((OUT / "posnet_predictions.json").read_text())

# --- model ---
model = make_model("B1")
model.load_state_dict(torch.load(RESULTS / "pilot" / "models" / "B1_ice.pt", map_location="cpu", weights_only=True))
model = model.to(DEV).eval()


def run(stack_xyz):
    """stack (x, y, z) at 0.125 A, last slice closest -> (8, nx, ny) prediction."""
    s = zscore_slices(stack_xyz[:, :, -10:])
    if s.shape[2] < 10:
        s = np.concatenate([np.zeros(s.shape[:2] + (10 - s.shape[2],), np.float32), s], 2)
    return predict_patches(model, torch.from_numpy(s).permute(2, 0, 1)[None], DEV)[0].numpy()


def peaks(out, thr):
    yx, pk = detect(out[0], thr, px=PXB)  # detect returns (col, row) = (y, x)
    if len(pk) == 0:
        return np.zeros((0, 3))
    return np.column_stack([yx[:, 1], yx[:, 0], out[7, pk[:, 0], pk[:, 1]]])


# --- threshold on Track B val sims ---
val = np.load(TB / "val_meta.npz")
Xv = np.load(TB / "val_X.npy", mmap_mode="r")
o = val["offsets"]
outs = [run(np.asarray(Xv[i], np.float32)) for i in range(len(Xv))]
grid = np.arange(0.1, 0.86, 0.05)


def val_f1(t):
    f = []
    for i, out in enumerate(outs):
        gt = val["atoms_xyz"][o[i]:o[i + 1], :2][:, ::-1]  # stored (y, x) -> (x, y)
        pr = peaks(out, t)[:, :2]
        f.append(f1_scores(len(pr), len(gt), len(match(pr, gt, 0.5)))[2])
    return float(np.mean(f))


scores = {round(float(t), 2): val_f1(t) for t in grid}
THR = max(scores, key=scores.get)
print("Track B val O-F1 by threshold:", {k: round(v, 3) for k, v in scores.items()}, "-> thr", THR, flush=True)


def resample(data, lx, ly):
    nx, ny, nz = data.shape
    mx, my = int(np.floor(lx / PXB)) + 1, int(np.floor(ly / PXB)) + 1
    gx = np.arange(mx) * PXB / (lx / (nx - 1))
    gy = np.arange(my) * PXB / (ly / (ny - 1))
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    return np.stack([map_coordinates(data[:, :, z], [XX, YY], order=1, mode="nearest") for z in range(nz)], -1)


def conv(xy, k, L):
    x, y = xy[:, 0], xy[:, 1]
    lx, ly = L
    return {0: np.c_[x, y], 1: np.c_[y, x], 2: np.c_[lx - x, y], 3: np.c_[x, ly - y],
            4: np.c_[lx - x, ly - y], 5: np.c_[ly - y, x], 6: np.c_[y, lx - x], 7: np.c_[ly - y, lx - x]}[k]


# --- frame convention from PosNet vs relaxed labels ---
gts, Ls = {}, {}
for lab in LABELS:
    pos, Zs = registered_atoms(lab)
    gts[lab] = pos[Zs == 8]
    _, lx, ly = load_exp(lab)
    Ls[lab] = (lx, ly)
conv_recall = {}
for k in range(8):
    hits = tot = 0
    for lab, p in posnet.items():
        pr = np.array(p["xyz"])[np.array(p["elem"]) == 8]
        hits += len(match(conv(pr[:, :2], k, Ls[lab]), gts[lab][:, :2], 1.0))
        tot += len(gts[lab])
    conv_recall[k] = hits / tot
# Identity convention: under it PosNet recovers 100% of O in both monolayer clusters (A, B), where every
# water is imaged. The summed-recall rule above is dominated by invisible lower-layer bilayer O and
# picked a transposed frame (0.207 vs 0.187); this override was made after seeing those numbers (marked dagger).
K = 0
print("PosNet O recall@1.0 by frame convention:", {k: round(v, 3) for k, v in conv_recall.items()}, "-> use", K, flush=True)


def metrics(pred, gt):
    r = {}
    for tol in (0.5, 1.0):
        m = match(pred[:, :2], gt[:, :2], tol)
        p_, r_, f_ = f1_scores(len(pred), len(gt), len(m))
        r[f"recall@{tol}"], r[f"precision@{tol}"] = r_, p_
        if tol == 0.5:
            if m:
                ph = pred[:, 2] - pred[:, 2].max()
                gh = gt[:, 2] - gt[:, 2].max()
                r["height_mae@0.5"] = float(np.mean([abs(ph[i] - gh[j]) for i, j in m]))
            else:
                r["height_mae@0.5"] = None
    # dagger (post hoc): recall over the imaged top layer only (O within 1.5 A of the highest O,
    # the authors' own "top waters" definition in their registration code).
    top = gt[gt[:, 2] > gt[:, 2].max() - 1.5]
    r["top_layer_recall@0.5_dagger"] = len(match(pred[:, :2], top[:, :2], 0.5)) / max(len(top), 1)
    r["top_layer_recall@1.0_dagger"] = len(match(pred[:, :2], top[:, :2], 1.0)) / max(len(top), 1)
    r["n_pred_O"], r["n_true_O"], r["n_top_O"] = len(pred), len(gt), len(top)
    return r


rows = {}
for lab in LABELS:
    data, lx, ly = load_exp(lab)
    out = run(resample(data, lx, ly))
    ours = peaks(out, THR)
    ours[:, :2] = conv(ours[:, :2], K, Ls[lab])
    row = {"ours": metrics(ours, gts[lab])}
    if lab in posnet:
        pr = np.array(posnet[lab]["xyz"])[np.array(posnet[lab]["elem"]) == 8]
        pr = np.c_[conv(pr[:, :2], K, Ls[lab]), pr[:, 2]]
        row["posnet"] = metrics(pr, gts[lab])
    else:
        row["posnet"] = None  # cluster I skipped (memory), as in the authors' own script
    rows[lab] = row
    f = lambda d: "n/a" if d is None else (f"R {d['recall@0.5']:.2f} P {d['precision@0.5']:.2f} (R@1.0 {d['recall@1.0']:.2f}) "
                                           f"topR {d['top_layer_recall@0.5_dagger']:.2f} hMAE {d['height_mae@0.5']}")
    print(f"{lab}: ours {f(row['ours'])} | PosNet {f(row['posnet'])}", flush=True)

res = {"threshold": THR, "val_f1_by_threshold": scores, "frame_convention": K, "posnet_recall_by_convention": conv_recall,
       "clusters": rows, "S5_reported": len(rows) == 9}
(OUT / "s5_ice.json").write_text(json.dumps(res, indent=1))
print("wrote", OUT / "s5_ice.json")
