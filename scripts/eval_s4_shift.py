"""S4: simulator shift. LJ (our ppafm) vs QUAM FDBM images of the same held-out molecules.

QUAM -> our convention, from Phase 0: q ~ rot90(raw_ppafm, 1)[..., ::-1] with raw axes (x, y),
so raw = rot90(q[..., ::-1], -1) and ours = raw^T; raw pixel j sits at j*cell/255 A. Resampled to
0.2 A. A mapping check (correlation with our LJ simulation of the same molecule) guards against a
wrong mapping masquerading as a simulator shift.
OOD flag (B3): mean per-pixel std of B2 heat predictions over the 8 rotations/reflections.
Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/eval_s4_shift.py
"""
import json

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates

from afmport.common import DERIVED, QUAM_CELL, QUAM_DIR, QUAM_Z_REF, RESULTS, quam_xyz_path, read_xyz, z_to_class
from afmport.evaluation import DEV, get_threshold, load_model, predict_many, prep, score
from afmport.infer import predict
from afmport.sim import Simulator, zscore_slices

N_PX = 120
FOLDERS = ["K040_Amp040", "K040_Amp100", "K040_Amp140", "K100_Amp040", "K100_Amp100", "K100_Amp140"]
test = json.loads((DERIVED / "scaffold_split.json").read_text())["test"]
g = np.arange(N_PX) * 0.2 * 255 / QUAM_CELL  # our pixel -> raw QUAM pixel index


def quam_stack(cid, tag):
    folder = next(QUAM_DIR.glob(f"K-*/Conformer3D_CID_{cid}_{tag}"))
    q = np.stack([np.asarray(Image.open(f), np.float32) for f in sorted(folder.glob("*_df_*.jpg"))], -1)
    raw = np.rot90(q[..., ::-1], -1, axes=(0, 1))
    ours = np.transpose(raw, (1, 0, 2))
    yy, xx = np.meshgrid(g, g, indexing="ij")
    return np.stack([map_coordinates(ours[:, :, z], [yy, xx], order=1) for z in range(ours.shape[2])], -1)


sim = Simulator(N_PX)
lj, fd, amps, ks, xy, cl = [], [], [], [], [], []
for tag in FOLDERS:
    k, amp = int(tag[1:4]) / 100, int(tag[8:11]) / 100
    for cid in test:
        _, Zs, xyz, _ = read_xyz(quam_xyz_path(cid))
        p = xyz.copy()
        p[:, 2] -= xyz[:, 2].mean()
        lj.append(sim.simulate(p, Zs, (0.0, 0.0), amp=amp, stiffness=k, z_start=QUAM_Z_REF, cell=QUAM_CELL))
        fd.append(quam_stack(cid, tag))
        amps.append(amp), ks.append(k), xy.append(p[:, :2]), cl.append(z_to_class(Zs))
amps, ks = np.array(amps), np.array(ks)

# Mapping check: per-slice correlation between z-scored QUAM and our LJ stacks.
r = [float(np.mean(np.sum(zscore_slices(a) * zscore_slices(b), axis=(0, 1)) / (N_PX * N_PX))) for a, b in zip(lj, fd)]
print(f"mapping check: mean r(LJ, FDBM) = {np.mean(r):.3f} (median {np.median(r):.3f}, 5th pct {np.percentile(r, 5):.3f}) "
      f"over {len(r)} stacks", flush=True)

res = {"mapping_check_r": {"mean": float(np.mean(r)), "median": float(np.median(r)),
                           "p5": float(np.percentile(r, 5))}, "n_stacks": len(lj), "folders": FOLDERS}
for kind in ("B0", "B1", "B2"):
    model, thr = load_model(kind), get_threshold(kind)
    f_lj = [x["f1"] for x in score(predict_many(kind, model, lj, amps, ks, seed=600), xy, cl, thr)]
    f_fd = [x["f1"] for x in score(predict_many(kind, model, fd, amps, ks, seed=600), xy, cl, thr)]
    res[kind] = {"f1_LJ": float(np.mean(f_lj)), "f1_FDBM": float(np.mean(f_fd)),
                 "drop": float(np.mean(f_lj) - np.mean(f_fd)),
                 "per_folder_FDBM": {t: float(np.mean(f_fd[i * len(test):(i + 1) * len(test)])) for i, t in enumerate(FOLDERS)}}
    print(kind, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in res[kind].items()}, flush=True)


# B3: dihedral-disagreement OOD score with B2, 200 LJ vs 200 FDBM stacks.
def dihedral_score(model, stack, amp, k, rng):
    x = prep(stack, 0.05, rng)
    preds = []
    for rot in range(4):
        for flip in (False, True):
            t = torch.rot90(x, rot, dims=(1, 2))
            if flip:
                t = torch.flip(t, dims=(2,))
            o = predict("B2", model, t[None], DEV, [amp], [k])[0, 0]
            if flip:
                o = torch.flip(o, dims=(1,))
            preds.append(torch.rot90(o, -rot, dims=(0, 1)))
    return float(torch.stack(preds).std(0).mean())


b2 = load_model("B2")
rng = np.random.default_rng(700)
idx = rng.choice(len(lj), 200, replace=False)
s_lj = [dihedral_score(b2, lj[i], amps[i], ks[i], rng) for i in idx]
s_fd = [dihedral_score(b2, fd[i], amps[i], ks[i], rng) for i in idx]
# AUROC = P(score_FDBM > score_LJ), ties count half (Mann-Whitney).
a, b = np.array(s_lj), np.array(s_fd)
auroc = float(((b[:, None] > a[None]).sum() + 0.5 * (b[:, None] == a[None]).sum()) / (len(a) * len(b)))
res["B3_ood"] = {"auroc": auroc, "mean_score_LJ": float(a.mean()), "mean_score_FDBM": float(b.mean()), "n_each": 200}
res["S4_pass"] = auroc >= 0.8
print("B3 OOD AUROC", round(auroc, 3), "S4 pass:", res["S4_pass"])
(RESULTS / "pilot" / "s4_shift.json").write_text(json.dumps(res, indent=1))
