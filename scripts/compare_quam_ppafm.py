"""Phase 0: how far is coordinates-only ppafm (LJ, no electrostatics) from QUAM-AFM's FDBM images?

For each molecule: simulate its QUAM .xyz with ppafm, sweep the unknown absolute tip height z0
and the 8 image orientations, and report the best mean Pearson correlation between matching
QUAM and ppafm slices (both slice orderings tried). A mismatched-molecule control shows what
"no real agreement" scores look like under the same search.

Usage: ~/.venvs/afm-spinoff/bin/python scripts/compare_quam_ppafm.py <QUAM_dir> <out_dir> [n_mols]
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ppafm
from PIL import Image
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env

QUAM = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
N_MOLS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
PARAM_TAG = "K040_Amp100"  # CO stiffness 0.40 N/m, amplitude 1.00 A (Au(111) ice experiments used 100 pm)
# Scan start heights to try, relative to the molecule's highest atom (QUAM molecules sit at
# different absolute z, e.g. 0.0 vs 0.9 A, so an absolute sweep misses some of them).
# The match is sharp in height (r falls from ~0.9 to ~0.6 within 0.25 A), so sweep finely.
Z0_SWEEP = np.round(np.arange(3.5, 8.51, 0.1), 2)
SYMBOLS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}

OUT.mkdir(parents=True, exist_ok=True)
# Apple OpenCL keeps quotes around ppafm's -I option; build kernels from inside the kernel dir.
os.chdir(Path(ppafm.__file__).parent / "ocl" / "cl")
init_env(i_platform=0)


def read_xyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    cell = float(lines[1].split('Lattice="')[1].split()[0])
    rows = [l.split() for l in lines[2 : 2 + n]]
    xyzs = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    Zs = np.array([SYMBOLS[r[0]] for r in rows], dtype=np.int32)
    return xyzs, Zs, cell


def read_quam_stack(cid):
    folder = next(QUAM.glob(f"K-*/Conformer3D_CID_{cid}_{PARAM_TAG}"))
    files = sorted(folder.glob("*_df_*.jpg"))
    return np.stack([np.asarray(Image.open(f), dtype=np.float32) / 255.0 for f in files], axis=-1)


def zscore(a):
    a = a - a.mean(axis=(0, 1), keepdims=True)
    return a / (a.std(axis=(0, 1), keepdims=True) + 1e-8)


def dihedral(a):
    for k in range(4):
        r = np.rot90(a, k, axes=(0, 1))
        yield f"rot{k * 90}", r
        yield f"rot{k * 90}+flip", r[:, ::-1]


def best_match(quam, sims):
    """Best mean slice-wise correlation over z0, orientation and slice ordering."""
    q = zscore(quam)
    best = (-np.inf, None)
    for z0, sim in sims.items():
        for name, s in dihedral(zscore(sim)):
            for order, ss in (("same", s), ("reversed", s[:, :, ::-1])):
                r = float(np.mean(np.sum(q * ss, axis=(0, 1)) / (q.shape[0] * q.shape[1])))
                if r > best[0]:
                    best = (r, (z0, name, order))
    return best


cids = sorted(p.name for p in (QUAM / "XYZ_FILES").iterdir())[:: max(1, 1755 // N_MOLS)][:N_MOLS]
afm = None
sims_by_cid, quam_by_cid, heights = {}, {}, {}
for cid in cids:
    xyzs, Zs, cell = read_xyz(next((QUAM / "XYZ_FILES" / cid).glob("*.xyz")))
    lvec = np.array([[0, 0, 0], [cell, 0, 0], [0, cell, 0], [0, 0, 18.0]])
    if afm is None:
        afm = AFMulator(
            pixPerAngstrome=10, lvec=lvec, scan_dim=(256, 256, 19),
            scan_window=((0.0, 0.0, Z0_SWEEP[0]), (cell, cell, Z0_SWEEP[0] + 1.8)),
            df_steps=10, tipStiffness=(0.40, 0.40, 0.0, 30.0),
        )
    sims = {}
    ztop = xyzs[:, 2].max()
    heights[cid] = (ztop, xyzs[:, 2].mean(), xyzs[:, 2].min())
    for z0 in Z0_SWEEP:
        z = ztop + z0
        afm.setScanWindow(scan_window=((0.0, 0.0, z), (cell, cell, z + 1.8)), scan_dim=(256, 256, 19), df_steps=10)
        sims[float(z0)] = afm(xyzs, Zs, np.zeros(len(Zs)))
    sims_by_cid[cid] = sims
    quam_by_cid[cid] = read_quam_stack(cid)

print(f"param set {PARAM_TAG}; z0 sweep {Z0_SWEEP[0]}-{Z0_SWEEP[-1]} A above top atom; {len(cids)} molecules\n")
print(f"{'CID':>12} {'matched r':>10} {'control r':>10} {'z_top':>6} {'z_mean':>6} {'z_min':>6} {'best abs z':>10}  (rel z0, orientation, slice order)")
matched, control, abs_best = [], [], []
for i, cid in enumerate(cids):
    r, how = best_match(quam_by_cid[cid], sims_by_cid[cid])
    other = cids[(i + 1) % len(cids)]
    rc, _ = best_match(quam_by_cid[cid], sims_by_cid[other])
    ztop, zmean, zmin = heights[cid]
    zabs = ztop + how[0]
    matched.append(r)
    control.append(rc)
    abs_best.append((zabs, ztop, zmean, zmin))
    print(f"{cid:>12} {r:10.3f} {rc:10.3f} {ztop:6.2f} {zmean:6.2f} {zmin:6.2f} {zabs:10.2f}  {how}")
m, c = np.array(matched), np.array(control)
print(f"\nmean matched r = {m.mean():.3f}  median {np.median(m):.3f}   mean control r = {c.mean():.3f}")
print(f"matched > control in {(m > c).sum()}/{len(m)} molecules")
a = np.array(abs_best)
for j, name in ((1, "z_top"), (2, "z_mean"), (3, "z_min")):
    print(f"best abs z minus {name}: mean {np.mean(a[:, 0] - a[:, j]):.2f}  std {np.std(a[:, 0] - a[:, j]):.2f} A")
print(f"best abs z itself: mean {a[:, 0].mean():.2f}  std {a[:, 0].std():.2f} A")

# Montage for the first molecule at its best settings.
cid = cids[0]
_, (z0, name, order) = best_match(quam_by_cid[cid], sims_by_cid[cid])
sim = dict(dihedral(sims_by_cid[cid][z0]))[name]
sim = sim[:, :, ::-1] if order == "reversed" else sim
q = quam_by_cid[cid]
fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
for col, k in enumerate((0, 3, 6, 9)):
    axes[0, col].imshow(q[:, :, k], cmap="gray")
    axes[0, col].set_title(f"QUAM FDBM slice {k}", fontsize=7)
    axes[1, col].imshow(sim[:, :, k], cmap="gray")
    axes[1, col].set_title(f"ppafm LJ-only slice {k}", fontsize=7)
for ax in axes.flat:
    ax.axis("off")
fig.suptitle(f"CID {cid}, {PARAM_TAG}, z0={z0} A, {name}, {order}", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "quam_vs_ppafm.png", dpi=90)
print("wrote", OUT / "quam_vs_ppafm.png")
