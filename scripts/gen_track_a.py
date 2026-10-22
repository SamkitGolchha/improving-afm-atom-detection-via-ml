"""Generate the Track A pilot dataset: ppafm (LJ) stacks of QUAM-AFM Lite molecules + targets.

Per draw: random in-plane rotation, +-1 A translation, amplitude U[0.4,1.4] A, CO stiffness
U[0.4,1.0] N/m, tip-height offset U[-0.3,0.3] A around (mean z + 5.93 A). Molecule mean z is
shifted to 0. Cell 28.8 A (144 px at 0.2 A). Noise is added later, at training time.

Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/gen_track_a.py <split> <draws> [limit]
Writes data/derived/track_a/<split>.npz-style memmaps + index.
"""
import json
import sys
import time

import numpy as np

from afmport.common import DERIVED, NZ, PX, QUAM_Z_REF, quam_xyz_path, read_xyz, z_to_class
from afmport.sim import Simulator
from afmport.targets import render

N_PX = 144
L = N_PX * PX
split_name, draws = sys.argv[1], int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
seed = {"train": 1, "val": 2, "test": 3}[split_name]
rng = np.random.default_rng(seed)
cids = json.loads((DERIVED / "scaffold_split.json").read_text())[split_name][:limit]

out = DERIVED / "track_a"
out.mkdir(parents=True, exist_ok=True)
n = len(cids) * draws
X = np.lib.format.open_memmap(out / f"{split_name}_X.npy", "w+", np.float16, (n, N_PX, N_PX, NZ))
HEAT = np.lib.format.open_memmap(out / f"{split_name}_heat.npy", "w+", np.float16, (n, N_PX, N_PX))
CLS = np.lib.format.open_memmap(out / f"{split_name}_cls.npy", "w+", np.uint8, (n, N_PX, N_PX))
HGT = np.lib.format.open_memmap(out / f"{split_name}_hgt.npy", "w+", np.float16, (n, N_PX, N_PX))
params = np.zeros((n, 6), np.float32)  # amp, stiffness, z_off, rot, tx, ty
atoms_xyz, atoms_cls, offsets, sample_cid = [], [], [0], []

sim = Simulator(N_PX)
t0 = time.time()
i = 0
for cid in cids:
    syms, Zs, xyz, _ = read_xyz(quam_xyz_path(cid))
    cls = z_to_class(Zs)
    base = xyz - np.array([*xyz[:, :2].mean(0), xyz[:, 2].mean()])
    for _ in range(draws):
        for _try in range(20):
            rot = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(rot), np.sin(rot)
            p = base @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            t = rng.uniform(-1, 1, 2)
            p[:, :2] += L / 2 + t
            if p[:, :2].min() > 1.0 and p[:, :2].max() < L - 1.0:
                break
        amp = float(np.round(rng.uniform(0.4, 1.4), 1))
        k = float(rng.uniform(0.4, 1.0))
        z_off = float(rng.uniform(-0.3, 0.3))
        X[i] = sim.simulate(p, Zs, origin=(0.0, 0.0), amp=amp, stiffness=k, z_start=QUAM_Z_REF + z_off, cell=L)
        h, cm, hg = render(p, cls, (0.0, 0.0), N_PX)
        HEAT[i], CLS[i], HGT[i] = h, cm, hg
        params[i] = (amp, k, z_off, rot, *t)
        atoms_xyz.append(p.astype(np.float32))
        atoms_cls.append(cls.astype(np.uint8))
        offsets.append(offsets[-1] + len(p))
        sample_cid.append(cid)
        i += 1
    if len(sample_cid) % (50 * draws) == 0:
        el = time.time() - t0
        print(f"{i}/{n} samples, {el:.0f} s, {el / i * 1000:.0f} ms/sample", flush=True)

np.savez(out / f"{split_name}_meta.npz", params=params, atoms_xyz=np.concatenate(atoms_xyz),
         atoms_cls=np.concatenate(atoms_cls), offsets=np.array(offsets), cid=np.array(sample_cid))
for a in (X, HEAT, CLS, HGT):
    a.flush()
print(f"done: {n} samples in {time.time() - t0:.0f} s ({(time.time() - t0) / n * 1000:.0f} ms/sample)")
