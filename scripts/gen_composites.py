"""T2: multi-molecule composite scenes from held-out (test-split) molecules.

Each scene: 48 A cell (240 px), 3-5 molecules, random rotations and positions, inter-molecule atom
distance >= 3.0 A, all atoms >= 1 A from the edges. Molecule mean z shifted to 0. Acquisition
parameters drawn from the training ranges. Usage: PYTHONPATH=src python scripts/gen_composites.py <n>
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from afmport.common import DERIVED, QUAM_Z_REF, quam_xyz_path, read_xyz, z_to_class
from afmport.sim import Simulator

N_PX, L = 240, 48.0
n_scenes = int(sys.argv[1])
rng = np.random.default_rng(11)
test = json.loads((DERIVED / "scaffold_split.json").read_text())["test"]
mols = {}
for cid in test:
    _, Zs, xyz, _ = read_xyz(quam_xyz_path(cid))
    mols[cid] = (Zs, xyz - np.array([*xyz[:, :2].mean(0), xyz[:, 2].mean()]))

sim = Simulator(N_PX)
X = np.zeros((n_scenes, N_PX, N_PX, 10), np.float16)
params, atoms, cls_all, offsets, members = [], [], [], [0], []
for s in range(n_scenes):
    want = int(rng.integers(3, 6))
    pos, Zl, used = np.zeros((0, 3)), np.zeros(0, np.int32), []
    for _ in range(200):
        if len(used) == want:
            break
        cid = test[rng.integers(len(test))]
        Zs, base = mols[cid]
        a = rng.uniform(0, 2 * np.pi)
        c, s_ = np.cos(a), np.sin(a)
        p = base @ np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]])
        p[:, :2] += rng.uniform(4, L - 4, 2)
        if p[:, :2].min() < 1.0 or p[:, :2].max() > L - 1.0:
            continue
        if len(pos) and cKDTree(pos[:, :2]).query(p[:, :2])[0].min() < 3.0:
            continue
        pos, Zl = np.concatenate([pos, p]), np.concatenate([Zl, Zs])
        used.append(cid)
    amp = float(np.round(rng.uniform(0.4, 1.4), 1))
    k = float(rng.uniform(0.4, 1.0))
    z_off = float(rng.uniform(-0.3, 0.3))
    X[s] = sim.simulate(pos, Zl, origin=(0.0, 0.0), amp=amp, stiffness=k, z_start=QUAM_Z_REF + z_off, cell=L)
    params.append((amp, k, z_off))
    atoms.append(pos.astype(np.float32))
    cls_all.append(z_to_class(Zl).astype(np.uint8))
    offsets.append(offsets[-1] + len(pos))
    members.append(",".join(used))
out = DERIVED / "track_a"
np.save(out / "t2_X.npy", X)
np.savez(out / "t2_meta.npz", params=np.array(params, np.float32), atoms_xyz=np.concatenate(atoms),
         atoms_cls=np.concatenate(cls_all), offsets=np.array(offsets), members=np.array(members))
n_m = [len(m.split(",")) for m in members]
print(f"{n_scenes} scenes; molecules per scene {np.bincount(n_m)[3:]} (3/4/5); atoms {offsets[-1]}")
