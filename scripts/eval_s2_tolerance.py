"""S2: tolerance envelope. One parameter swept at a time, others nominal, on 100 held-out molecules.

Nominal: amplitude 1.0 A, CO stiffness 0.6 N/m, tip-height offset 0, noise 0.05. Training ranges:
amp 0.4-1.4, stiffness 0.4-1.0, offset +-0.3, noise 0-0.15; sweeps extend beyond them.
Envelope = values with mean F1 >= 0.9 x F1 at nominal (locked protocol).
Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/eval_s2_tolerance.py
"""
import json

import numpy as np

from afmport.common import DERIVED, QUAM_Z_REF, RESULTS, quam_xyz_path, read_xyz, z_to_class
from afmport.evaluation import get_threshold, load_model, predict_many, score
from afmport.sim import Simulator

N_PX, L = 144, 28.8
NOM = {"amp": 1.0, "stiffness": 0.6, "z_off": 0.0, "noise": 0.05}
SWEEPS = {
    "amp": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
    "stiffness": [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    "z_off": [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
    "noise": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
}
rng = np.random.default_rng(21)
test = json.loads((DERIVED / "scaffold_split.json").read_text())["test"][:100]
mols = []
for cid in test:
    _, Zs, xyz, _ = read_xyz(quam_xyz_path(cid))
    p = xyz - np.array([*xyz[:, :2].mean(0), xyz[:, 2].mean()])
    a = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(a), np.sin(a)
    p = p @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    p[:, :2] += L / 2
    mols.append((Zs, p))
xy = [p[:, :2] for _, p in mols]
cl = [z_to_class(Zs) for Zs, _ in mols]

sim = Simulator(N_PX)


def stacks_for(amp, k, z_off):
    return [sim.simulate(p, Zs, (0.0, 0.0), amp=amp, stiffness=k, z_start=QUAM_Z_REF + z_off, cell=L) for Zs, p in mols]


cache = {}
results = {}
for kind in ("B0", "B1", "B2"):
    model, thr = load_model(kind), get_threshold(kind)
    results[kind] = {}
    for param, values in SWEEPS.items():
        f1s = []
        for v in values:
            cfg = {**NOM, param: v}
            key = (cfg["amp"], cfg["stiffness"], cfg["z_off"])
            if key not in cache:
                cache[key] = stacks_for(*key)
            n = len(mols)
            outs = predict_many(kind, model, cache[key], np.full(n, cfg["amp"]), np.full(n, cfg["stiffness"]),
                                noise=cfg["noise"], seed=500)
            f1s.append(float(np.mean([r["f1"] for r in score(outs, xy, cl, thr)])))
        nom = f1s[values.index(NOM[param])]
        ok = [v for v, f in zip(values, f1s) if f >= 0.9 * nom]
        results[kind][param] = {"values": values, "f1": f1s, "f1_nominal": nom,
                                "envelope": [min(ok), max(ok)] if ok else None,
                                "envelope_contiguous_with_gaps": len(ok) != (values.index(max(ok)) - values.index(min(ok)) + 1) if ok else None}
        print(kind, param, "F1", [round(f, 3) for f in f1s], "envelope", results[kind][param]["envelope"], flush=True)
(RESULTS / "pilot" / "s2_tolerance.json").write_text(json.dumps(results, indent=1))
print("S2 reported for all four parameters:", all(len(results[k]) == 4 for k in results))
