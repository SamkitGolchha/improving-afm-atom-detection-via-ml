"""RDKit Bemis-Murcko scaffold split of QUAM-AFM Lite molecules (80/10/10, seed 0).

Bonds are perceived from 3D coordinates. Acyclic molecules (empty scaffold) are each their own
group. Whole scaffold groups go to one split, so test scaffolds never appear in training.
Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/make_split.py
"""
import json
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.Scaffolds import MurckoScaffold

from afmport.common import DERIVED, quam_cids, quam_xyz_path, read_xyz

RDLogger.DisableLog("rdApp.*")


def scaffold(cid):
    syms, Zs, xyz, _ = read_xyz(quam_xyz_path(cid))
    block = f"{len(syms)}\n\n" + "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(syms, xyz))
    mol = Chem.MolFromXYZBlock(block)
    rdDetermineBonds.DetermineConnectivity(mol)
    Chem.FastFindRings(mol)  # Murcko decomposition needs ring info; connectivity-only mols lack it
    scf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scf) if scf.GetNumAtoms() else ""


cids = quam_cids()
groups, failed = defaultdict(list), []
for cid in cids:
    try:
        s = scaffold(cid)
    except Exception as e:  # perception failure: own group, recorded
        failed.append(cid)
        s = ""
    groups[s if s else f"acyclic:{cid}"].append(cid)

rng = np.random.default_rng(0)
keys = list(groups)
rng.shuffle(keys)
n = len(cids)
split = {"train": [], "val": [], "test": []}
for k in keys:
    g = groups[k]
    if len(split["test"]) + len(g) <= 0.1 * n:
        split["test"] += g
    elif len(split["val"]) + len(g) <= 0.1 * n:
        split["val"] += g
    else:
        split["train"] += g
DERIVED.mkdir(parents=True, exist_ok=True)
out = DERIVED / "scaffold_split.json"
out.write_text(json.dumps({**split, "perception_failed": failed}, indent=1))
print({k: len(v) for k, v in split.items()}, "scaffold groups", len(groups),
      "acyclic", sum(k.startswith("acyclic") for k in groups), "perception failures", len(failed))
print("largest groups:", sorted(((len(v), k) for k, v in groups.items()), reverse=True)[:5])
print("wrote", out)
