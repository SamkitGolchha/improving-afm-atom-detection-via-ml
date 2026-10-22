"""Shared constants, paths and small IO helpers."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
QUAM_DIR = RAW / "quam_afm_lite_BFAU11" / "extracted" / "SUBMIT_QUAM-AFM" / "QUAM"

PX = 0.2            # A per pixel (locked pilot protocol)
NZ = 10             # constant-height slices per stack
DZ = 0.1            # A between slices
QUAM_Z_REF = 5.93   # scan start above molecule mean z (Phase 0 finding)
QUAM_CELL = 24.0270328648

SYMBOLS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
           "Cu": 29, "Au": 79}
# Element classes for the organic track: 0 background, 1 H, 2 C, 3 N, 4 O, 5 other heavy (F P S Cl Br I).
CLASS_NAMES = ["bg", "H", "C", "N", "O", "X"]
Z_TO_CLASS = {1: 1, 6: 2, 7: 3, 8: 4}


def z_to_class(Zs):
    return np.array([Z_TO_CLASS.get(int(z), 5) for z in Zs], dtype=np.int64)


def read_xyz(path):
    """Read an (extended) xyz file -> (symbols, Zs, xyz (N,3), lattice 3x3 or None)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    lat = None
    if 'Lattice="' in lines[1]:
        vals = [float(v) for v in lines[1].split('Lattice="')[1].split('"')[0].split()]
        lat = np.array(vals).reshape(3, 3)
    rows = [l.split() for l in lines[2:2 + n]]
    syms = [r[0] for r in rows]
    xyz = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    Zs = np.array([SYMBOLS[s] for s in syms], dtype=np.int32)
    return syms, Zs, xyz, lat


def quam_xyz_path(cid):
    return next((QUAM_DIR / "XYZ_FILES" / str(cid)).glob("*.xyz"))


def quam_cids():
    return sorted(p.name for p in (QUAM_DIR / "XYZ_FILES").iterdir() if p.is_dir())
