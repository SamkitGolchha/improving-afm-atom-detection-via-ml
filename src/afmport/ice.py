"""Experimental ice clusters A-I: loading, and registration of the authors' relaxed structures.

Registration reproduces ml-spm papers/ice_structure_discovery (commit 8471fe2):
plot_relaxed_structures.py get_sim() for A-H and plot_prediction_extra.py for I. The whole slab is
rotated by rot_angle about the mean of all atoms; the scan is centred on the xy bounding box of
the top waters (z > max - 1.5 A) plus offset; scan size = (lengthX, lengthY) of the experiment.
The experimental arrays are indexed (x, y, z) like raw ppafm output; pixel i sits at
scan_start + i * size / (n - 1) (ppafm endpoint-inclusive sampling). Last slice is closest.
Ground truth includes periodic images of the slab that fall inside the scan frame.
"""
import numpy as np

from .common import RAW, RESULTS, read_xyz

EXP_DIR = RAW / "ice_afm_zenodo10054847" / "extracted"
GEO_DIR = RESULTS / "phase0" / "geometries"

# label: (experiment file, rot_angle deg, offset (A), authors' pretrained PosNet)
REG = {
    "A": ("Ying_Jiang_7", -145.000, (1.0, 0.0), "au111-monolayer"),
    "B": ("Chen_CO", 80.000, (-1.0, 0.5), "cu111"),
    "C": ("Ying_Jiang_1", -95.000, (0.0, -1.5), "au111-bilayer"),
    "D": ("Ying_Jiang_2_1", 180.000, (1.0, 0.5), "au111-bilayer"),
    "E": ("Ying_Jiang_2_2", -60.000, (0.0, 0.0), "au111-bilayer"),
    "F": ("Ying_Jiang_3", -145.000, (0.0, -3.5), "au111-bilayer"),
    "G": ("Ying_Jiang_5", 94.000, (2.0, 0.0), "au111-bilayer"),
    "H": ("Ying_Jiang_6", -165.974, (1.5, 2.0), "au111-bilayer"),
    "I": ("Ying_Jiang_4", -25.000, (0.0, 0.0), "au111-bilayer"),
}
LABELS = list(REG)


def load_exp(label):
    z = np.load(EXP_DIR / f"{REG[label][0]}.npz", allow_pickle=False)
    return np.asarray(z["data"], np.float32), float(z["lengthX"]), float(z["lengthY"])


def registered_atoms(label, margin=0.0):
    """Water atoms (O, H) of the relaxed structure in the experiment frame.

    Returns xyz (N,3) with x, y in A relative to scan_start, z relative to the highest O, and Zs (N,).
    """
    syms, Zs, xyz, lat = read_xyz(GEO_DIR / f"cluster_{label}.xyz")
    _, ang, off, _ = REG[label]
    data, lx, ly = load_exp(label)
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    center = xyz.mean(0)
    rot = (xyz - center) @ R.T + center
    water = (Zs != 79) & (Zs != 29)
    top = rot[water]
    top = top[top[:, 2] > top[:, 2].max() - 1.5]
    mid = (top[:, :2].max(0) + top[:, :2].min(0)) / 2
    start = mid - np.array([lx, ly]) / 2 + np.array(off)
    # Periodic images (lattice vectors rotated the same way).
    a1, a2 = lat[0] @ R.T, lat[1] @ R.T
    pos, zz = [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            p = rot[water] + i * a1 + j * a2
            pos.append(p)
            zz.append(Zs[water])
    pos, zz = np.concatenate(pos), np.concatenate(zz)
    pos[:, :2] -= start
    inside = ((pos[:, 0] >= -margin) & (pos[:, 0] <= lx + margin) &
              (pos[:, 1] >= -margin) & (pos[:, 1] <= ly + margin))
    pos, zz = pos[inside], zz[inside]
    pos[:, 2] -= pos[zz == 8, 2].max()
    return pos, zz
