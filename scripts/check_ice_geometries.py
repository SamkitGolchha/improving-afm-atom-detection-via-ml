"""Check the relaxed ice-cluster geometries (Zenodo 10362511) behind the experimental check.

Reads only the atom block of each hartree_<label>.xsf (the potential grid is skipped) and reports,
per cluster: water count, H/O ratio, O-H bond sanity, O heights above the metal surface,
monolayer/bilayer character, and lateral extent. Also compares clusters D and E, which map to
experiment files Ying_Jiang_2_1 and Ying_Jiang_2_2.

Label -> experiment mapping is from ml-spm papers/ice_structure_discovery/predictions/
plot_relaxed_structures.py (A-H) and the paper's one extra cluster (I).

The archive holds ~1.7 GB XSF files (mostly potential grid), so it is streamed: only each file's
atom block is read, and the geometry is saved as a small .xyz in <out_dir>/geometries/.

Usage: python3 -I scripts/check_ice_geometries.py <relaxed_structures.tar.gz> <out_dir>
"""
import sys
import tarfile
from pathlib import Path

import numpy as np

EXP_FILE = {
    "A": "Ying_Jiang_7", "B": "Chen_CO", "C": "Ying_Jiang_1", "D": "Ying_Jiang_2_1",
    "E": "Ying_Jiang_2_2", "F": "Ying_Jiang_3", "G": "Ying_Jiang_5", "H": "Ying_Jiang_6",
    "I": "Ying_Jiang_4",
}
PAPER_LAYER = {"A": "mono", "B": "mono"}  # paper: A, B monolayers; C-H bilayers
# Experimental scan extents (lengthX, lengthY) from the .npz files, for a fit-in-frame check.
EXP_SCAN = {"A": "25.0,25.0", "B": "25.3,25.3", "C": "32.0,32.0", "D": "34.0,34.0", "E": "34.0,34.0",
            "F": "40.0,40.0", "G": "40.0,40.0", "H": "39.0,39.0", "I": "60.0,60.0"}
METALS = {29: "Cu", 79: "Au"}


def read_xsf_atoms(f):
    """Return (Zs, xyz, cell) from an iterable of XSF text lines, stopping before any DATAGRID block."""
    cell, atoms = None, []
    if True:
        lines = iter(f)
        for line in lines:
            key = line.strip().upper()
            if key.startswith("BEGIN_BLOCK_DATAGRID") or key.startswith("BEGIN_DATAGRID"):
                break
            if key == "PRIMVEC":
                cell = np.array([[float(v) for v in next(lines).split()] for _ in range(3)])
            elif key == "PRIMCOORD":
                n = int(next(lines).split()[0])
                for _ in range(n):
                    p = next(lines).split()
                    atoms.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
            elif key == "ATOMS" and not atoms:
                for l in lines:
                    p = l.split()
                    if len(p) < 4 or not p[0].isdigit():
                        break
                    atoms.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    a = np.array(atoms)
    return a[:, 0].astype(int), a[:, 1:4], cell


def min_image(d, cell):
    """Minimum-image displacement in the xy plane (periodic slab)."""
    if cell is None:
        return d
    inv = np.linalg.inv(cell[:2, :2])
    frac = d[..., :2] @ inv
    frac -= np.round(frac)
    out = d.copy()
    out[..., :2] = frac @ cell[:2, :2]
    return out


SYMBOL = {1: "H", 8: "O", 29: "Cu", 79: "Au"}


def write_xyz(path, label, Zs, xyz, cell):
    lat = "" if cell is None else 'Lattice="' + " ".join(f"{v:.6f}" for v in cell.ravel()) + '" '
    rows = [f"{SYMBOL.get(int(z), z)} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(Zs, xyz)]
    path.write_text(f"{len(Zs)}\n{lat}cluster={label} exp={EXP_FILE.get(label, '?')}\n" + "\n".join(rows) + "\n")


def analyze(label, Zs, xyz, cell):
    metal_mask = np.isin(Zs, list(METALS))
    O, H = xyz[Zs == 8], xyz[Zs == 1]
    metal = METALS.get(int(Zs[metal_mask][0]), "?") if metal_mask.any() else "none"
    # Some slabs have their bottom layer wrapped through the z boundary (z ~ cell_z - 0.1),
    # so map metal z above half the cell back down before taking the top surface plane.
    mz = xyz[metal_mask, 2].copy()
    if cell is not None:
        cz = cell[2, 2]
        mz[mz > cz / 2] -= cz
    surf_top = mz.max() if metal_mask.any() else np.nan

    # Each H to its nearest O.
    d = min_image(H[:, None, :] - O[None, :, :], cell)
    dist = np.linalg.norm(d, axis=-1)
    nearest = dist.argmin(axis=1)
    oh = dist[np.arange(len(H)), nearest]
    h_per_o = np.bincount(nearest, minlength=len(O))

    o_h = O[:, 2] - surf_top
    # Heights are continuous in 3D clusters, so no hard layer split. Heuristic second-layer proxy:
    # first-layer water on these metals sits ~2.5-3.5 A above the top metal plane.
    n_upper = int((o_h > 4.0).sum())

    # Nearest O-O distances (H-bond network sanity, expect ~2.6-3.0 A).
    doo = np.linalg.norm(min_image(O[:, None, :] - O[None, :, :], cell), axis=-1)
    np.fill_diagonal(doo, np.inf)
    oo_nn = doo.min(axis=1)

    # Clusters can straddle the periodic boundary. Unwrap by walking the H-bond network
    # (O-O < 3.3 A, minimum image) outward from O[0], so large clusters unwrap correctly.
    O_unw = O.copy()
    seen = np.zeros(len(O), bool)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        step = min_image(O - O_unw[i], cell)
        for j in np.nonzero((np.linalg.norm(step, axis=1) < 3.3) & ~seen)[0]:
            O_unw[j] = O_unw[i] + step[j]
            seen[j] = True
            stack.append(j)
    n_components_unvisited = int((~seen).sum())
    ext = O_unw[:, :2].max(axis=0) - O_unw[:, :2].min(axis=0)
    return {
        "label": label, "exp": EXP_FILE.get(label, "?"), "metal": metal,
        "n_metal": int(metal_mask.sum()), "n_O": len(O), "n_H": len(H),
        "oh_min": oh.min(), "oh_max": oh.max(), "h_per_o": np.bincount(h_per_o).tolist(),
        "o_h_min": o_h.min(), "o_h_max": o_h.max(), "n_upper": n_upper,
        "oo_nn_min": oo_nn.min(), "oo_nn_max": oo_nn.max(), "extent": ext,
        "unconnected_O": n_components_unvisited,
        "cell": None if cell is None else np.diag(cell)[:2], "O": O_unw, "surf": surf_top,
    }


def main():
    archive, out = Path(sys.argv[1]), Path(sys.argv[2])
    geo_dir = out / "geometries"
    geo_dir.mkdir(parents=True, exist_ok=True)
    res, others = {}, []
    try:
        with tarfile.open(archive, mode="r|gz") as tf:  # streaming: no random access, no extraction
            for m in tf:
                name = Path(m.name).name
                if not (m.isfile() and name.startswith("hartree_") and name.endswith(".xsf")):
                    others.append(f"{m.name} ({m.size} B)")
                    continue
                label = Path(name).stem.split("_", 1)[1]
                raw = tf.extractfile(m)  # stream member: not seekable, so decode line by line
                Zs, xyz, cell = read_xsf_atoms(l.decode("ascii", "replace") for l in iter(raw.readline, b""))
                write_xyz(geo_dir / f"cluster_{label}.xyz", label, Zs, xyz, cell)
                res[label] = analyze(label, Zs, xyz, cell)
                print(f"read {m.name}: {len(Zs)} atoms")
    except (EOFError, tarfile.ReadError) as e:
        print(f"archive ended early ({type(e).__name__}): partial download? results below are incomplete")
    print(f"{len(res)} clusters read; other members: {others[:20]}")

    lines = []
    hdr = (f"{'lbl':3} {'experiment':15} {'metal':5} {'#M':>4} {'#O':>3} {'#H':>3} {'H/O':>5} "
           f"{'O-H min-max (A)':>16} {'H per O hist':>14} {'O height above surf':>20} {'#O>4A':>6} "
           f"{'paper':>6} {'O-O nn min-max':>15} {'O extent x,y (A)':>17} {'exp scan x,y (A)':>17} {'O off-network':>13}")
    lines.append(hdr)
    for k in sorted(res):
        r = res[k]
        paper = PAPER_LAYER.get(k, "bi" if k in "CDEFGH" else "?")
        lines.append(
            f"{k:3} {r['exp']:15} {r['metal']:5} {r['n_metal']:4d} {r['n_O']:3d} {r['n_H']:3d} "
            f"{r['n_H'] / max(r['n_O'], 1):5.2f} {r['oh_min']:7.2f}-{r['oh_max']:<7.2f} "
            f"{str(r['h_per_o']):>14} {r['o_h_min']:9.2f}-{r['o_h_max']:<9.2f} {r['n_upper']:6d} "
            f"{paper:>6} {r['oo_nn_min']:6.2f}-{r['oo_nn_max']:<7.2f} "
            f"{r['extent'][0]:7.1f},{r['extent'][1]:<7.1f} "
            f"{EXP_SCAN.get(k, '?'):>17} {r['unconnected_O']:13d}"
        )
    if "D" in res and "E" in res:
        D, E = res["D"], res["E"]
        lines.append("")
        lines.append(f"D vs E: #O {D['n_O']} vs {E['n_O']}; O heights {D['o_h_min']:.2f}-{D['o_h_max']:.2f} "
                     f"vs {E['o_h_min']:.2f}-{E['o_h_max']:.2f} A; extents {np.round(D['extent'], 1)} vs "
                     f"{np.round(E['extent'], 1)} A")
        if D["n_O"] == E["n_O"]:
            # Compare sorted O-O distance spectra (rotation/translation invariant).
            def spec(O):
                dd = np.linalg.norm(O[:, None] - O[None], axis=-1)
                return np.sort(dd[np.triu_indices(len(O), 1)])
            diff = np.abs(spec(D["O"]) - spec(E["O"]))
            lines.append(f"D vs E O-O distance spectrum: mean |diff| {diff.mean():.3f} A, max {diff.max():.3f} A")
    text = "\n".join(lines)
    print(text)
    (out / "ice_geometries.txt").write_text(text + "\n")
    print("wrote", out / "ice_geometries.txt")


if __name__ == "__main__":
    main()
