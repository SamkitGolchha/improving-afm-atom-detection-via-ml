"""Track B pilot dataset from the Aalto AFM-ice simulations (streamed; nothing extracted to disk).

Each sample: 15 PNG slices (192x192, 0.125 A/px, image rows = x, columns = y, as in ppafm raw
output and the experimental .npz arrays) + xyz whose header holds the scan window. Slice 14 is
taken as closest (ml-spm feeds experimental data[..., -15:] whose last slice is closest straight
into models trained on these images). We keep the closest 10 slices.
Targets: O-atom heatmap, class map (O = 4), O height relative to the highest O.

Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/gen_track_b.py <per_set_train> <per_set_val>
"""
import io
import sys
import tarfile

import numpy as np
from PIL import Image

from afmport.common import DERIVED, RAW
from afmport.targets import render

PXB, N, KEEP = 0.125, 192, 10
SETS = ["AFM-ice-Au111-monolayer", "AFM-ice-Au111-bilayer", "AFM-ice-Cu111"]
n_train, n_val = int(sys.argv[1]), int(sys.argv[2])
out = DERIVED / "track_b"
out.mkdir(parents=True, exist_ok=True)


def parse_xyz(txt):
    lines = txt.splitlines()
    hdr = lines[1]
    win = hdr.split("Scan window: ")[1].split(", Amplitude")[0]
    nums = [float(v) for v in win.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    amp = float(hdr.split("Amplitude: ")[1].split(",")[0])
    rows = [l.split() for l in lines[2:] if l.strip()]
    Zs = np.array([int(float(r[0])) for r in rows])
    xyz = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    return np.array(nums[:3]), np.array(nums[3:6]), amp, Zs, xyz


def samples(inner_bytes):
    with tarfile.open(fileobj=io.BytesIO(inner_bytes)) as tf:
        files = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
    for key in sorted({n.split(".")[0] for n in files}, key=int):
        if f"{key}.xyz" not in files or f"{key}.14.png" not in files:
            continue
        stack = np.stack([np.asarray(Image.open(io.BytesIO(files[f"{key}.{i:02d}.png"])), np.float32)
                          for i in range(15 - KEEP, 15)], -1)
        # PNG rows run along -y and columns along x (measured: target~rot90(image) wins in all three
        # sets). Rotate into the (x, y) layout used everywhere else: A[ix, iy] = I[n-1-iy, ix].
        stack = np.ascontiguousarray(np.rot90(stack, -1, axes=(0, 1)))
        yield stack, parse_xyz(files[f"{key}.xyz"].decode())


buf = {s: {k: [] for k in ("X", "heat", "cls", "hgt", "amp", "xyz", "set")} for s in ("train", "val")}
for set_i, name in enumerate(SETS):
    path = RAW / "afm_ice_sims" / f"{name}.tar.gz"
    if not path.exists():
        print(f"SKIP {name}: archive missing")
        continue
    need = {"train": n_train, "val": n_val}
    try:
        with tarfile.open(path, mode="r|gz") as outer:
            for m in outer:
                split = "train" if "_train_" in m.name else "val" if "_val_" in m.name else None
                if split is None or not m.isfile() or need[split] <= 0:
                    if all(v <= 0 for v in need.values()):
                        break
                    continue
                for stack, (w0, w1, amp, Zs, xyz) in samples(outer.extractfile(m).read()):
                    if need[split] <= 0:
                        break
                    O = xyz[Zs == 8]
                    if stack.shape[0] != N:  # bilayer set is 256 px: centre-crop to N, shift window
                        c = (stack.shape[0] - N) // 2
                        stack = stack[c:c + N, c:c + N]
                        w0 = w0 + np.array([c * PXB, c * PXB, 0.0])
                    origin = (w0[0], w0[1])
                    heat, cls, hgt = render(O[:, [1, 0, 2]], np.full(len(O), 4), (origin[1], origin[0]), N,
                                            z_ref=O[:, 2].max(), sigma=0.25, px=PXB)
                    # render() returns [row=y-axis, col=x-axis]; we passed (y, x) so rows = x, cols = y.
                    b = buf[split]
                    b["X"].append(stack.astype(np.float16)); b["heat"].append(heat.astype(np.float16))
                    b["cls"].append(cls); b["hgt"].append(hgt.astype(np.float16)); b["amp"].append(amp)
                    pos = O.copy()
                    pos[:, 0] -= w0[0]
                    pos[:, 1] -= w0[1]
                    pos[:, 2] -= O[:, 2].max()
                    pos = pos[(pos[:, 0] >= 0) & (pos[:, 0] < N * PXB) & (pos[:, 1] >= 0) & (pos[:, 1] < N * PXB)]
                    b["xyz"].append(pos.astype(np.float32)); b["set"].append(set_i)
                    need[split] -= 1
    except (EOFError, tarfile.ReadError) as e:
        print(f"{name}: archive ended early ({type(e).__name__}); kept what was read")
    print(name, {s: sum(1 for v in buf[s]['set'] if v == set_i) for s in buf}, flush=True)

for s, b in buf.items():
    if not b["X"]:
        continue
    for k in ("X", "heat", "cls", "hgt"):
        np.save(out / f"{s}_{k}.npy", np.stack(b[k]))
    offs = np.cumsum([0] + [len(p) for p in b["xyz"]])
    # atoms_xyz stored as (row, col) = (x, y) in A so patch sampling can divide by PXB directly.
    np.savez(out / f"{s}_meta.npz", params=np.stack([b["amp"], np.zeros(len(b["amp"]))], 1).astype(np.float32),
             atoms_xyz=np.concatenate(b["xyz"])[:, [1, 0, 2]], atoms_cls=np.full(offs[-1], 4, np.uint8),
             offsets=offs, set=np.array(b["set"]), px=np.float32(PXB))
    print(s, len(b["X"]), "samples")
