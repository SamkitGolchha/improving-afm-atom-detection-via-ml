"""Phase 0: extract and inspect the experimental ice AFM archive (Zenodo 10054847).

Usage: python3 -I scripts/inspect_ice_data.py <archive.tar.gz> <extract_dir>
"""
import sys
import tarfile
from pathlib import Path

import numpy as np


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        print(f"{len(members)} archive members")
        for m in members:
            print(f"  {m.name}  {m.size} B  {'dir' if m.isdir() else 'file'}")
        # filter='data' rejects absolute paths, '..' traversal, links outside dest, devices.
        tf.extractall(dest, filter="data")


def inspect(dest: Path) -> None:
    files = sorted(dest.rglob("*.npz"))
    print(f"\n{len(files)} .npz files")
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            keys = list(z.keys())
            data = z["data"]
            lx, ly = float(z["lengthX"]), float(z["lengthY"])
            ny, nx = data.shape[0], data.shape[1]
            print(f"\n{f.relative_to(dest)}")
            print(f"  keys={keys}")
            print(f"  data shape={data.shape} dtype={data.dtype}")
            print(f"  extent X={lx:.2f} A  Y={ly:.2f} A")
            print(f"  pixel ~ {lx / nx:.3f} A (x, if axis1=x)  {ly / ny:.3f} A (y, if axis0=y)")
            print(f"  value range [{np.nanmin(data):.3g}, {np.nanmax(data):.3g}]  NaNs={int(np.isnan(data).sum())}")
            for k in keys:
                if k not in ("data", "lengthX", "lengthY"):
                    v = z[k]
                    print(f"  {k}: shape={v.shape} dtype={v.dtype} value={v if v.size < 10 else '...'}")


if __name__ == "__main__":
    archive, dest = Path(sys.argv[1]), Path(sys.argv[2])
    extract(archive, dest)
    inspect(dest)
