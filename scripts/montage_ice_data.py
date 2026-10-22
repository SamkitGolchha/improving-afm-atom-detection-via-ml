"""Phase 0: montage of first / middle / last slice of each experimental ice stack.

Usage: python3 -I scripts/montage_ice_data.py <extract_dir> <out.png>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(src.rglob("*.npz"))
fig, axes = plt.subplots(len(files), 3, figsize=(7.5, 2.4 * len(files)))
for row, f in zip(axes, files):
    with np.load(f, allow_pickle=False) as z:
        d = z["data"]
    n = d.shape[2]
    for ax, k in zip(row, (0, n // 2, n - 1)):
        ax.imshow(d[:, :, k], cmap="gray", origin="lower")
        ax.set_title(f"{f.stem}  slice {k}/{n - 1}", fontsize=7)
        ax.axis("off")
fig.tight_layout()
fig.savefig(out, dpi=90)
print("wrote", out)
