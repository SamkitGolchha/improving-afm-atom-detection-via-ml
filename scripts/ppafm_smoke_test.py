"""Phase 0: smoke-test ppafm's GPU AFMulator on this machine (benzene, LJ only, no electrostatics).

Usage: ~/.venvs/afm-spinoff/bin/python scripts/ppafm_smoke_test.py <out.png>
"""
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ppafm
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env

out_path = Path(sys.argv[1]).resolve()
# Apple's OpenCL compiler keeps pyopencl's quotes around ppafm's "-I <kernel dir>" option,
# so `#include "splines.cl"` fails. Building from inside the kernel dir resolves it relatively.
os.chdir(Path(ppafm.__file__).parent / "ocl" / "cl")
init_env(i_platform=0)

# Benzene, flat in z=0 (Angstrom); C-C 1.39, C-H 1.09.
ang = np.arange(6) * np.pi / 3
c = np.stack([1.39 * np.cos(ang), 1.39 * np.sin(ang), np.zeros(6)], axis=1)
h = np.stack([2.48 * np.cos(ang), 2.48 * np.sin(ang), np.zeros(6)], axis=1)
xyzs = np.concatenate([c, h]) + np.array([8.0, 8.0, 5.0])
Zs = np.array([6] * 6 + [1] * 6, dtype=np.int32)
qs = np.zeros(len(Zs))  # no electrostatics: the coordinates-only regime

afm = AFMulator(
    pixPerAngstrome=10,
    scan_dim=(128, 128, 19),  # 19 z-steps, df_steps=10 -> 10 output slices
    scan_window=((4.0, 4.0, 12.0), (12.0, 12.0, 13.9)),
    lvec=np.array([[0, 0, 0], [16, 0, 0], [0, 16, 0], [0, 0, 16]]),
    df_steps=10,
    tipStiffness=(0.25, 0.25, 0.0, 30.0),
)

t0 = time.perf_counter()
X = afm(xyzs, Zs, qs)
t1 = time.perf_counter()
X2 = afm(xyzs, Zs, qs)  # second call excludes one-time kernel compilation
t2 = time.perf_counter()
print(f"output shape {X.shape} dtype {X.dtype}")
print(f"first call {t1 - t0:.2f} s, second call {t2 - t1:.3f} s")
print(f"df range [{X.min():.3g}, {X.max():.3g}]  deterministic={np.allclose(X, X2)}")

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
for ax, k in zip(axes, (0, X.shape[2] // 2, X.shape[2] - 1)):
    ax.imshow(X[:, :, k].T, cmap="gray", origin="lower")
    ax.set_title(f"slice {k}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(out_path, dpi=90)
print("wrote", out_path)
