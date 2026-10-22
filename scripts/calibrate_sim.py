"""Calibrate ppafm raw output convention: axis order, pixel origin, slice order.

Places atoms at known asymmetric positions, simulates with no axis transforms, and reports where
the per-atom contrast peaks land. Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/calibrate_sim.py
"""
import numpy as np
from scipy import ndimage

from afmport.common import PX
from afmport.sim import Simulator

Simulator.RAW_IS_XY = False
Simulator.RAW_Z_CLOSE_FIRST = False
n = 144
sim = Simulator(n)
L = n * PX
atoms = np.array([[8.0, 20.0, 0.0], [20.0, 6.0, 0.0]])  # x, y (A)
Zs = np.array([6, 8], np.int32)
X = sim.simulate(atoms, Zs, origin=(0.0, 0.0), amp=1.0, stiffness=0.4, z_start=6.0, cell=L)
print("raw shape", X.shape)
contrast = [float(X[:, :, k].std()) for k in range(X.shape[2])]
print("per-slice std (index 0 .. last):", np.round(contrast, 4))
print("RAW slice 0 is", "CLOSEST" if contrast[0] > contrast[-1] else "FARTHEST")
k_close = int(np.argmax(contrast))
img = X[:, :, k_close]
# Atom sites are local maxima of (img - smooth background) at close range.
d = img - ndimage.gaussian_filter(img, 10)
for (x, y, _), z in zip(atoms, Zs):
    for name, (i_exp, j_exp) in {"axis0=x": (x / PX, y / PX), "axis0=y": (y / PX, x / PX)}.items():
        i0, j0 = int(round(i_exp)), int(round(j_exp))
        win = d[max(i0 - 8, 0):i0 + 9, max(j0 - 8, 0):j0 + 9]
        di, dj = np.unravel_index(np.argmax(win), win.shape)
        pi, pj = max(i0 - 8, 0) + di, max(j0 - 8, 0) + dj
        print(f"Z={z} at x={x}, y={y}: hypothesis {name}: expected idx ({i_exp:.1f},{j_exp:.1f}) "
              f"peak ({pi},{pj}) value {d[pi, pj]:.3f} offset ({pi - i_exp:+.1f},{pj - j_exp:+.1f}) px")
print("global max of detrended close slice at", np.unravel_index(np.argmax(d), d.shape))
