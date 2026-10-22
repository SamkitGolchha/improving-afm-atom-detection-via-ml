"""ppafm (coordinates-only, Lennard-Jones) stack simulation in a fixed pixel convention.

Convention returned by `Simulator.simulate`: array (H, W, NZ) indexed [iy, ix, iz]; pixel (iy, ix)
is centred at (x0 + ix*PX, y0 + iy*PX) in A, where (x0, y0) is the scan origin; iz = 0 is the
FARTHEST slice and iz = NZ-1 the closest (same ordering as the experimental ice data).
The raw ppafm axis order and pixel origin are calibrated empirically (scripts/calibrate_sim.py).
"""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .common import DZ, NZ, PX


@contextmanager
def _in_kernel_dir():
    import ppafm
    prev = os.getcwd()
    # Apple OpenCL keeps quotes around ppafm's -I option; build kernels from inside the kernel dir.
    os.chdir(Path(ppafm.__file__).parent / "ocl" / "cl")
    try:
        yield
    finally:
        os.chdir(prev)


class Simulator:
    """One AFMulator per scan size; reused across molecules and parameters."""

    # Raw-output convention, measured by scripts/calibrate_sim.py: axes (x, y, z), last slice closest.
    RAW_IS_XY = True
    RAW_Z_CLOSE_FIRST = False

    def __init__(self, n_px, lz=20.0):
        from ppafm.ocl.oclUtils import init_env
        with _in_kernel_dir():
            init_env(i_platform=0)
        self.n_px = n_px
        self.L = n_px * PX
        self.lz = lz
        self.afm = None

    def _ensure(self, lvec, scan_window, df_steps, stiffness):
        from ppafm.ocl.AFMulator import AFMulator
        scan_dim = (self.n_px, self.n_px, NZ + df_steps - 1)
        with _in_kernel_dir():
            if self.afm is None:
                self.afm = AFMulator(pixPerAngstrome=10, lvec=lvec, scan_dim=scan_dim,
                                     scan_window=scan_window, df_steps=df_steps,
                                     tipStiffness=(stiffness, stiffness, 0.0, 30.0))
            else:
                self.afm.setLvec(lvec)
                self.afm.setScanWindow(scan_window=scan_window, scan_dim=scan_dim, df_steps=df_steps)
                self.afm.setStiffness((stiffness, stiffness, 0.0, 30.0))

    def simulate(self, xyz, Zs, origin, amp=1.0, stiffness=0.4, z_start=None, cell=None):
        """Simulate a stack over the square window [origin, origin + n_px*PX).

        xyz: (N,3) A. z_start: lowest probe-scan height (closest slice), absolute A.
        cell: periodic cell edge (A) for the force field; defaults to the scan size.
        Returns float32 (n_px, n_px, NZ) in the module convention.
        """
        cell = self.L if cell is None else cell
        lvec = np.array([[0, 0, 0], [cell, 0, 0], [0, cell, 0], [0, 0, self.lz]], dtype=float)
        df_steps = max(1, int(round(amp / DZ)))
        x0, y0 = origin
        # ppafm samples the window endpoint-inclusive (spacing = (x1-x0)/(n-1), calibrated), so the
        # window end is origin + (n-1)*PX to get exactly PX spacing; same for the NZ+df_steps-1 heights.
        z_hi = z_start + (NZ + df_steps - 2) * DZ
        win = ((x0, y0, z_start), (x0 + (self.n_px - 1) * PX, y0 + (self.n_px - 1) * PX, z_hi))
        self._ensure(lvec, win, df_steps, stiffness)
        with _in_kernel_dir():
            X = self.afm(np.asarray(xyz, float), np.asarray(Zs, np.int32), np.zeros(len(Zs)))
        X = np.asarray(X, np.float32)
        if self.RAW_IS_XY:
            X = np.transpose(X, (1, 0, 2))
        if self.RAW_Z_CLOSE_FIRST:
            X = X[:, :, ::-1]
        return np.ascontiguousarray(X)


def zscore_slices(stack, eps=1e-6):
    s = stack - stack.mean(axis=(0, 1), keepdims=True)
    return (s / (s.std(axis=(0, 1), keepdims=True) + eps)).astype(np.float32)
