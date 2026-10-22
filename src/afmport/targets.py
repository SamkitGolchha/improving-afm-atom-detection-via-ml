"""Training targets rendered from atomic coordinates, in the Simulator pixel convention.

heat: max over atoms of exp(-r^2 / 2 sigma^2), in [0, 1]
cls:  element class of the nearest atom within r_cls, else 0 (background)
hgt:  z (relative to z_ref) of that atom, 0 where cls == 0
"""
import numpy as np

from .common import PX


def render(xyz, classes, origin, n_px, z_ref=0.0, sigma=0.25, r_cls=0.35, px=PX):
    x0, y0 = origin
    gx = x0 + np.arange(n_px) * px
    gy = y0 + np.arange(n_px) * px
    heat = np.zeros((n_px, n_px), np.float32)
    best = np.full((n_px, n_px), np.inf, np.float32)
    cls = np.zeros((n_px, n_px), np.uint8)
    hgt = np.zeros((n_px, n_px), np.float32)
    reach = int(np.ceil(4 * sigma / px))
    for (x, y, z), c in zip(xyz, classes):
        ix, iy = int(round((x - x0) / px)), int(round((y - y0) / px))
        xs, xe = max(ix - reach, 0), min(ix + reach + 1, n_px)
        ys, ye = max(iy - reach, 0), min(iy + reach + 1, n_px)
        if xs >= xe or ys >= ye:
            continue
        dx = gx[xs:xe][None, :] - x
        dy = gy[ys:ye][:, None] - y
        r2 = (dx * dx + dy * dy).astype(np.float32)
        np.maximum(heat[ys:ye, xs:xe], np.exp(-r2 / (2 * sigma * sigma)), out=heat[ys:ye, xs:xe])
        near = (r2 < r_cls * r_cls) & (r2 < best[ys:ye, xs:xe])
        best[ys:ye, xs:xe][near] = r2[near]
        cls[ys:ye, xs:xe][near] = c
        hgt[ys:ye, xs:xe][near] = z - z_ref
    return heat, cls, hgt
