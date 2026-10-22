"""Track A datasets over the generated memmaps, with on-the-fly noise and dihedral augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import DERIVED
from .models import PATCH
from .sim import zscore_slices

TA = DERIVED / "track_a"
TB = DERIVED / "track_b"


def load_split(name, root=TA):
    d = {k: np.load(root / f"{name}_{k}.npy", mmap_mode="r") for k in ("X", "heat", "cls", "hgt")}
    meta = np.load(root / f"{name}_meta.npz")
    d.update({k: meta[k] for k in meta.files})
    return d


def add_noise(stack, rel_sigma, rng):
    if rel_sigma <= 0:
        return stack
    s = stack.std(axis=(0, 1), keepdims=True)
    return stack + rng.normal(0, 1, stack.shape).astype(np.float32) * rel_sigma * s


def dihedral(arrs, k, flip):
    out = []
    for a in arrs:
        a = np.rot90(a, k, axes=(0, 1))
        if flip:
            a = a[:, ::-1]
        out.append(np.ascontiguousarray(a))
    return out


class TrackA(Dataset):
    """mode 'whole' -> full 144 px images; mode 'patch' -> PATCH crops, mostly centred near atoms."""

    def __init__(self, name, mode, noise=(0.0, 0.15), augment=True, n_items=None, seed=0, root=TA,
                 min_slices=None):
        self.d = load_split(name, root)
        self.px = float(self.d["px"]) if "px" in self.d else 0.2
        self.min_slices = min_slices  # Track B: randomly drop far slices (zero-filled) to mimic short stacks
        self.mode, self.noise, self.augment = mode, noise, augment
        self.n = len(self.d["X"])
        self.n_items = n_items or self.n
        self.seed = seed

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx, torch.initial_seed() % 2**31 if self.augment else 0))
        i = idx % self.n
        d = self.d
        x = np.asarray(d["X"][i], np.float32)
        heat = np.asarray(d["heat"][i], np.float32)
        cls = np.asarray(d["cls"][i], np.int64)
        hgt = np.asarray(d["hgt"][i], np.float32)
        if self.mode == "patch":
            n_px = x.shape[0]
            if rng.random() < 0.8:
                a0, a1 = d["offsets"][i], d["offsets"][i + 1]
                ax, ay = d["atoms_xyz"][rng.integers(a0, a1), :2] / self.px
                cy, cx = int(ay) + rng.integers(-8, 9), int(ax) + rng.integers(-8, 9)
            else:
                cy, cx = rng.integers(0, n_px), rng.integers(0, n_px)
            y0 = int(np.clip(cy - PATCH // 2, 0, n_px - PATCH))
            x0 = int(np.clip(cx - PATCH // 2, 0, n_px - PATCH))
            sl = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
            x, heat, cls, hgt = x[sl], heat[sl], cls[sl], hgt[sl]
        sig = rng.uniform(*self.noise) if self.augment else self.noise[0]
        x = zscore_slices(add_noise(x, sig, rng))
        if self.min_slices and self.augment:
            keep = int(rng.integers(self.min_slices, x.shape[2] + 1))
            x[:, :, :x.shape[2] - keep] = 0.0
        if self.augment:
            x, heat, cls, hgt = dihedral([x, heat, cls, hgt], int(rng.integers(4)), bool(rng.integers(2)))
        amp, k = d["params"][i, 0], d["params"][i, 1]
        return (torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(heat)[None], torch.from_numpy(cls),
                torch.from_numpy(hgt)[None], torch.tensor([amp, k], dtype=torch.float32))
