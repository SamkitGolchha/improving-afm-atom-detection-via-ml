"""Atom detection from heatmaps, matching, F1, element accuracy, paired bootstrap."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

from .common import PX


def detect(heat, threshold, min_dist_A=0.6, origin=(0.0, 0.0), px=PX):
    """Return (K, 2) (col, row) positions in A of heatmap peaks, and the (row, col) pixel peaks."""
    pk = peak_local_max(np.asarray(heat, np.float32), min_distance=max(1, int(round(min_dist_A / px))),
                        threshold_abs=threshold, exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 2)), pk
    return np.stack([origin[0] + pk[:, 1] * px, origin[1] + pk[:, 0] * px], 1), pk


def match(pred_xy, true_xy, tol=0.5):
    """Optimal one-to-one matching within tol (A). Returns list of (i_pred, j_true)."""
    if len(pred_xy) == 0 or len(true_xy) == 0:
        return []
    d = np.linalg.norm(pred_xy[:, None] - true_xy[None], axis=-1)
    cost = np.where(d <= tol, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if d[i, j] <= tol]


def f1_scores(n_pred, n_true, n_match):
    p = n_match / n_pred if n_pred else (1.0 if n_true == 0 else 0.0)
    r = n_match / n_true if n_true else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_sample(out, true_xy, true_cls, threshold, tol=0.5, origin=(0.0, 0.0)):
    """out: (8, H, W) numpy prediction. true_xy (N,2) A, true_cls (N,)."""
    heat = out[0]
    pred_xy, pk = detect(heat, threshold, origin=origin)
    m = match(pred_xy, true_xy, tol)
    p, r, f = f1_scores(len(pred_xy), len(true_xy), len(m))
    if m:
        pred_cls = np.argmax(out[1:7, pk[[i for i, _ in m], 0], pk[[i for i, _ in m], 1]][1:], axis=0) + 1
        el_acc = float(np.mean(pred_cls == true_cls[[j for _, j in m]]))
    else:
        el_acc = np.nan
    return {"precision": p, "recall": r, "f1": f, "elem_acc": el_acc, "n_pred": len(pred_xy),
            "n_true": len(true_xy), "n_match": len(m)}


def paired_bootstrap(a, b, n=10000, seed=0):
    """95% CI of mean(b - a) over paired samples."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
