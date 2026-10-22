"""Tune detection thresholds on val (only), then evaluate B0/B1/B2 on T1 (held-out molecules) and
T2 (composites) and test S1 with a paired bootstrap. Noise at the nominal 0.05.

Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/eval_t1_t2.py
"""
import json

import numpy as np

from afmport.common import RESULTS
from afmport.data import TA
from afmport.evaluation import load_model, predict_many, score, split_atoms, thresholds_path, tune_threshold
from afmport.metrics import paired_bootstrap

KINDS = ["B0", "B1", "B2"]
out_dir = RESULTS / "pilot"
res, thr = {}, {}
sets = {}
for name, xfile, mfile in (("val", "val_X.npy", "val_meta.npz"), ("T1", "test_X.npy", "test_meta.npz"),
                           ("T2", "t2_X.npy", "t2_meta.npz")):
    meta = np.load(TA / mfile)
    xy, cl = split_atoms(meta)
    sets[name] = (np.load(TA / xfile, mmap_mode="r"), meta["params"][:, 0], meta["params"][:, 1], xy, cl)

for kind in KINDS:
    model = load_model(kind)
    X, a, k, xy, cl = sets["val"]
    outs = predict_many(kind, model, X, a, k, seed=100)
    thr[kind] = tune_threshold(outs, xy, cl)
    res[kind] = {}
    for name in ("T1", "T2"):
        X, a, k, xy, cl = sets[name]
        outs = predict_many(kind, model, X, a, k, seed=200 if name == "T1" else 300)
        res[kind][name] = score(outs, xy, cl, thr[kind])
    print(kind, "threshold", thr[kind], {n: round(float(np.mean([r["f1"] for r in res[kind][n]])), 4)
                                          for n in ("T1", "T2")}, flush=True)
thresholds_path().write_text(json.dumps(thr, indent=1))

summary = {"thresholds": thr}
for name in ("T1", "T2"):
    summary[name] = {}
    for kind in KINDS:
        r = res[kind][name]
        summary[name][kind] = {m: float(np.nanmean([x[m] for x in r])) for m in ("f1", "precision", "recall", "elem_acc")}
    for a_, b_ in (("B0", "B2"), ("B0", "B1"), ("B1", "B2")):
        d, lo, hi = paired_bootstrap([x["f1"] for x in res[a_][name]], [x["f1"] for x in res[b_][name]])
        summary[name][f"{b_}-{a_}"] = {"mean": d, "ci95": [lo, hi]}
s1 = all(summary[n]["B2-B0"]["ci95"][0] > 0 for n in ("T1", "T2"))
summary["S1_pass"] = s1
(out_dir / "t1_t2_summary.json").write_text(json.dumps(summary, indent=1))
(out_dir / "t1_t2_per_sample_f1.json").write_text(
    json.dumps({k: {n: [x["f1"] for x in res[k][n]] for n in ("T1", "T2")} for k in KINDS}))
print(json.dumps(summary, indent=1))
