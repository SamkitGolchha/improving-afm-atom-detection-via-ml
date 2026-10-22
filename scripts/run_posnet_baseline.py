"""Authors' baseline (S5 comparator): ml-spm GraphImgNetIce PosNet + graph on the 9 experimental clusters.

Uses the authors' preprocessing (interpolate_and_crop to 0.125 A, add_norm), match thresholds
(cu111 0.5, au111-monolayer 0.5, au111-bilayer 0.6) and per-cluster model choice. Weights are our
MD5-verified local copies, loaded with weights_only=True (the package itself re-downloads and
unpickles with weights_only=False). Positions are mapped back to the experiment frame (A from
scan start) by undoing the crop. Runs on CPU.
One cluster per process (memory safety after the 2026-10-16 crash); results merged into one JSON.
Usage: PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/run_posnet_baseline.py <label>
"""
import json
import sys

import numpy as np
import torch

torch.set_num_threads(4)
import mlspm.preprocessing as pp
from mlspm import graph
from mlspm.models import GraphImgNetIce

from afmport.common import RAW, RESULTS
from afmport.ice import LABELS, REG, load_exp

Z_RANGES = {"cu111": (-2.9, 0.5), "au111-monolayer": (-2.9, 0.5), "au111-bilayer": (-3.5, 0.5)}
THRESH = {"cu111": 0.5, "au111-monolayer": 0.5, "au111-bilayer": 0.6}
WDIR = RAW / "ice_weights_zenodo10054348"
out = RESULTS / "pilot" / "track_b"
out.mkdir(parents=True, exist_ok=True)

def main():
    models = {}
    res_path = out / "posnet_predictions.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for label in [sys.argv[1]]:
        exp_name, _, _, wname = REG[label]
        if wname not in models:
            m = GraphImgNetIce(pretrained_weights=None, grid_z_range=Z_RANGES[wname], device="cpu")
            state = torch.load(WDIR / f"weights_ice-{wname}.pth", map_location="cpu", weights_only=True)
            m.load_state_dict(state)
            models[wname] = m.eval()
        model = models[wname]
        data, lx, ly = load_exp(label)
        X = data[None]
        X = pp.interpolate_and_crop([X], (lx, ly))
        pp.add_norm(X)
        X = X[0]
        # Crop offsets used inside interpolate_and_crop (pixels at 0.125 A).
        nx, ny = int(lx // 0.125), int(ly // 0.125)
        x_start, y_start = int(np.floor((nx % 8) / 2)), int(np.floor((ny % 8) / 2))
        with torch.no_grad():
            bb = graph.make_box_borders(X.shape[1:3], (model.afm_res, model.afm_res), z_range=model.posnet.grid_z_range)
            xt = torch.from_numpy(X).float()
            grid, _ = model.posnet(xt.unsqueeze(1))
            pos, matches, labels = graph.find_gaussian_peaks(grid, bb, match_threshold=THRESH[wname],
                                                             std=model.posnet.peak_std, method=model.posnet.match_method)
            pred_graph, _ = model.predict_graph(X, pos=pos)
        xyz = pred_graph[0].array(xyz=True)
        cls = pred_graph[0].array(class_index=True)[:, 0]  # 0 = H, 1 = O
        # Back to experiment frame. PIL resize maps the full extent L onto n' = L // 0.125 pixels.
        sx, sy = lx / nx, ly / ny
        xyz_exp = xyz.copy()
        xyz_exp[:, 0] = (xyz[:, 0] / 0.125 + x_start) * sx
        xyz_exp[:, 1] = (xyz[:, 1] / 0.125 + y_start) * sy
        res[label] = {"model": wname, "xyz": xyz_exp.tolist(), "elem": [8 if c == 1 else 1 for c in cls]}
        print(label, exp_name, wname, f"pred atoms {len(xyz)} (O {int((cls == 1).sum())}, H {int((cls == 0).sum())})", flush=True)
    res_path.write_text(json.dumps(res))
    print("wrote", res_path)


if __name__ == "__main__":
    main()
