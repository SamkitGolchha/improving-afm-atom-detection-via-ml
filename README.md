# Improving AFM Atom Detection via ML

Neural networks that read CO-tip atomic force microscopy (AFM) image stacks and locate each atom, its element and its height. All training data is simulated: 1,755 open organic molecules run through a GPU AFM simulator on a single laptop. The models are tested on unseen molecules, multi-molecule scenes, images from a different simulator, and real experimental AFM scans of water on metal surfaces.

## Results (pilot run)

An atom counts as found if predicted within 0.5 Å of its true position.

| Test | Atoms found | Predictions that are real atoms | Element correct | Prediction–truth correlation (r) |
|---|---|---|---|---|
| Unseen molecules (350 images) | 88–92% | 95–98% | up to 96% | 0.91–0.97 (chance 0.17–0.19) |
| Multi-molecule scenes (150) | 90–91% | 95–98% | up to 94% | 0.93–0.96 (chance ≈ 0.01) |
| Images from a different simulator (175) | F1 78–81% | — | — | 0.76–0.80 (chance ≈ 0.21) |
| Experimental water cluster on Cu(111) | 100% within 1 Å | — | — | — |

Ranges span the three model variants. Pilot scale: one Apple M5 Pro GPU, about 45 minutes of training per model, single seed.

## Models

| Model | Description | Weights |
|---|---|---|
| B0 | Whole-image U-Net | `results/pilot/models/B0.pt` |
| B1 | Patch (48 px) → centre (16 px) U-Net, stitched by overlap averaging | `results/pilot/models/B1.pt` |
| B2 | B1 + FiLM conditioning on oscillation amplitude and CO-tip stiffness | `results/pilot/models/B2.pt` |
| B1 (ice) | B1 trained on simulated ice clusters on Au(111) and Cu(111) | `results/pilot/models/B1_ice.pt` |

Detection thresholds tuned on validation data are in `results/pilot/models/thresholds.json`.

## Layout

```
src/afmport/   simulator wrapper, target rendering, models, inference/stitching, metrics, datasets
scripts/       data download and generation, training, evaluation (eval_*.py), utilities
results/pilot/models/   trained weights and training logs
```

## Setup

```bash
python3 -m venv ~/.venvs/afm-spinoff
~/.venvs/afm-spinoff/bin/pip install -r requirements.txt
```

On macOS, ppafm's C++ helpers need OpenMP. Build them with `scripts/build_ppafm_macos.sh`, which uses an existing libomp (Anaconda's by default). Run scripts with `PYTHONPATH=src`.

## Pipeline

1. **Data.** Download the datasets below into `data/raw/` (`scripts/download_trackb.sh` fetches the ice data), then run `scripts/make_split.py` (scaffold split) and `scripts/gen_track_a.py` (simulated training stacks).
2. **Train.** `scripts/train.py B0|B1|B2 <minutes>`.
3. **Evaluate.** `scripts/eval_t1_t2.py`, `eval_s2_tolerance.py`, `eval_s3_distortion.py`, `eval_s4_shift.py`, `eval_correlation.py`.
4. **Experimental ice.** `scripts/gen_track_b.py`, then training, then `scripts/eval_s5_ice.py`.

Run heavy jobs one at a time; `scripts/mem_watchdog.sh` guards against running out of memory.

## Data (not included)

- QUAM-AFM Lite, DOI [10.21950/BFAU11](https://doi.org/10.21950/BFAU11) (CC BY-NC-SA 4.0)
- Experimental ice AFM data, Zenodo [10054847](https://doi.org/10.5281/zenodo.10054847) (CC BY 4.0)
- Relaxed ice structures, Zenodo [10362511](https://doi.org/10.5281/zenodo.10362511) (CC BY 4.0)
- AFM-ice simulations, Zenodo [10049832](https://doi.org/10.5281/zenodo.10049832), [10049856](https://doi.org/10.5281/zenodo.10049856), [10047850](https://doi.org/10.5281/zenodo.10047850) (CC BY 4.0)

## References

- Local patch-to-centre reconstruction approach adapted from K. Loehr et al., PtyRANNOSAUR, arXiv:2606.27587 (2026).
- Simulator: [ppafm](https://github.com/Probe-Particle/ppafm) (MIT). Experimental-ice baseline: [ml-spm](https://github.com/SINGROUP/ml-spm) (MIT); F. Priante et al., ACS Nano (2024).

## Licence

Code: MIT (`LICENSE`). Model weights: see `WEIGHTS_LICENSE.md`. The B0, B1 and B2 weights are CC BY-NC-SA 4.0 because they were trained on QUAM-AFM data.
