# Spot crossing detection

Binary classification on synthetic light curves: predict **`has_crossing`** from **`flux_noisy`** (+ gradient). SNR is not used in training.

## Quick start

```bash
git clone <your-repo-url>
cd spot-detect
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Not in this repo (too large for GitHub). Download from UNSW OneDrive:

**[spot-detect data](https://unsw-my.sharepoint.com/:f:/g/personal/z5721853_ad_unsw_edu_au/IgAY6zD57-n_Rbn3LYRnGW1nAY_Bx38mzn4ZUZuGzLf1zOQ?e=YgK9Oa)**

Copy into `dataset/`:

```text
dataset/metadata.csv
dataset/lightcurves/lc_*.csv
```

Column reference: [dataset/README.md](dataset/README.md).

```bash
test -f dataset/metadata.csv && test -d dataset/lightcurves && echo OK
```

## Workflow

### Step 1 — Baseline model

```bash
python train_model.py
```

Wide CNN + BiLSTM + attention; focal loss; validation threshold tuned to reduce false negatives.

### Step 2 — Edge pipeline (optional)

```bash
python train_edge_pipeline.py
```

Two models (`model_main`, `model_edge`) + wing-score routing for limb / U-shaped light curves.

### Step 3 — Strong-crossing label (optional)

```bash
python snr.py
```

Writes `has_strong_crossing` to `metadata.csv` only. Not used by Steps 1–2.

### Step 4 — Error analysis

```bash
python error_analysis/plot_basic.py
python error_analysis/plot_edge.py
```

See [error_analysis/README.md](error_analysis/README.md).

## Outputs

| File | Produced by | In Git? |
|------|-------------|---------|
| `best_model.keras` | `train_model.py` | no |
| `model_config.json` | `train_model.py` | no (see `config/model_config.example.json`) |
| `model_main.keras`, `model_edge.keras` | `train_edge_pipeline.py` | no |
| `edge_pipeline_config.json` | `train_edge_pipeline.py` | no (see `config/edge_pipeline_config.example.json`) |
| `error_analysis/figures/basic/` | `error_analysis/plot_basic.py` | no |
| `error_analysis/figures/edge/` | `error_analysis/plot_edge.py` | no |

## Repository layout

```text
spot-detect/
├── README.md
├── requirements.txt
├── train_model.py
├── train_edge_pipeline.py
├── snr.py
├── error_analysis/
│   ├── README.md
│   ├── plot_basic.py
│   ├── plot_edge.py
│   └── figures/              ← outputs (not in Git)
├── config/
│   ├── model_config.example.json
│   └── edge_pipeline_config.example.json
└── dataset/
    ├── README.md
    ├── metadata.csv          ← OneDrive only
    └── lightcurves/          ← OneDrive only
```

## Project notes

**Problem.** Detect whether a stellar spot crossing occurred (`has_crossing`). Input is preprocessed `flux_noisy` (median norm, ×1000, roll deepest point to centre, + gradient channel, length 200).

**Hard cases.** High `level` (noise, shallow dips) and limb spots: ingress/egress on both wings → U-shaped curves. A single model leaves many false negatives on these samples.

**Baseline (`train_model.py`).** Conv1D stack → BiLSTM → attention → sigmoid. Focal loss; higher sample weight on positive class and on high `level` / large `spot0_radius_deg`. Threshold on validation set minimizes FN with an FP cap (`threshold_mode: min_fn`).

**Edge pipeline (`train_edge_pipeline.py`).** Train `model_main` on all data and `model_edge` on an edge subset. At inference, wing activity scores route each curve; separate probability thresholds per branch. Typical test accuracy ~90–91% for `level <= 5` (20% hold-out, seed 42); remaining errors are mostly FN (often low SNR in plots).

**Labels.** Training uses `has_crossing`, not `has_strong_crossing` (SNR > 5, from `snr.py`).

## Notes

- Put **code** on GitHub; put **data and `.keras` files** on OneDrive or local disk (see `.gitignore`).
- Test split: 20% stratified, `random_state=42`; validation is 15% of train.
- Config examples in `config/` document JSON fields after a local run.
