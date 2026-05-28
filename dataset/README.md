# Dataset

Download: **[UNSW OneDrive — spot-detect data](https://unsw-my.sharepoint.com/:f:/g/personal/z5721853_ad_unsw_edu_au/IgAY6zD57-n_Rbn3LYRnGW1nAY_Bx38mzn4ZUZuGzLf1zOQ?e=YgK9Oa)**

Install under this folder:

```text
dataset/metadata.csv
dataset/lightcurves/lc_XXXXXX.csv
```

`metadata.name` must match the light-curve filename (e.g. `lc_000046` → `lightcurves/lc_000046.csv`).

## `metadata.csv` (key columns)

| Column | Description |
|--------|-------------|
| `name` | Light-curve ID |
| `level` | Difficulty (higher = harder) |
| `has_crossing` | **Training label** (0/1) |
| `has_strong_crossing` | Optional; run `python snr.py` from repo root |
| `noise_ppm` | Noise level (used by `snr.py`) |
| `spot0_radius_deg` | Used for sample weights / edge flags in training scripts |

Other columns (`inc_star`, `P_rot`, spot geometry, crossing times, etc.) are documented in the CSV header.

## `lightcurves/lc_*.csv`

| Column | Description |
|--------|-------------|
| `time_days` | Time |
| `flux_noisy` | Noisy flux — **model input** |
| `flux_clean` | Clean flux (`snr.py`, plots) |
| `mask` | 1 during crossing segments |

## Notes

- Files here are **not** committed to Git.
- After copy, run `python train_model.py` from the **repository root**.
