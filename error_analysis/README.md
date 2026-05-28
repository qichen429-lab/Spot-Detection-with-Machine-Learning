# Error analysis

Plot false negatives (FN) and false positives (FP) on the held-out test split (20%, seed 42).

Run from the **repository root**:

```bash
python error_analysis/plot_basic.py
python error_analysis/plot_edge.py
```

## Requirements

| Script | Needs (repo root) |
|--------|-------------------|
| `plot_basic.py` | `best_model.keras`, optional `model_config.json` |
| `plot_edge.py` | `model_main.keras`, `model_edge.keras`, `edge_pipeline_config.json` |

## Outputs (local, not in Git)

```text
error_analysis/figures/basic/   fn.png, fp.png, fn_top6.csv, fp_top6.csv
error_analysis/figures/edge/    fn.png, fp.png, fn_top6.csv, fp_top6.csv
```

Each plot: 6 FN (lowest predicted probability) + 6 FP (highest probability).
