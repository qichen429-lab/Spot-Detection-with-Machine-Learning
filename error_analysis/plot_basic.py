#!/usr/bin/env python3
"""
Plot 6 FN + 6 FP for the baseline model.

  python error_analysis/plot_basic.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures" / "basic"
sys.path.insert(0, str(ROOT))

from train_model import preprocess_lc  # noqa: E402


def scalar_snr(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, list):
        return float(max(val)) if val else np.nan
    if isinstance(val, str) and val.strip():
        s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", val.strip())
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)) and v:
                return float(max(v))
            if isinstance(v, (int, float)):
                return float(v)
        except (SyntaxError, ValueError):
            pass
    return np.nan


def plot_panel(ax, row: pd.Series, prob: float) -> None:
    lc = pd.read_csv(ROOT / "dataset/lightcurves" / f"{row['name']}.csv")
    ax.plot(lc["time_days"].values, lc["flux_noisy"].values, "k-", lw=0.9)
    if "mask" in lc.columns and lc["mask"].values.any():
        raw = lc["flux_noisy"].values
        ax.fill_between(
            lc["time_days"].values,
            raw.min(),
            raw.max(),
            where=lc["mask"].values.astype(bool),
            color="orange",
            alpha=0.3,
        )
    snr = scalar_snr(row.get("approx_snr", np.nan))
    ax.set_title(
        f"{row['name']}\nL{row['level']}  crossing={int(row['has_crossing'])}  "
        f"{'SNR=' + format(snr, '.1f') if not np.isnan(snr) else 'SNR=n/a'}  p={prob:.2f}",
        fontsize=8,
    )
    ax.set_xlabel("time (days)", fontsize=7)
    ax.set_ylabel("flux_noisy", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def plot_six(examples: pd.DataFrame, title: str, png_name: str) -> None:
    if examples.empty:
        print(f"  {title}: no samples, skip")
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(title, fontsize=11)
    for i, ax in enumerate(axes.flatten()):
        if i >= min(6, len(examples)):
            ax.axis("off")
            continue
        plot_panel(ax, examples.iloc[i], float(examples.iloc[i]["prob"]))
    out = OUT / png_name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    model_path = ROOT / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError("missing best_model.keras — run: python train_model.py")

    cfg_path = ROOT / "model_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    df = pd.read_csv(ROOT / "dataset/metadata.csv")
    df = df[df.level <= int(cfg.get("level_max", 5))].reset_index(drop=True)

    X = pad_sequences(
        [preprocess_lc(r) for _, r in df.iterrows()],
        maxlen=200,
        padding="post",
        truncating="post",
        dtype="float32",
    )
    y = np.array([int(r["has_crossing"]) for _, r in df.iterrows()])

    _, X_test, _, y_test, _, meta = train_test_split(X, y, df, test_size=0.2, random_state=42, stratify=y)
    prob = tf.keras.models.load_model(model_path, compile=False).predict(X_test, verbose=0).flatten()
    meta = meta.reset_index(drop=True).assign(
        y_true=y_test,
        y_pred=(prob >= float(cfg.get("threshold", 0.5))).astype(int),
        prob=prob,
    )

    fn = meta[(meta.y_true == 1) & (meta.y_pred == 0)].sort_values("prob")
    fp = meta[(meta.y_true == 0) & (meta.y_pred == 1)].sort_values("prob", ascending=False)

    cm = confusion_matrix(meta.y_true, meta.y_pred)
    print(
        f"\n[baseline] level<={int(cfg.get('level_max', 5))}  n={len(meta)}  "
        f"acc={(meta.y_true == meta.y_pred).mean():.2%}  thr={float(cfg.get('threshold', 0.5)):.3f}"
    )
    print(f"FN={cm[1, 0]}  FP={cm[0, 1]}  TP={cm[1, 1]}  TN={cm[0, 0]}")
    print(classification_report(meta.y_true, meta.y_pred, digits=3))

    OUT.mkdir(parents=True, exist_ok=True)
    fn.head(6).to_csv(OUT / "fn_top6.csv", index=False)
    fp.head(6).to_csv(OUT / "fp_top6.csv", index=False)
    plot_six(fn.head(6), f"FN — basic (level<={int(cfg.get('level_max', 5))})", "fn.png")
    plot_six(fp.head(6), f"FP — basic (level<={int(cfg.get('level_max', 5))})", "fp.png")
    print(f"\ndone → {OUT}/")
