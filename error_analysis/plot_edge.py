#!/usr/bin/env python3
"""
Plot 6 FN + 6 FP for the edge pipeline.

  python error_analysis/plot_edge.py
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
OUT = Path(__file__).resolve().parent / "figures" / "edge"
sys.path.insert(0, str(ROOT))

from train_edge_pipeline import combined_predict, preprocess_lc, route_by_wings  # noqa: E402


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
    for name in ("model_main.keras", "model_edge.keras", "edge_pipeline_config.json"):
        if not (ROOT / name).exists():
            raise FileNotFoundError(f"missing {name} — run: python train_edge_pipeline.py")

    cfg = json.loads((ROOT / "edge_pipeline_config.json").read_text())

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
    prob_main = tf.keras.models.load_model(ROOT / "model_main.keras", compile=False).predict(X_test, verbose=0).flatten()
    prob_edge = tf.keras.models.load_model(ROOT / "model_edge.keras", compile=False).predict(X_test, verbose=0).flatten()
    wing = route_by_wings(X_test)
    meta = meta.reset_index(drop=True).assign(
        y_true=y_test,
        y_pred=combined_predict(prob_main, prob_edge, wing, cfg["thr_main"], cfg["thr_edge"], cfg["thr_route"]),
        prob=np.where(wing >= cfg["thr_route"], prob_edge, prob_main),
    )

    fn = meta[(meta.y_true == 1) & (meta.y_pred == 0)].sort_values("prob")
    fp = meta[(meta.y_true == 0) & (meta.y_pred == 1)].sort_values("prob", ascending=False)

    cm = confusion_matrix(meta.y_true, meta.y_pred)
    print(
        f"\n[edge pipeline] level<={int(cfg.get('level_max', 5))}  n={len(meta)}  "
        f"acc={(meta.y_true == meta.y_pred).mean():.2%}"
    )
    print(f"FN={cm[1, 0]}  FP={cm[0, 1]}  TP={cm[1, 1]}  TN={cm[0, 0]}")
    print(f"routed to edge: {(wing >= cfg['thr_route']).sum()} / {len(meta)}")
    print(classification_report(meta.y_true, meta.y_pred, digits=3))

    OUT.mkdir(parents=True, exist_ok=True)
    fn.head(6).to_csv(OUT / "fn_top6.csv", index=False)
    fp.head(6).to_csv(OUT / "fp_top6.csv", index=False)
    plot_six(fn.head(6), f"FN — edge (level<={int(cfg.get('level_max', 5))})", "fn.png")
    plot_six(fp.head(6), f"FP — edge (level<={int(cfg.get('level_max', 5))})", "fp.png")
    print(f"\ndone → {OUT}/")
