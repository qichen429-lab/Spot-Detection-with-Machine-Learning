"""
Train edge pipeline: model_main + model_edge, wing-score routing.

  python train_edge_pipeline.py

Writes: model_main.keras, model_edge.keras, edge_pipeline_config.json
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    MaxPooling1D,
    Multiply,
    Permute,
    RepeatVector,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

from train_model import build_model, focal_loss, preprocess_lc, set_seeds


def is_edge_meta(row: pd.Series) -> bool:
    if int(row["level"]) >= 4:
        return True
    spot0_radius_deg = row.get("spot0_radius_deg")
    return pd.notna(spot0_radius_deg) and float(spot0_radius_deg) >= 10.0


def wing_activity(flux: np.ndarray) -> tuple[float, float]:
    n = len(flux)
    q = max(8, n // 4)
    mid = flux[q : 3 * q] if 3 * q <= n else flux[n // 3 : 2 * n // 3]
    base = float(np.median(mid))
    wing_percentile = 90
    left = float(np.percentile(np.abs(flux[:q] - base), wing_percentile))
    right = float(np.percentile(np.abs(flux[-q:] - base), wing_percentile))
    return left, right


def is_edge_wings(flux: np.ndarray) -> bool:
    """Both wings show spot signal (U-shaped ingress/egress)."""
    left, right = wing_activity(flux)
    wing_both_sides_minimum = 3.5
    return left >= wing_both_sides_minimum and right >= wing_both_sides_minimum


def is_edge_sample(row: pd.Series, x: np.ndarray) -> bool:
    flux = x[:, 0] if x.ndim > 1 else x
    left, right = wing_activity(flux)
    if is_edge_wings(flux):
        return True
    wing_signal_threshold = 4.0
    if is_edge_meta(row) and max(left, right) >= wing_signal_threshold:
        return True
    return False


def route_by_wings(x: np.ndarray) -> np.ndarray:
    """Wing activity scores for routing at inference (no metadata)."""
    scores = np.array([max(wing_activity(xi[:, 0])) for xi in x], dtype=np.float32)
    return scores


def load_dataset(level_max: int = 5):
    df = pd.read_csv("dataset/metadata.csv")
    df = df[df.level <= level_max].reset_index(drop=True)

    X_list, y_list, edge_list = [], [], []
    for _, row in df.iterrows():
        x = preprocess_lc(row)
        X_list.append(x)
        y_list.append(int(row["has_crossing"]))
        edge_list.append(int(is_edge_sample(row, x)))

    sequence_length = 200
    X = pad_sequences(X_list, maxlen=sequence_length, padding="post", truncating="post", dtype="float32")
    y = np.array(y_list)
    edge = np.array(edge_list)
    return X, y, edge, df


def build_router(seq_len: int = 200, n_channels: int = 2) -> Model:
    x_in = Input(shape=(seq_len, n_channels))
    x = Conv1D(32, 11, padding="same", activation="relu")(x_in)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(64, 7, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    y_out = Dense(1, activation="sigmoid")(x)
    return Model(x_in, y_out, name="edge_router")


def pos_sample_weights(y: np.ndarray, positive_class_weight_multiplier: float = 2.0) -> np.ndarray:
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
    m = {0: float(cw[0]), 1: float(cw[1]) * positive_class_weight_multiplier}
    return np.array([m[int(l)] for l in y], dtype=np.float32)


def tune_threshold(y_val: np.ndarray, prob_val: np.ndarray, mode: str = "f1") -> float:
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 193):
        pred = (prob_val >= t).astype(int)
        score = f1_score(y_val, pred, zero_division=0) if mode == "f1" else accuracy_score(y_val, pred)
        if score > best:
            best, best_t = score, float(t)
    return best_t


def train_classifier(
    model: Model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    *,
    epochs: int = 40,
    positive_class_weight_multiplier: float = 2.0,
) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss=focal_loss(),
        metrics=["accuracy"],
    )
    sw = pos_sample_weights(y_tr, positive_class_weight_multiplier)
    model.fit(
        X_tr,
        y_tr,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        sample_weight=sw,
        callbacks=[
            EarlyStopping("val_loss", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ],
        verbose=1,
    )
    return model


def combined_predict(
    prob_main: np.ndarray,
    prob_edge: np.ndarray,
    route_score: np.ndarray,
    thr_main: float,
    thr_edge: float,
    thr_route: float,
) -> np.ndarray:
    use_edge = route_score >= thr_route
    prob = np.where(use_edge, prob_edge, prob_main)
    thr = np.where(use_edge, thr_edge, thr_main)
    return (prob >= thr).astype(int)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, edge_mask: np.ndarray | None = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"acc={accuracy_score(y_true, y_pred):.2%}  F1={f1_score(y_true, y_pred):.3f}")
    print(f"FN={cm[1,0]}  FP={cm[0,1]}")
    print(cm)
    if edge_mask is not None and edge_mask.any():
        ye, pe = y_true[edge_mask], y_pred[edge_mask]
        cm_e = confusion_matrix(ye, pe)
        print(f"  [edge subset n={edge_mask.sum()}] acc={accuracy_score(ye, pe):.2%} FN={cm_e[1,0]} FP={cm_e[0,1]}")


level_max = 5

if __name__ == "__main__":
    set_seeds()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    X, y, edge_flag, df = load_dataset(level_max)
    print(f"level<={level_max}  n={len(y)}  edge={edge_flag.sum()} ({edge_flag.mean():.1%})  pos={y.sum()}")

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15, random_state=42, stratify=y[tr_idx])

    splits = {}
    for name, ids in [("tr", tr_idx), ("va", va_idx), ("te", te_idx)]:
        splits[name] = {
            "X": X[ids],
            "y": y[ids],
            "edge": edge_flag[ids],
            "meta": df.iloc[ids].reset_index(drop=True),
        }

    # --- 1. Router (wings or small CNN) ---
    print("\n--- train edge_router ---")
    use_wing_router = len(np.unique(edge_flag)) < 2 or edge_flag.mean() > 0.92 or edge_flag.mean() < 0.08
    if use_wing_router:
        print("  edge labels nearly all one class → wing-score routing")
        wing_va = route_by_wings(splits["va"]["X"])
        thr_route = float(np.percentile(wing_va, 55))
        router = None
    else:
        router = build_router()
        router.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        rw = class_weight.compute_class_weight(
            "balanced", classes=np.unique(splits["tr"]["edge"]), y=splits["tr"]["edge"]
        )
        rw_map = {int(c): float(w) for c, w in zip(np.unique(splits["tr"]["edge"]), rw)}
        router.fit(
            splits["tr"]["X"],
            splits["tr"]["edge"],
            epochs=25,
            batch_size=32,
            validation_data=(splits["va"]["X"], splits["va"]["edge"]),
            class_weight=rw_map,
            callbacks=[EarlyStopping("val_loss", patience=4, restore_best_weights=True, verbose=1)],
            verbose=1,
        )
        wing_va = router.predict(splits["va"]["X"], verbose=0).flatten()
        thr_route = tune_threshold(splits["va"]["edge"], wing_va, mode="f1")

    # --- 2. Edge crossing model ---
    print("\n--- train model_edge (edge subset only) ---")
    e_tr = splits["tr"]["edge"].astype(bool)
    e_va = splits["va"]["edge"].astype(bool)
    model_edge = build_model()
    if e_tr.sum() < 64:
        raise RuntimeError("too few edge training samples")
    train_classifier(
        model_edge,
        splits["tr"]["X"][e_tr],
        splits["tr"]["y"][e_tr],
        splits["va"]["X"][e_va],
        splits["va"]["y"][e_va],
        epochs=45,
        positive_class_weight_multiplier=2.5,
    )
    thr_edge = tune_threshold(
        splits["va"]["y"][e_va],
        model_edge.predict(splits["va"]["X"][e_va], verbose=0).flatten(),
    )

    # --- 3. Main model (full data) ---
    print("\n--- train model_main (full) ---")
    model_main = build_model()
    train_classifier(
        model_main,
        splits["tr"]["X"],
        splits["tr"]["y"],
        splits["va"]["X"],
        splits["va"]["y"],
        epochs=45,
        positive_class_weight_multiplier=2.0,
    )
    thr_main = tune_threshold(
        splits["va"]["y"],
        model_main.predict(splits["va"]["X"], verbose=0).flatten(),
    )

    # --- 4. Test: routed merge ---
    Xte, yte, ete = splits["te"]["X"], splits["te"]["y"], splits["te"]["edge"].astype(bool)
    p_main = model_main.predict(Xte, verbose=0).flatten()
    p_edge = model_edge.predict(Xte, verbose=0).flatten()
    p_route = route_by_wings(Xte) if router is None else router.predict(Xte, verbose=0).flatten()

    pred_main = (p_main >= thr_main).astype(int)
    pred_edge_only = (p_edge >= thr_edge).astype(int)
    pred_combined = combined_predict(p_main, p_edge, p_route, thr_main, thr_edge, thr_route)

    report("model_main only", yte, pred_main, ete)
    report("model_edge only (all test)", yte, pred_edge_only, ete)
    report("combined (router → edge/main)", yte, pred_combined, ete)
    print(f"\nrouted to edge: {(p_route >= thr_route).sum()} / {len(yte)}  (thr_route={thr_route:.3f})")
    print(classification_report(yte, pred_combined))

    if router is not None:
        router.save("edge_router.keras")
    model_edge.save("model_edge.keras")
    model_main.save("model_main.keras")

    cfg = {
        "label": "has_crossing",
        "level_max": level_max,
        "thr_main": thr_main,
        "thr_edge": thr_edge,
        "thr_route": thr_route,
        "route_mode": "wings" if router is None else "router",
        "edge_radius_deg": 10.0,
        "edge_level": 4,
        "wing_thresh": 4.0,
    }
    with open("edge_pipeline_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    saved = ["model_edge.keras", "model_main.keras", "edge_pipeline_config.json"]
    if router is not None:
        saved.insert(0, "edge_router.keras")
    print("\nSaved " + ", ".join(saved))
