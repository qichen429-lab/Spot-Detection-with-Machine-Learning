"""
Train baseline has_crossing classifier.

  python train_model.py

Writes: best_model.keras, model_config.json
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


def set_seeds() -> None:
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def edge_weight(row: pd.Series) -> float:
    if int(row["level"]) >= 4:
        return 1.5
    if pd.notna(row.get("spot0_radius_deg")) and float(row["spot0_radius_deg"]) >= 10.0:
        return 1.5
    return 1.0


def preprocess_lc(row: pd.Series) -> np.ndarray:
    lc = pd.read_csv(f"dataset/lightcurves/{row['name']}.csv")
    flux = lc["flux_noisy"].values.astype(np.float32)
    flux = flux / np.median(flux) - 1.0
    flux = flux * 1000.0
    center = int(np.argmin(flux))
    flux = np.roll(flux, len(flux) // 2 - center)
    grad = np.gradient(flux).astype(np.float32)
    return np.stack([flux, grad], axis=-1)


def prepare_data():
    df = pd.read_csv("dataset/metadata.csv")
    df = df[df.level <= 10].reset_index(drop=True)

    X_list, y_list, sw_list = [], [], []
    for _, row in df.iterrows():
        X_list.append(preprocess_lc(row))
        y_list.append(int(row["has_crossing"]))
        sw_list.append(edge_weight(row))

    X = pad_sequences(X_list, maxlen=200, padding="post", truncating="post", dtype="float32")
    y = np.array(y_list)
    sample_w = np.array(sw_list, dtype=np.float32)

    X_train, X_test, y_train, y_test, meta_train, meta_test, sw_train, _ = train_test_split(
        X, y, df, sample_w, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
        X_train, y_train, sw_train, test_size=0.15, random_state=42, stratify=y_train
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test, meta_test, sw_tr


def build_model(seq_len: int = 200, n_channels: int = 2) -> Model:
    x_in = Input(shape=(seq_len, n_channels))

    x = Conv1D(48, 11, padding="same", activation="relu")(x_in)
    x = BatchNormalization()(x)
    x = Conv1D(48, 9, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.1)(x)

    x = Conv1D(96, 7, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, 5, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.1)(x)

    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 5, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.15)(x)

    res = Conv1D(128, 1, padding="same")(x)
    res = BatchNormalization()(res)
    x = Conv1D(128, 5, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    x = Activation("relu")(x)

    res2 = Conv1D(256, 1, padding="same")(x)
    res2 = BatchNormalization()(res2)
    x = Conv1D(256, 5, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, res2])
    x = Activation("relu")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(x)
    attention = Dense(1)(lstm)
    attention = Flatten()(attention)
    attention = Activation("softmax")(attention)
    attention = RepeatVector(256)(attention)
    attention = Permute([2, 1])(attention)
    attended = Multiply()([lstm, attention])

    out = Concatenate()([GlobalAveragePooling1D()(attended), GlobalMaxPooling1D()(attended)])
    out = Dense(128, activation="relu")(out)
    out = Dropout(0.25)(out)
    out = Dense(64, activation="relu")(out)
    y_out = Dense(1, activation="sigmoid")(out)

    return Model(x_in, y_out)


def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        modulating = K.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return K.mean(alpha_t * modulating * bce)

    return loss


def make_sample_weights(y: np.ndarray, edge_w: np.ndarray) -> np.ndarray:
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_map = {0: float(cw[0]), 1: float(cw[1]) * 2.0}
    return np.array([class_map[int(lbl)] * edge_w[i] for i, lbl in enumerate(y)], dtype=np.float32)


def tune_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 193):
        pred = (prob_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        score = -fn if fp <= 800 else -1e9
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


if __name__ == "__main__":
    set_seeds()

    X_tr, X_val, X_test, y_tr, y_val, y_test, meta_test, sw_tr = prepare_data()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=focal_loss(),
        metrics=["accuracy"],
    )

    model.fit(
        X_tr,
        y_tr,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        sample_weight=make_sample_weights(y_tr, sw_tr),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        ],
        verbose=1,
    )

    threshold = tune_threshold(y_val, model.predict(X_val, verbose=0).flatten())
    y_pred = (model.predict(X_test, verbose=0).flatten() >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print(f"threshold={threshold:.3f}  mode=min_fn  test_acc={accuracy_score(y_test, y_pred):.2%}")
    print(f"FN={cm[1,0]}  FP={cm[0,1]}  F1={f1_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    print(cm)

    model.save("best_model.keras")
    with open("model_config.json", "w") as f:
        json.dump(
            {
                "label": "has_crossing",
                "level_max": 10,
                "threshold": threshold,
                "threshold_mode": "min_fn",
                "max_fp": 800,
                "pos_class_mult": 2.0,
                "channels": ["flux", "gradient"],
            },
            f,
            indent=2,
        )
    print("Saved best_model.keras, model_config.json")
