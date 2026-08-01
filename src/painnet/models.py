"""Keras model builders. TF/Keras throughout -- no PyTorch anywhere in this course.

Four architectures, roughly in order of ambition:
    build_cnn1d   -- the baseline. local waveform structure, nothing clever.
    build_cnn_lstm-- conv features handed to a recurrent layer
    build_tcn     -- dilated causal convs; long receptive field, no recurrence
    build_fusion  -- the centrepiece. two branches (EEG + wristband), late fusion.

Every builder calls set_random_seed(SEED) first so comparisons between models
are fair and reruns match the numbers written in the markdown.

Capacity is kept deliberately small. With 83 subjects, a big model just
memorises people.
"""
from __future__ import annotations

import keras
from keras import layers

from . import config


def _seed():
    """Call before building anything. Makes runs reproducible on CPU."""
    keras.utils.set_random_seed(config.SEED)


def build_cnn1d(input_shape, n_classes: int = len(config.PAIN_CLASSES)) -> keras.Model:
    """Plain 1D convnet. Our baseline -- everything else has to beat this."""
    _seed()
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            layers.Dense(32, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn1d",
    )


def build_cnn_lstm(input_shape, n_classes: int = len(config.PAIN_CLASSES)) -> keras.Model:
    """Conv front-end for local shape, LSTM for the longer-range stuff."""
    _seed()
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 5, padding="same", activation="relu"),
            layers.MaxPooling1D(2),
            layers.LSTM(64),
            layers.Dropout(0.4),
            layers.Dense(32, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn_lstm",
    )


def build_tcn(input_shape, n_classes: int = len(config.PAIN_CLASSES)) -> keras.Model:
    """Dilated causal convs. Same receptive field as the LSTM, trains faster.

    Dilation 1,2,4,8 over kernel 5 covers ~61 timesteps, which is basically our
    whole 60s window -- so the last position sees everything.
    """
    _seed()
    inp = keras.Input(shape=input_shape)
    x = inp
    for d in (1, 2, 4, 8):
        skip = x
        x = layers.Conv1D(64, 5, dilation_rate=d, padding="causal", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        # residual connection -- needs a 1x1 conv when channel counts differ
        if skip.shape[-1] != x.shape[-1]:
            skip = layers.Conv1D(64, 1, padding="same")(skip)
        x = layers.Add()([x, skip])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="tcn")


def build_fusion(
    eeg_shape,
    watch_shape,
    n_classes: int = len(config.PAIN_CLASSES),
) -> keras.Model:
    """Two-branch late fusion -- the actual point of the project.

    EEG arrives at 1 Hz, the Empatica signals at 4 Hz. Rather than resample one
    to match the other (and pretend we measured something we didn't), give each
    modality its own encoder and concatenate the learned embeddings.

    This is the Keras Functional API doing the thing it exists for: you cannot
    express two inputs at different sample rates with Sequential.
    """
    _seed()

    # --- branch 1: EEG band powers @ 1 Hz ---
    eeg_in = keras.Input(shape=eeg_shape, name="eeg")
    e = layers.Conv1D(32, 5, padding="same", activation="relu")(eeg_in)
    e = layers.BatchNormalization()(e)
    e = layers.MaxPooling1D(2)(e)
    e = layers.Conv1D(64, 3, padding="same", activation="relu")(e)
    e = layers.GlobalAveragePooling1D()(e)
    e = layers.Dense(32, activation="relu", name="eeg_embedding")(e)

    # --- branch 2: Empatica BVP/EDA/TEMP/ACC @ 4 Hz ---
    w_in = keras.Input(shape=watch_shape, name="watch")
    w = layers.Conv1D(32, 7, padding="same", activation="relu")(w_in)
    w = layers.BatchNormalization()(w)
    w = layers.MaxPooling1D(4)(w)  # 4 Hz -> roughly 1 Hz, matching the EEG branch
    w = layers.Conv1D(64, 5, padding="same", activation="relu")(w)
    w = layers.GlobalAveragePooling1D()(w)
    w = layers.Dense(32, activation="relu", name="watch_embedding")(w)

    # --- fuse ---
    x = layers.Concatenate(name="fusion")([e, w])
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model([eeg_in, w_in], out, name="fusion")


BUILDERS = {
    "cnn1d": build_cnn1d,
    "cnn_lstm": build_cnn_lstm,
    "tcn": build_tcn,
}


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def default_callbacks(patience: int = 8):
    """Early stopping on val loss. Small data -> overfits fast, watch it."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
