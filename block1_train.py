# %% [BLOCK 1] TRAIN AND SAVE
"""
Trains DenseNet121 (Phase 1: head only, Phase 2: fine-tune top layers)
and saves a self-contained joblib bundle (architecture + weights + metadata).
A .keras copy is also saved as a backup.
"""
import os
import warnings
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import DenseNet121

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
TRAIN_DIR   = "training"
JOBLIB_PATH = "brain_mri_densenet.joblib"
KERAS_PATH  = "brain_mri_densenet.keras"

CLASS_NAMES = ["notumor", "glioma", "meningioma", "pituitary"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
EPOCHS_P1   = 20
EPOCHS_P2   = 30
UNFREEZE_N  = 50          # number of top backbone layers to fine-tune

tf.keras.utils.set_random_seed(SEED)


def load_train_val():
    kw = dict(labels="inferred", label_mode="int", class_names=CLASS_NAMES,
              image_size=IMG_SIZE, batch_size=BATCH_SIZE,
              validation_split=0.2, seed=SEED)
    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, subset="training", **kw)
    val_ds   = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, subset="validation", **kw)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def class_weights_from_folders():
    counts = np.array([
        len([f for f in os.listdir(os.path.join(TRAIN_DIR, c)) if not f.startswith(".")])
        for c in CLASS_NAMES
    ], dtype=float)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))          # raw pixels 0-255

    # Augmentation: Keras applies it ONLY during fit(), never at inference
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.2)(x)

    # DenseNet preprocessing (ImageNet mean/std on 0-1 scale), done with layers
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Normalization(mean=[0.485, 0.456, 0.406],
                             variance=[0.229**2, 0.224**2, 0.225**2])(x)

    backbone = DenseNet121(weights="imagenet", include_top=False,
                           input_shape=(*IMG_SIZE, 3), name="densenet121")
    backbone.trainable = False
    x = backbone(x, training=False)                      # BN stays in inference mode

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(inputs, outputs), backbone


def make_callbacks(ckpt_path):
    return [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-7, verbose=1),
    ]


def main():
    train_ds, val_ds = load_train_val()
    cw = class_weights_from_folders()
    print("Class weights:", {CLASS_NAMES[k]: round(v, 2) for k, v in cw.items()})

    model, backbone = build_model()

    # ---- Phase 1: head only ----
    print("\n=== PHASE 1: training head (frozen backbone) ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P1,
                   class_weight=cw, callbacks=make_callbacks("phase1_best.keras"))

    # ---- Phase 2: fine-tune top layers (BatchNorm layers stay frozen) ----
    print("\n=== PHASE 2: fine-tuning top DenseNet layers ===")
    backbone.trainable = True
    for layer in backbone.layers[:-UNFREEZE_N]:
        layer.trainable = False
    for layer in backbone.layers[-UNFREEZE_N:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_P2,
                   class_weight=cw, callbacks=make_callbacks("phase2_best.keras"))

    # ---- Save ----
    model.save(KERAS_PATH)
    bundle = {
        "architecture": model.to_json(),
        "weights": model.get_weights(),
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "history": {"phase1": h1.history, "phase2": h2.history},
    }
    joblib.dump(bundle, JOBLIB_PATH, compress=3)
    print(f"\nSaved: {JOBLIB_PATH}  (+ backup {KERAS_PATH})")


if __name__ == "__main__":
    main()