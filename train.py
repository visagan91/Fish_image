# train.py
# Multi-backbone training script:
# - Scratch CNN
# - VGG16 / ResNet50 / MobileNet / InceptionV3 / EfficientNetB0 (via model_zoo)
# Saves per-model best weights, then rebuilds the best model and exports once to .keras

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_zoo import build_scratch_cnn, build_transfer_model, BACKBONES

# ========================
# Config
# ========================
DATA_DIR   = "data"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
OUT_DIR    = "models"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16          # set to 8 for faster runs on low memory
EPOCHS_STAGE1   = 5           # set to 3 for a quicker first pass
EPOCHS_STAGE2   = 5           # set to 2 for a quicker fine-tune

# ========================
# Data generators
# ========================
def make_gens(img_size=IMG_SIZE):
    # For transfer learning we apply the backbone's preprocess inside the model,
    # so we do NOT rescale here. (Scratch CNN has its own Rescaling layer.)
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        color_mode="rgb",
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb",
    )
    return train_gen, val_gen

# ========================
# Train utility
# ========================
def compile_fit(model, train_gen, val_gen, epochs, out_path):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        ModelCheckpoint(
            out_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,   # weights-only -> smaller & avoids legacy JSON issues
        ),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cbs)
    return hist

# ========================
# Main pipeline
# ========================
def run_all():
    train_gen, val_gen = make_gens()
    n_classes = train_gen.num_classes
    input_shape = (*IMG_SIZE, 3)

    # Save label map (index -> class_name)
    with open(os.path.join(OUT_DIR, "class_indices.json"), "w") as f:
        json.dump({v: k for k, v in train_gen.class_indices.items()}, f, indent=2)

    histories = {}
    best_models = []

    # ---------- 1) Train a small scratch CNN ----------
    scratch = build_scratch_cnn(input_shape, n_classes)
    scratch_ckpt = os.path.join(OUT_DIR, "scratch_cnn_best.weights.h5")
    hist = compile_fit(scratch, train_gen, val_gen, EPOCHS_STAGE1 + EPOCHS_STAGE2, scratch_ckpt)
    best_acc = float(max(hist.history["val_accuracy"]))
    histories["scratch_cnn"] = hist.history
    best_models.append(("scratch_cnn", scratch_ckpt, best_acc))

    # ---------- 2) Transfer learning backbones ----------
    for name in BACKBONES.keys():
        # Stage 1: train head with frozen base
        model, _ = build_transfer_model(name, input_shape, n_classes, train_base=False, dropout=0.3)
        stage1_ckpt = os.path.join(OUT_DIR, f"{name}_stage1_best.weights.h5")
        hist1 = compile_fit(model, train_gen, val_gen, EPOCHS_STAGE1, stage1_ckpt)

        # Stage 2: fine-tune top ~20% of layers
        n_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            if i > int(0.8 * n_layers):
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        stage2_ckpt = os.path.join(OUT_DIR, f"{name}_finetuned_best.weights.h5")
        hist2 = compile_fit(model, train_gen, val_gen, EPOCHS_STAGE2, stage2_ckpt)

        histories[name] = {"stage1": hist1.history, "stage2": hist2.history}
        best_acc = float(max(hist2.history["val_accuracy"]))
        best_models.append((name, stage2_ckpt, best_acc))

    # ---------- Pick global best and export once ----------
    best_models.sort(key=lambda x: x[2], reverse=True)
    best_name, best_path, best_acc = best_models[0]

    # Rebuild the same architecture and load weights
    if best_name == "scratch_cnn":
        best_model = build_scratch_cnn(input_shape, n_classes)
    else:
        best_model, _ = build_transfer_model(best_name, input_shape, n_classes, train_base=False, dropout=0.3)

    best_model.load_weights(best_path)

    # Save the winner in modern Keras format
    final_keras_path = os.path.join(OUT_DIR, "best_model.keras")
    best_model.save(final_keras_path)

    with open(os.path.join(OUT_DIR, "best_model_meta.json"), "w") as f:
        json.dump(
            {"name": best_name, "val_accuracy": best_acc, "path": "models/best_model.keras"},
            f,
            indent=2,
        )

    # Save all training curves for later inspection
    with open(os.path.join(OUT_DIR, "histories.json"), "w") as f:
        json.dump(histories, f)

    print(f"\nBest model: {best_name} (val_acc={best_acc:.4f})")
    print(f"Exported:   {final_keras_path}")

# ========================
# Entry point
# ========================
if __name__ == "__main__":
    run_all()
