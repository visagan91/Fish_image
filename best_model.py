# best_model.py
import os, json, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_zoo import build_scratch_cnn, build_transfer_model

DATA_DIR, OUT_DIR = "data", "models"
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# No rescale here (transfer models handle preprocess; scratch CNN has its own Rescaling layer)
val_gen = ImageDataGenerator().flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    color_mode="rgb",
)
n_classes = val_gen.num_classes
input_shape = (*IMG_SIZE, 3)

# Find candidate checkpoints (support both weights-only and older full .h5 names)
files = os.listdir(OUT_DIR)
candidates = [
    f for f in files
    if f.endswith("_finetuned_best.weights.h5") or f.endswith("_finetuned_best.h5")
]

# If nothing found, try evaluating the already-exported modern model
if not candidates:
    best_keras = os.path.join(OUT_DIR, "best_model.keras")
    if os.path.exists(best_keras):
        model = tf.keras.models.load_model(best_keras)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        loss, acc = model.evaluate(val_gen, verbose=0)
        print(f"best_model.keras: val_acc={acc:.4f}")
        # also print meta if present
        meta_path = os.path.join(OUT_DIR, "best_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                print("meta:", json.load(f))
    else:
        print("No candidate checkpoints found and best_model.keras is missing.")
    raise SystemExit(0)

scores = []
for f in candidates:
    name = f.replace("_finetuned_best.weights.h5", "").replace("_finetuned_best.h5", "")
    if name == "scratch_cnn":
        model = build_scratch_cnn(input_shape, n_classes)
    else:
        model, _ = build_transfer_model(name, input_shape, n_classes, train_base=False, dropout=0.3)

    # IMPORTANT: compile before evaluate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.load_weights(os.path.join(OUT_DIR, f))
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"{name}: val_acc={acc:.4f}")
    scores.append((float(acc), name, f))

scores.sort(reverse=True)
best_acc, best_name, best_file = scores[0]
print("Best:", best_name, best_acc)

# Rebuild the best architecture cleanly and save once in modern format
if best_name == "scratch_cnn":
    best_model = build_scratch_cnn(input_shape, n_classes)
else:
    best_model, _ = build_transfer_model(best_name, input_shape, n_classes, train_base=False, dropout=0.3)

best_model.load_weights(os.path.join(OUT_DIR, best_file))
best_model_path = os.path.join(OUT_DIR, "best_model.keras")
best_model.save(best_model_path)

with open(os.path.join(OUT_DIR, "best_model_meta.json"), "w") as f:
    json.dump({"name": best_name, "val_accuracy": float(best_acc), "path": "models/best_model.keras"}, f, indent=2)

print(f"Saved: {best_model_path}")
