import os, json, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_zoo import build_scratch_cnn, build_transfer_model

DATA_DIR, OUT_DIR = "data", "models"
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224, 224)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=16,
    class_mode="categorical", shuffle=False, color_mode="rgb"
)
n_classes = val_gen.num_classes
input_shape = (*IMG_SIZE, 3)

candidates = [f for f in os.listdir(OUT_DIR) if f.endswith('_finetuned_best.h5')]
scores = []
for f in candidates:
    name = f.replace('_finetuned_best.h5','')
    if name == "scratch_cnn":
        model = build_scratch_cnn(input_shape, n_classes)
    else:
        model, _ = build_transfer_model(name, input_shape, n_classes, train_base=False, dropout=0.3)
    model.load_weights(os.path.join(OUT_DIR, f))
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"{name}: val_acc={acc:.4f}")
    scores.append((acc, name, f))

scores.sort(reverse=True)
best_acc, best_name, best_file = scores[0]
print("Best:", best_name, best_acc)

# rebuild best & save once in modern Keras format
if best_name == "scratch_cnn":
    best_model = build_scratch_cnn(input_shape, n_classes)
else:
    best_model, _ = build_transfer_model(best_name, input_shape, n_classes, train_base=False, dropout=0.3)

best_model.load_weights(os.path.join(OUT_DIR, best_file))
best_model.save(os.path.join(OUT_DIR, "best_model.keras"))
with open(os.path.join(OUT_DIR, "best_model_meta.json"), "w") as f:
    json.dump({"name": best_name, "val_accuracy": float(best_acc), "path": "models/best_model.keras"}, f, indent=2)
