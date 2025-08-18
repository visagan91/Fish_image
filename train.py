import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_zoo import build_scratch_cnn, build_transfer_model, BACKBONES

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def make_gens(img_size=IMG_SIZE):
    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator()# no (rescale=1./255)

    # in src/train.py (make_gens)
    train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=img_size, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, color_mode="rgb"   # <— EffiecientNet fixx
)
    val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=img_size, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False, color_mode="rgb"  # <— EffNet fix
)

    return train_gen, val_gen

def compile_fit(model, train_gen, val_gen, epochs, out_path):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        ModelCheckpoint(out_path, monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cbs)
    return hist

def run_all():
    train_gen, val_gen = make_gens()
    n_classes = train_gen.num_classes
    input_shape = (*IMG_SIZE, 3)

    # save label map
    class_indices = train_gen.class_indices
    with open(os.path.join(OUT_DIR, "class_indices.json"), "w") as f:
        json.dump({v:k for k,v in class_indices.items()}, f, indent=2)

    histories = {}
    best_models = []

    # 1) from-scratch CNN
    scratch = build_scratch_cnn(input_shape, n_classes)
    out_path = os.path.join(OUT_DIR, "scratch_cnn_best.h5")
    hist = compile_fit(scratch, train_gen, val_gen, EPOCHS_STAGE1 + EPOCHS_STAGE2, out_path)
    best_acc = max(hist.history["val_accuracy"])
    histories["scratch_cnn"] = hist.history
    best_models.append(("scratch_cnn", out_path, float(best_acc)))

    # 2) transfer backbones
    for name in BACKBONES.keys():
        model, _ = build_transfer_model(name, input_shape, n_classes, train_base=False, dropout=0.3)
        out_path = os.path.join(OUT_DIR, f"{name}_stage1_best.h5")
        hist1 = compile_fit(model, train_gen, val_gen, EPOCHS_STAGE1, out_path)
        # fine-tune top ~20%
        n_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            if i > int(0.8 * n_layers):
                layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        out_path2 = os.path.join(OUT_DIR, f"{name}_finetuned_best.h5")
        hist2 = compile_fit(model, train_gen, val_gen, EPOCHS_STAGE2, out_path2)

        histories[name] = {"stage1": hist1.history, "stage2": hist2.history}
        best_acc = max(hist2.history["val_accuracy"])
        best_models.append((name, out_path2, float(best_acc)))

    # pick global best
    best_models.sort(key=lambda x: x[2], reverse=True)
    best_name, best_path, best_acc = best_models[0]
    final_link = os.path.join(OUT_DIR, "best_model.h5")
    tf.keras.models.load_model(best_path).save(final_link)
    with open(os.path.join(OUT_DIR, "best_model_meta.json"), "w") as f:
        json.dump({"name": best_name, "val_accuracy": best_acc, "path": "models/best_model.h5"}, f, indent=2)

    with open(os.path.join(OUT_DIR, "histories.json"), "w") as f:
        json.dump(histories, f)

if __name__ == "__main__":
    run_all()
