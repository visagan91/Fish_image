import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

DATA_DIR = "data"
TEST_DIR = os.path.join(DATA_DIR, "test")
OUT_DIR = "models"

def test_gen(img_size=(224,224), batch=32):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=batch, class_mode="categorical", shuffle=False
    )

def evaluate(model_path, img_size=(224,224)):
    gen = test_gen(img_size)
    model = load_model(model_path)
    y_prob = model.predict(gen)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = gen.classes
    target_names = list(gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    with open(os.path.join(OUT_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    plt.yticks(range(len(target_names)), target_names)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))

    top5 = np.mean([int(t in np.argsort(p)[-5:]) for t, p in zip(y_true, y_prob)])
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Accuracy: {report['accuracy']:.4f}\nTop-5 Accuracy: {top5:.4f}\n")

if __name__ == "__main__":
    evaluate(os.path.join(OUT_DIR, "best_model.h5"))
