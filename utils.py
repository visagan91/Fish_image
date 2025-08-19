# utils.py  (or infer_utils.py if that's the name you import)
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

DEFAULT_MODEL_PATH = "models/best_model.keras"
DEFAULT_LABELS_PATH = "models/class_indices.json"
DEFAULT_IMG_SIZE = (224, 224)

class Predictor:
    """
    Lightweight predictor for the exported Keras model.
    - Assumes preprocessing is already inside the model (no extra rescaling here).
    - Works with the modern .keras model file produced after training.
    """

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 labels_path: str = DEFAULT_LABELS_PATH,
                 img_size=DEFAULT_IMG_SIZE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Label map not found: {labels_path}")

        # Load model (no compile needed for inference)
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = tuple(img_size)

        # Load id->label mapping written by train.py
        with open(labels_path) as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        self.labels = [id2label[i] for i in range(len(id2label))]

        # Sanity check: output units vs number of labels
        n_out = self.model.output_shape[-1]
        if n_out != len(self.labels):
            raise ValueError(
                f"Model outputs {n_out} classes but labels file has {len(self.labels)} entries."
            )

    def _to_batch(self, img_or_path):
        """Convert a PIL image or path into a (1, H, W, 3) float32 array.
        Do NOT rescale: the model graph already contains preprocessing."""
        if isinstance(img_or_path, (str, os.PathLike)):
            img = Image.open(img_or_path)
        else:
            img = img_or_path
        img = img.convert("RGB").resize(self.img_size)
        arr = np.asarray(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    def predict(self, img_or_path, top_k: int = 3):
        """Returns a list of (label, probability) sorted by probability desc."""
        x = self._to_batch(img_or_path)
        probs = self.model.predict(x, verbose=0)[0]

        # If final layer wasn't softmax for some reason, normalize
        if probs.ndim == 1 and (probs.min() < 0 or probs.max() > 1 or not np.isclose(probs.sum(), 1, atol=1e-3)):
            probs = tf.nn.softmax(probs).numpy()

        idxs = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in idxs]
