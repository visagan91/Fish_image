import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_label_map(path="models/class_indices.json"):
    with open(path) as f:
        idx2label = json.load(f)
    return {int(k):v for k,v in idx2label.items()}

def preprocess_img(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = arr/255.0
    arr = np.expand_dims(arr, 0)
    return arr

class Predictor:
    def __init__(self, model_path="models/best_model.h5", labels_path="models/class_indices.json"):
        self.model = load_model(model_path)
        self.labels = load_label_map(labels_path)
        self.target_size = self.model.input_shape[1:3]

    def predict(self, img_path, top_k=3):
        arr = preprocess_img(img_path, self.target_size)
        probs = self.model.predict(arr)[0]
        idxs = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in idxs]
