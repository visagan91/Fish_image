import tensorflow as tf
from tensorflow.keras import layers, models

def classification_head(inputs, n_classes, dropout=0.2):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return outputs

def build_scratch_cnn(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    outputs = classification_head(x, n_classes)
    return models.Model(inputs, outputs, name="scratch_cnn")

BACKBONES = {
    "VGG16": {
        "builder": tf.keras.applications.VGG16,
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
        "kwargs": {"weights":"imagenet", "include_top":False}
    },
    "ResNet50": {
        "builder": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "kwargs": {"weights":"imagenet", "include_top":False}
    },
    "MobileNet": {
        "builder": tf.keras.applications.MobileNet,
        "preprocess": tf.keras.applications.mobilenet.preprocess_input,
        "kwargs": {"weights":"imagenet", "include_top":False}
    },
    "InceptionV3": {
        "builder": tf.keras.applications.InceptionV3,
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
        "kwargs": {"weights":"imagenet", "include_top":False}
    },
    "EfficientNetB0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "kwargs": {"weights":"imagenet", "include_top":False}
    },
}

def build_transfer_model(name, input_shape, n_classes, train_base=False, dropout=0.2):
    cfg = BACKBONES[name]
    base = cfg["builder"](input_shape=input_shape, **cfg["kwargs"])
    base.trainable = train_base
    inputs = tf.keras.Input(shape=input_shape)
    x = cfg["preprocess"](inputs)
    x = base(x, training=False)
    outputs = classification_head(x, n_classes, dropout=dropout)
    return models.Model(inputs, outputs, name=f"{name}_tl"), cfg["preprocess"]
