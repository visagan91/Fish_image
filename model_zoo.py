import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


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
    #"EfficientNetB0": {
    #    "builder": tf.keras.applications.EfficientNetB0,
    #   "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    #    "kwargs": {"weights":"imagenet", "include_top":False}
    #},
}

def build_transfer_model(name, input_shape, n_classes, train_base=False, dropout=0.2):
    K.set_image_data_format("channels_last")
    h, w = input_shape[:2]
    input_shape_3c = (h, w, 3) # ensure 3 channels
    cfg = BACKBONES[name]
    # Our 3-channel input
    h, w = input_shape[:2]
    inputs = tf.keras.Input(shape=(h, w, 3), name=f"{name}_input")
    # Preprocess as part of the graph
    x = cfg["preprocess"](inputs)
    base = cfg["builder"](input_shape=input_shape_3c, **cfg["kwargs"])
    base.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape_3c)
    x = cfg["preprocess"](inputs)
    # Build the backbone by attaching it to our tensor (forces 3ch)
    base = cfg["builder"](input_tensor=x, include_top=False, weights="imagenet")
    base.trainable = train_base

    outputs = classification_head(base.output, n_classes, dropout=dropout)
    model = models.Model(inputs, outputs, name=f"{name}_tl")
    return model, cfg["preprocess"]