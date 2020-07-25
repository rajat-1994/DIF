import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from utils import load_image


def load_model():
    input_layer = layers.Input(
        shape=(224, 224, 3))

    # Loading base model
    mobilenet = MobileNet(weights="imagenet",
                          input_tensor=input_layer,
                          alpha=0.5,
                          include_top=False)

    mobilenet.trainable = False
    x = layers.GlobalAveragePooling2D()(mobilenet.output)
    model = Model(inputs=input_layer,
                  outputs=x,
                  name='pose_model')

    return model


def embeddings(files, model):
    embs = []
    for file_ in files:
        image = load_image(file_)
        embs.append(model.predict(np.expand_dims(image, axis=0))[0])
    return embs
