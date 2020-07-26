import numpy as np
import multiprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.utils import Sequence
from utils import load_image


class Embedding():
    def __init__(self, files, batch_size=32, input_shape=(224, 224)):
        self.files = files
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.model = self.load_model()
        self.device = tf.test.is_gpu_available()

    def load_model(self):
        input_layer = layers.Input(
            shape=(self.input_shape + (3,)))

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

    def embeddings(self):
        if self.device:
            return self.predict_on_batch()
        return self.predict_on_cpu()

    def predict_on_cpu(self):
        embs = []
        for file_ in self.files:
            image = load_image(file_)
            embs.append(self.model.predict(np.expand_dims(image, axis=0))[0])
        return embs

    def predict_on_batch(self):
        data_loader = DataGenerator(
            self.files, self.batch_size, self.input_shape)
        predictions = self.model.predict(
            data_loader, verbose=1, workers=multiprocessing.cpu_count())
        return predictions


class DataGenerator(Sequence):

    def __init__(self, img_list, bs=16, input_size=(224, 224)):

        self.img_list = img_list
        self.bs = bs
        self.input_size = input_size

    def __len__(self):

        return np.ceil(len(self.img_list) / self.bs).astype(int)

    def __getitem__(self, idx):

        x_batch = []
        start = idx*self.bs
        end = (idx+1)*self.bs
        image_ids = self.img_list[start:end]
        for i, ids in enumerate(image_ids):
            image = load_image(image_ids[i], self.input_size)
            x_batch.append(image/255.)
        x_batch = np.array(x_batch, np.float32)
        return x_batch
