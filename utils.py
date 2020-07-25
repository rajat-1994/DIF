import cv2
import os
import numpy as np
from numba import njit


def similarity_matrix(emb1, emb2):
    emb1_norm = np.linalg.norm(emb1, axis=1, ord=2, keepdims=True)
    emb2_norm = np.linalg.norm(emb2, axis=1, ord=2, keepdims=True)
    matrix = np.matmul(emb1, emb2.transpose()) / \
        (emb1_norm * emb2_norm.transpose())
    return np.tril(matrix, -1)


@njit
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))


def read_files(img_dir):
    filenames = []
    for dirpath, _, filename in os.walk(img_dir):
        for file_ in filename:
            if os.path.splitext(file_)[-1] in (".jpg", ".png", ".jpeg"):
                filenames.append(os.path.join(dirpath, file_))
    return filenames


def load_image(path, shape=(224, 224)):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, shape)
    return image
