import os
import cv2
import numpy as np
from numba import njit


def similarity_matrix(mat1, mat2):
    """find cosine similarity between two matrices
    Args:
        mat1 (np.array): array of shape [num_images,len_of_embedding]
        mat2 (np.array): array of shape [num_images,len_of_embedding]
    Return:
        np.array : lower triangle of cosine similarity matrix
                    of shape [num_images,num_images]
    """
    emb1_norm = np.linalg.norm(mat1, axis=1, ord=2, keepdims=True)
    emb2_norm = np.linalg.norm(mat2, axis=1, ord=2, keepdims=True)
    matrix = np.matmul(mat1, mat2.transpose()) / \
        (emb1_norm * emb2_norm.transpose())
    return np.tril(matrix, -1)


def sort_matrix(mat):
    """Sort a matrix based on its value
    Args:
        mat: array of shape [num_images,num_images]
    Returns:
        idxs_pair (list) : list of indexs sorted in non-increasing order
    """
    flat = mat.flatten()
    # we are using argpartition bcoz it is faster than argsort
    # when we only need a subset of an array
    indices = np.argpartition(flat, -len(mat))[-len(mat):]
    indices = indices[np.argsort(-flat[indices])]
    idxs = np.array(np.unravel_index(indices, mat.shape))
    idxs_pair = list(zip(idxs[0, :], idxs[1, :]))
    return np.array(idxs_pair)


@njit
def cosine_similarity(emb1, emb2):
    """find cosine similarity between two array
    Args:
        emb1 (np.array): array of shape [len_of_embedding]
        emb2 (np.array): array of shape [len_of_embedding]
    Return:
        float : cosine similarity value
    """
    return np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))


def read_files(img_dir):
    """Read all the images in dataset directory
    Args:
        img_dir (str): path of image folder
    Return:
        filenames (list): list of paths of all the images in `img_dir`
    """
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
