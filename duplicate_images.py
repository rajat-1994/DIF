import numpy as np
from model_utils import load_model, embeddings
from utils import read_files, similarity_matrix
import time

img_dir = "./data/"
files = read_files(img_dir)
# files = files[:10]
print(len(files))

model = load_model()
print("Model loaded...")

a = time.time()
embs = np.array(embeddings(files, model))
print(time.time()-a)

a = time.time()
# matrix = np.zeros((len(files),)*2)
matrix = similarity_matrix(embs, embs)
print(time.time()-a)

print(matrix.shape)
