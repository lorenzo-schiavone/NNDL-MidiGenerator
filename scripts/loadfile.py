import numpy as np
from glob import glob

path = list(glob("./npy/*"))

mid = np.load(path[0])

print(mid.shape)