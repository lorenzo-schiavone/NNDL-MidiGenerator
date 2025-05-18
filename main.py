import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import mido
from utils import *

midi_paths = list(glob("./maestro-v1.0.0/2004/*.midi")) 

mid = mido.MidiFile(midi_paths[1], clip=True)
unit_fraction = 1/4
X = mid2matrix(mid, unit_fraction)
tokens = tokenize(X, 128)

# plt.figure(figsize=(12, 4))
# plt.imshow(tokens[11])
# plt.show()

n_examples = 5  
plt.figure(figsize=(15, 3))  # dimensioni larghe per n_examples affiancati

for i in range(n_examples):
    plt.subplot(1, n_examples, i+1)  # 1 riga, n_examples colonne, i+1-esima subplot
    plt.imshow(tokens[i], cmap='gray')  # o altro cmap se preferisci
    plt.axis('off')  # togli assi per chiarezza
    plt.title(f'Token {i}')

plt.tight_layout()
plt.show()