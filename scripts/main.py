import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import mido
from utils import *

midi_paths = list(glob("../data/*.mid")) 

mid = mido.MidiFile(midi_paths[0], clip=True)

# for m in mid.tracks[1][:1000]:
#     print(m)

# for i,track in enumerate(mid.tracks):
#     try:
#         plt.figure(i)
#         new_mid = mido.MidiFile()
#         new_mid.tracks.append(track)
#         result_array = downsample_piano_roll(mid2arry(new_mid)[:100000],new_mid.ticks_per_beat, 1/8)
#         plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
#     except:
#         print(f'error with track {i}')
# plt.show()

print(len((mid.tracks)))


arr = track2seq(mid.tracks[1])
arr = np.array(arr).T
print(arr.shape)
np.save(midi_paths[0].split('/')[-1].split('.')[-2], arr)


# # result_array = downsample_piano_roll(mid2arry(mid)[:100000],mid.ticks_per_beat, 1/8)
# plt.figure(figsize=(12, 4))
# plt.imshow(arr[:100,:])
# plt.show()

# plt.plot(range(arr.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# plt.title("")
# plt.show()
# unit_fraction = 1/4
# X = mid2matrix(mid, unit_fraction)
# tokens = tokenize(X, 128)

# # plt.figure(figsize=(12, 4))
# # plt.imshow(tokens[11])
# # plt.show()

# n_examples = 5  
# plt.figure(figsize=(15, 3))  # dimensioni larghe per n_examples affiancati

# for i in range(n_examples):
#     plt.subplot(1, n_examples, i+1)  # 1 riga, n_examples colonne, i+1-esima subplot
#     plt.imshow(tokens[i])  # o altro cmap se preferisci
#     plt.axis('off')  # togli assi per chiarezza
#     plt.title(f'Token {i}')

# plt.tight_layout()
# plt.show()