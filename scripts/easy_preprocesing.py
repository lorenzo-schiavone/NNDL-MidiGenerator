import numpy as np
import os
from glob import glob
import pretty_midi
from tqdm import tqdm
import matplotlib.pyplot as plt

## utils

def mid2pianoroll(mid, ticks_per_bar = 16):
    ## traccia singola
    fs = ticks_per_bar / (60/mid.get_tempo_changes()[1][0] * 4) # 4: beats_per bar
    pr = mid.get_piano_roll(fs=fs)
    pr[pr>0]=1
    return pr

def get_harmony(mid, nbar, similarity_threshold = .7):
    piano_roll = mid2pianoroll(mid)
    # get chroma as in the pretty midi library
    chroma_matrix = np.zeros((12, piano_roll.shape[1]))
    for note in range(12):
        chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)

    bar_len = 16
    # num_bars = int(np.floor(piano_roll.shape[1] / bar_len))
    chords = []
    for i in range(0, nbar):
        mean_chroma = np.mean(chroma_matrix[:, i*bar_len:(i+1)*bar_len], axis=1)
        chord = get_best_match(mean_chroma, similarity_threshold)
        chords.append(chord)
    return chords

CHORD_TEMPLATES = [np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]), #maggiore
                   np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), #minore
                   ] 

def get_best_match(chroma_vector: np.ndarray, similarity_threshold: float):
    best_match = [0,0]
    best_match_score = -1.0
    if np.sum(chroma_vector) < 1e-6:
        return best_match

    chroma_norm = np.linalg.norm(chroma_vector)
    for i, template in enumerate(CHORD_TEMPLATES):
        norm_product = chroma_norm*np.linalg.norm(template)
        for root_pc in range(12):
            # Ruota il template di accordo per la tonica corrente
            rotated_template = np.roll(template, root_pc)
            # Calcola la similarità coseno
            dot_product = np.dot(chroma_vector, rotated_template)
            similarity = dot_product / norm_product

            if similarity > best_match_score:
                best_match_score = similarity
                best_match = [root_pc, i]

    return best_match if best_match_score >= similarity_threshold else [0,0]

def write_harmony_file(chords, output_path):
    with open(output_path, 'w') as f:
        for chord in chords:
            key, quality = chord
            f.write(f'{key} {quality}\n')
    return 

def read_harmony_file(input_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    T = len(lines)
    chords = np.zeros((13, T))
    for i,line in enumerate(lines):
        key_str, minor_str = line.strip().split()
        key = int(key_str)
        is_minor = int(minor_str)
        chords[key,i] = 1
        chords[12,i] = is_minor
    return chords 



## DAL SITO hookthoery/theorytabs 
## i midi scsaricati hanno la melodia nella prima traccia e gli accordi nella seconda

## plan: - estrarre prima traccia e salvarla altrove
##       - estrarre seconda, fare il piano roll, fare il chroma medio per battuta e trovare l'accordo corrispondente
##       - salvarlo in formato utile: es 7 1 \n 12 0 \n ...
##       - funzione per leggere questo formato e caricarlo in one hot representation 13 x 1 vettore per battuta

base_path = '../data'

hookthoery_folder = 'hooktheory'
output_folder = 'melody_chord_ht'

hookthoery_folder_path = os.path.join(base_path, hookthoery_folder)
output_folder_path = os.path.join(base_path, output_folder)

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

midi_files = glob(f'{hookthoery_folder_path}/*.mid')

for midi_file in midi_files:
    print(f"Doing {midi_file}")
    output_midi_file = midi_file.replace(hookthoery_folder, output_folder)
    output_chord_file = output_midi_file.replace('.mid', '.txt')

    midi = pretty_midi.PrettyMIDI(midi_file)
    try:

        melody_midi = pretty_midi.PrettyMIDI()
        melody_midi.instruments.append(midi.instruments[0])
        
        harmony_midi = pretty_midi.PrettyMIDI()
        harmony_midi.instruments.append(midi.instruments[1])
    except Exception as e:
        print(f"Error with {midi_file}: {e}")

    melody_midi.write(output_midi_file)

    # pr = mid2pianoroll(harmony_midi)
    # plt.imshow(pr, cmap='gray', aspect='auto')
    # plt.show()
    nbar =  int(np.floor( mid2pianoroll(melody_midi).shape[1] / 16))
    chords = get_harmony(harmony_midi, nbar)
    # salva su file
    write_harmony_file(chords, output_chord_file)

    # read_chord = read_harmony_file(output_chord_file)

    # pr = mid2pianoroll(melody_midi)
    # print("shape pr: ", int(np.floor(pr.shape[1] / 16)))
    # print("shape chords: ", read_chord.shape)



