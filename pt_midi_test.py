import pretty_midi
import matplotlib.pyplot as plt


midi_path = './Acqua azzurra, acqua chiara.mid'

mid = pretty_midi.PrettyMIDI(midi_path)

tempo = mid.get_tempo_changes()[1][0] 
beats_per_bar = 4  
# Compute fs for 16 steps per bar
fs = 16 / (60/tempo * beats_per_bar)

'''
pr = mid.get_piano_roll(fs=fs)
pr[pr>0]=1
plt.figure(figsize=(14,6))
plt.imshow(pr, cmap="gray", aspect='auto', origin='lower')
plt.show()
'''
chroma = mid.get_chroma(fs=fs/16)
plt.figure(figsize=(14,6))
plt.imshow(chroma, cmap="gray", aspect='auto', origin='lower')
plt.show()

## poi confronto con maggiore e minore rollato per dodice note 
## coseno per prendere quella più simile