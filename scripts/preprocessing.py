import numpy as np
import string
from glob import glob
import re
import os
from collections import defaultdict
from typing import List, Optional, Dict, Set, Tuple, Any
import pretty_midi
from tqdm import tqdm
#from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

## UTILITIES

def piano_roll_to_pretty_midi(piano_roll, bpm, fs=100, program=0):
    """
    Converte un piano roll in un oggetto PrettyMIDI. 
    Rimuove opzionalmente le note più brevi di un sedicesimo se bpm è specificato.
    """
    notes, frames = piano_roll.shape
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Aggiunge padding per rilevare le note alla fine
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    # Durata minima (in secondi) per mantenere una nota
    min_duration = (15.0 / bpm) if bpm else 0.0

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            duration = time - note_on_time[note]
            if duration >= min_duration:
                note_velocity = np.clip(int(prev_velocities[note]), 1, 127)
                pm_note = pretty_midi.Note(
                    velocity=note_velocity,
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    midi.instruments.append(instrument)
    return midi


#-------------------------------------------------------------------------------

def monophonize_track(track: pretty_midi.Instrument, bpm, fs: int = 100) -> pretty_midi.Instrument:
    """
    Converte una traccia potenzialmente polifonica in una monofonica,
    mantenendo solo la nota più acuta (highest pitch) per ogni step temporale.
    """
    if not track.notes:
        return track # Ritorna la traccia vuota se non ci sono note

    piano_roll = track.get_piano_roll(fs=fs)

    monophonic_roll = np.zeros_like(piano_roll)
    for t in range(piano_roll.shape[1]):
        # notes_at_t = np.where(piano_roll[:, t] > 0)[0]
        # if notes_at_t.size > 0:
        #     highest_note_pitch = np.max(notes_at_t)
        #     monophonic_roll[highest_note_pitch, t] = piano_roll[highest_note_pitch, t]
        notes_at_t = piano_roll[:, t]
        if notes_at_t.any():
            highest = notes_at_t.argmax()
            monophonic_roll[highest, t] = notes_at_t[highest]

    mono_midi = piano_roll_to_pretty_midi(monophonic_roll, bpm, fs=fs, program=track.program)

    if mono_midi.instruments:
        mono_track = mono_midi.instruments[0]
        mono_track.name = track.name
        return mono_track
    else:
        return pretty_midi.Instrument(program=track.program, name=track.name)


def mid2pianoroll(mid, ticks_per_bar = 16):
    ## traccia singola
    fs = ticks_per_bar / (60/mid.get_tempo_changes()[1][0] * 4) # 4: beats_per bar
    pr = mid.get_piano_roll(fs=fs)
    pr[pr>0]=1
    return pr

#------------------------------------------------------------------------------#
#                                 FILTERS                                      #
#------------------------------------------------------------------------------# 

def filter_percussive_tracks(
    tracks: List[pretty_midi.Instrument],
    percussive_programs: Set[int]
) -> Set[int]:
    """
    Identifica gli indici delle tracce percussive in una lista di tracce MIDI.

    L'identificazione si basa su due criteri principali:
    1. Metadati espliciti (flag `is_drum` o programma MIDI percussivo).
    2. Un'analisi euristica che valuta la varietà dei pitch e la durata media delle note.
       Le tracce percussive tendono ad avere bassa varietà di pitch e note di breve durata.

    Args:
        tracks: Una lista di oggetti `pretty_midi.Instrument`.
        percussive_programs: Un set di numeri di programma MIDI da considerare percussivi.

    Returns:
        Un set contenente gli indici delle tracce identificate come percussive.
    """
    percussive_indices: Set[int] = set()

    for i, track in enumerate(tracks):
        # Criterio 1: Metadati
        if track.is_drum or track.program in percussive_programs:
            percussive_indices.add(i)
            continue

        # Esclusione di tracce con poche note per un'analisi statistica irrilevante
        if len(track.notes) < 10:
            continue

        # Criterio 2: Analisi euristica
        unique_pitches = len(set(note.pitch for note in track.notes))
        # Lo score diminuisce all'aumentare della varietà dei pitch. Normalizzato su un'ottava (12 semitoni).
        score_pitch_variety = max(0, 1.0 - (unique_pitches - 1) / 12.0)

        avg_duration = np.mean([note.end - note.start for note in track.notes])
        # Lo score diminuisce all'aumentare della durata media delle note.
        score_note_duration = 1 / (1 + avg_duration * 10)

        # Combinazione pesata degli score con una soglia di attivazione.
        # I parametri (pesi 0.6/0.4, soglia 0.7) sono euristici.
        if (score_pitch_variety * 0.6 + score_note_duration * 0.4) > 0.7:
            percussive_indices.add(i)

    return percussive_indices

#-------------------------------------------------------------------------------

def filter_track_by_name(track: pretty_midi.Instrument, exclusion_keywords: List[str]) -> bool:
    """
    Filtra le tracce il cui nome contiene parole chiave da escludere (es. 'bass', 'piano').
    Ritorna True se la traccia supera il filtro (non contiene keyword), altrimenti False.
    """
    if not track.name:
        return True # Mantiene la traccia se non ha nome

    track_name_lower = track.name.lower()
    for keyword in exclusion_keywords:
        if keyword in track_name_lower:
            return False # Esclude la traccia se trova una keyword

    return True

#-------------------------------------------------------------------------------

def filter_track_by_pitch(track: pretty_midi.Instrument, min_pitch_threshold: int = 40) -> bool:
    """
    Filtra le tracce che contengono note al di sotto di una soglia di pitch minima.
    Ritorna True se la traccia supera il filtro (tutte le note sono sopra la soglia), altrimenti False.
    - min_pitch_threshold (default=40): Corrisponde a E2 -> nota più bassa di un uomo con voce da basso
    """
    if not track.notes:
        return False  # Scarta tracce vuote

    # Trova la nota più bassa nella traccia
    lowest_pitch = min(note.pitch for note in track.notes)

    # Ritorna True solo se la nota più bassa è uguale o superiore alla soglia
    return lowest_pitch >= min_pitch_threshold

#-------------------------------------------------------------------------------

def filter_track_by_note_density(track: pretty_midi.Instrument, total_duration: float, min_density: float = 0.35) -> bool:
    """
    Filtra le tracce che hanno una bassa densità di note rispetto alla durata totale.
    Ritorna True se la densità di note è superiore alla soglia, altrimenti False.
    - min_density (default=0.35): La traccia deve avere note attive per almeno il 35% del tempo.
    """
    if not track.notes or total_duration == 0:
        return False
    note_time = sum(note.end - note.start for note in track.notes)
    density = note_time / total_duration
    return density >= min_density

#-------------------------------------------------------------------------------

def filter_track_by_polyphony(track: pretty_midi.Instrument, fs: int = 100, polyphony_threshold: float = 0.15) -> bool:
    """
    Filtra le tracce con un'alta percentuale di polifonia. 3 TRACCE CONTEMPORANEAMENTE, 2 VA BENE
    Ritorna True se la traccia è prevalentemente monofonica, altrimenti False.
    - polyphony_threshold (default=0.15): Soglia massima di passi temporali polifonici.
    """
    if len(track.notes) < 2:
        return True
    pianoroll = track.get_piano_roll(fs=fs)
    polyphonic_steps = np.sum(np.sum(pianoroll > 0, axis=0) > 2) # SE METTO >1 DIVENTA ANCHE PER 2 TRACCE
    total_steps = pianoroll.shape[1]
    if total_steps == 0:
        return True
    polyphony_ratio = polyphonic_steps / total_steps
    return polyphony_ratio <= polyphony_threshold

#-------------------------------------------------------------------------------

def find_vocal_by_keywords(midi_data: pretty_midi.PrettyMIDI, keywords: List[str]) -> Optional[pretty_midi.Instrument]:
    """
    Cerca una traccia tramite keyword nel nome. Ha la precedenza su tutti i filtri.
    """
    for i, track in enumerate(midi_data.instruments):
        if not track.name: continue
        track_name_lower = track.name.lower()
        for keyword in keywords:
            if re.search(r'\\b' + re.escape(keyword.lower()) + r'\\b', track_name_lower):
                print(f"Override per nome: Trovata traccia '{track.name}' con keyword '{keyword}'.")
                return track
    return None


#------------------------------------------------------------------------------#
#                       SCORING MELODY FUNCTIONS                               #
#------------------------------------------------------------------------------# 

def score_instrument_type(track: pretty_midi.Instrument) -> float:
    melodic_programs = list(range(0, 8)) + list(range(24, 32))  + \
                       list(range(52, 69))
    main_programs = list(range(40, 44)) + list(range(80, 88)) + list(range(72, 80))
    bass_programs = list(range(32, 40))
    pad_programs = list(range(88, 96))
    if track.is_drum: return 0.0
    if track.program in bass_programs: return 0.1
    if track.program in pad_programs: return 0.3
    if track.program in melodic_programs: return 0.8
    if track.program in main_programs: return 1
    return 0.5

#-------------------------------------------------------------------------------

def score_note_density_heuristic(track: pretty_midi.Instrument) -> float:
    if not track.notes: return 0.0
    duration = track.get_end_time()
    if duration == 0: return 0.0
    return 1 - np.exp(-0.25 * (len(track.notes) / duration))

#-------------------------------------------------------------------------------

def score_mean_pitch(track: pretty_midi.Instrument) -> float:
    if not track.notes: return 0.0
    return np.mean([note.pitch for note in track.notes]) / 127.0

#-------------------------------------------------------------------------------

def score_pitch_std(track: pretty_midi.Instrument) -> float:
    if len(track.notes) < 2: return 0.0
    return np.clip(np.std([note.pitch for note in track.notes]) / 20.0, 0, 1.0)

# CHORD EXTRACTIONS
def get_best_match(chroma_vector: np.ndarray, similarity_threshold: float) -> str:

    if np.sum(chroma_vector) < 1e-6:
        return "N.C."

    best_match = {"score": -1.0, "name": "N.C."}
    chroma_norm = np.linalg.norm(chroma_vector)
    for quality, template in CHORD_TEMPLATES.items():
        norm_product = chroma_norm*np.linalg.norm(template)
        for root_pc in range(12):
            # Ruota il template di accordo per la tonica corrente
            rotated_template = np.roll(template, root_pc)
            # Calcola la similarità coseno
            dot_product = np.dot(chroma_vector, rotated_template)
            similarity = dot_product / norm_product

            if similarity > best_match["score"]:
                best_match["score"] = similarity
                best_match["name"] = f"{PITCH_CLASSES[root_pc]}_{quality}"

    return best_match["name"] if best_match["score"] >= similarity_threshold else "N.C."

#------------------------------------------------------------------------------#
#                              CHORDS EXTRACTION                               #
#------------------------------------------------------------------------------# 
def analyze_harmonic_progression(
    midi_data: pretty_midi.PrettyMIDI,
    percussive_indices: Set[int],
    tracks_to_exclude: Optional[Set[int]] = None,
    similarity_threshold: float = 0.5
) -> List[str]:
    """
    Esegue una pipeline di analisi armonica su un oggetto pretty_midi.

    Args:
        midi_data: L'oggetto `pretty_midi.PrettyMIDI` da analizzare.
        percussive_indices: Un set di indici di tracce percussive, calcolato esternamente.
        tracks_to_exclude: Un set opzionale di indici di tracce da escludere a priori
                           dall'analisi (es. la melodia principale).
        similarity_threshold: La soglia di similarità (0-1) per il riconoscimento degli accordi.

    Returns:
        Una lista di stringhe che rappresenta la progressione armonica
    """
    # --- Fase 1: Filtro delle Tracce ---
    # Unisce gli indici delle tracce percussive (pre-calcolati) con quelli specificati dall'utente.
    all_excluded_indices = percussive_indices.union(tracks_to_exclude if tracks_to_exclude else set())

    # --- Fase 2: Creazione del MIDI Armonico ---
    # Costruisce un nuovo oggetto MIDI contenente solo le tracce armoniche.
    harmonic_midi = pretty_midi.PrettyMIDI()
    for i, track in enumerate(midi_data.instruments):
        if i not in all_excluded_indices:
            harmonic_midi.instruments.append(track)

    if not harmonic_midi.instruments:
        return ["Nessuna traccia armonica rilevata."]

    piano_roll = mid2pianoroll(harmonic_midi)
    # get chroma as in the pretty midi library
    chroma_matrix = np.zeros((12, piano_roll.shape[1]))
    for note in range(12):
        chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)

    bar_len = 16
    num_bars = int(piano_roll.shape[1] / bar_len)

    progression: List[str] = []
    for i in range(0, num_bars):
      mean_chroma = np.mean(chroma_matrix[:, i*bar_len:(i+1)*bar_len], axis=1)
      chord = get_best_match(mean_chroma, similarity_threshold)
      progression.append(chord)

    return progression

#------------------------------------------------------------------------------#
#                              MELODY EXTRACTION                               #
#------------------------------------------------------------------------------# 

def get_candidate_melody_indices(
    midi_data: pretty_midi.PrettyMIDI,
    drum_track_indices: Set[int],
    verbose: bool = True
) -> List[int]:
    """
    Identifica e ritorna gli indici delle tracce candidate a essere la melodia.

    La funzione esegue una ricerca prioritaria per nome e, in assenza di risultati,
    procede con una pipeline di filtraggio escludendo le tracce non idonee.
    L'output è una lista di indici pronti per la successiva fase di scoring.

    Args:
        midi_data (pretty_midi.PrettyMIDI): L'oggetto MIDI da analizzare.
        drum_track_indices (Set[int]): Un set di indici di tracce percussive pre-calcolati.
        verbose (bool): Se True, stampa log dettagliati del processo di filtraggio.

    Returns:
        List[int]: Una lista contenente gli indici delle tracce candidate.
    """
    # --- Configurazione dei parametri per il filtraggio ---
    params = {
        'min_notes_for_analysis': 20, 'polyphony_fs': 100,
        'polyphony_threshold': 0.30, 'note_density_threshold': 0.30,
        'min_pitch': 40,  # E2
        'melody_keywords': ['melody', 'melodia', 'lead', 'vocal', 'vocals', 'voice',
                            'singer', 'cantante', 'main', 'solo', 'trumpet', 'sax', 'choir'],
        'exclusion_keywords': ['bass', 'pad', 'rhodes']
    }

    # --- FILTRAGGIO DELLE TRACCE ---
    candidate_indices: List[int] = []
    total_duration = midi_data.get_end_time()

    if total_duration == 0:
        if verbose: print("Durata del brano è zero. Impossibile procedere.")
        return []

    for i, track in enumerate(midi_data.instruments):
        # La logica di filtraggio determina se l'indice della traccia deve essere aggiunto ai candidati.
        log_message = ""
        is_candidate = True

        if len(track.notes) < params['min_notes_for_analysis']:
            is_candidate = False
            log_message = f"SCARTATA: Poche note ({len(track.notes)})."
        elif i in drum_track_indices:
            is_candidate = False
            log_message = "SCARTATA: Indice percussivo pre-calcolato."
        elif not filter_track_by_name(track, params['exclusion_keywords']):
            is_candidate = False
            log_message = "SCARTATA: Nome traccia escluso."
        elif not filter_track_by_pitch(track, params['min_pitch']):
            is_candidate = False
            log_message = "SCARTATA: Pitch medio troppo basso."
        elif not filter_track_by_note_density(track, total_duration, params['note_density_threshold']):
            is_candidate = False
            log_message = "SCARTATA: Densità note bassa."
        elif not filter_track_by_polyphony(track, params['polyphony_fs'], params['polyphony_threshold']):
            is_candidate = False
            log_message = "SCARTATA: Troppo polifonica."

        if is_candidate:
            candidate_indices.append(i)
            log_message = "CANDIDATA"

        if verbose:
            print(f"Traccia {i} ('{track.name}'): {log_message}")

    if verbose:
        print("-" * 50)
        print(f"Trovati {len(candidate_indices)} indici candidati dopo il filtraggio.")

    return candidate_indices

#-------------------------------------------------------------------------------

def get_best_candidate(midi_data, candidate_melody_indices, weights=None, verbose=False):
    '''monophonize, filter, SCORING E SELEZIONE FINALE'''
    if not weights:
        weights = {
            'instrument_type': 0.3,
            'note_density': 0.4,
            'mean_pitch': 0.15,
            'pitch_std': 0.15
        }

    if not candidate_melody_indices:
        if verbose: print("\nNessuna traccia candidata trovata. Impossibile estrarre la melodia.")
        return None, []

    best_score = -10
    best_track = None

    bpm = midi_data.estimate_tempo()

    for index in candidate_melody_indices:
        track = midi_data.instruments[index]
        monophonic_track = monophonize_track(track, bpm)
        scores = {
            'instrument_type': score_instrument_type(track),
            'note_density': score_note_density_heuristic(track),
            'mean_pitch': score_mean_pitch(track),
            'pitch_std': score_pitch_std(track)
        }
        final_score = sum(scores[key] * weights.get(key, 0) for key in scores)
        if final_score > best_score:
            best_score = final_score
            best_track = monophonic_track

    return best_track



# Definizioni dei Template e Costanti
CHORD_TEMPLATES: Dict[str, np.ndarray] = {
    'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
    'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    #'dom7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    #'pwr': np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
}

PITCH_CLASSES: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # 12 note fondamentali

PERCUSSIVE_PROGRAMS: Set[int] = {47, 113, 114, 115, 116, 117, 118, 119} # tipicamente usati per traccie ritmiche

'''
def ensure_same_length(midi_obj):
  max_length = max(len(track) for track in midi_obj.instruments)
  for track in midi_obj.instruments:
      if len(track) < max_length:
          track.notes.extend([0] * (max_length - len(track)))
  return midi_obj
'''
def ensure_same_duration(midi_obj):
    max_time = max(note.end for track in midi_obj.instruments for note in track.notes)
    for track in midi_obj.instruments:
        if not track.notes:
            continue
        last_note_end = max(note.end for note in track.notes)
        if last_note_end < max_time:
            # Estendi la durata dell'ultima nota se possibile
            last_note = max(track.notes, key=lambda n: n.end)
            last_note.end = max_time
    return 

def process_midi_file(midi_input_path, output_directory):
    try:
        # 1. CARICAMENTO MIDI
        midi_obj = pretty_midi.PrettyMIDI(midi_input_path)
        base_filename = os.path.basename(midi_input_path)

        # Trovo indici drum
        drum_index = filter_percussive_tracks(midi_obj.instruments, PERCUSSIVE_PROGRAMS)

        # Garantire la stessa lunghezza per tutte le tracks
        ensure_same_duration(midi_obj)

        # 2. ANALISI ARMONICA
        harmonic_progression = analyze_harmonic_progression(midi_obj, drum_index)

        if harmonic_progression:
            harmony_output_path = os.path.join(output_directory, base_filename.replace('.mid','.txt'))
            with open(harmony_output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(harmonic_progression))
            #print(f"Analisi armonica salvata in: '{harmony_output_path}'")
        else:
            #print("Nessuna progressione armonica è stata rilevata.")
            pass

        # 3. ESTRAZIONE MELODIA
        vocal_line = find_vocal_by_keywords(midi_obj, ['vocal', 'vocals', 'voice', 'melody', 'choir'])
        if vocal_line:
            melody_track = monophonize_track(vocal_line)
        else:
            candidate_melody_indices = get_candidate_melody_indices(midi_obj, drum_index, verbose=False)
            # Monophonize, filter, SCORING
            melody_track = get_best_candidate(midi_obj, candidate_melody_indices)

        # 4. SALVATAGGIO MELODIA (con pipeline di pulizia)
        if melody_track:
            # Salvataggio della traccia finale pulita
            final_midi = pretty_midi.PrettyMIDI()
            final_midi.instruments.append(melody_track)
            melody_output_path = os.path.join(output_directory, base_filename)
            final_midi.write(melody_output_path)
            #print(f"Melodia pulita salvata in: '{melody_output_path}'")
        else:
            pass #print("Nessuna traccia melodia trovata per questo file.")
    except Exception as e:
        print(f"Error processing {midi_input_path}: {str(e)}")

if __name__ == "__main__":
    ###### PATH TO MODIFY
    input_directory = '../data/clean_midi'
    output_directory = '../data/Melody_Chords'

    # CICLO SU TUTTI I BRANI
    midi_path = glob(f"{input_directory}/**/*.mid", recursive=True)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #with ProcessPoolExecutor() as exe:
    #    results = list(exe.map(process_midi_file, midi_path, repeat(output_directory)))

    # with ThreadPoolExecutor() as exe:
    #     results = list(exe.map(process_midi_file, midi_path, repeat(output_directory)))
    for midi_file_path in tqdm(midi_path):
        process_midi_file(midi_file_path, output_directory)

        

