import string
import numpy as np

## code by https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result

def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    # all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]

def extract_melody(piano_roll):
    melody = []
    for frame in piano_roll:
        on_notes = np.where(frame > 0)[0]
        if len(on_notes) > 0:
            # Pick highest-pitch note
            note = on_notes[-1] #+ 21  # MIDI note number (21–108)
        else:
            note = 0  
        melody.append(note)
    return np.array(melody)

def downsample_piano_roll(piano_roll, ticks_per_beat, unit_fraction=1/8):
    ticks_per_unit = int(ticks_per_beat * unit_fraction)
    T, N = piano_roll.shape
    T_trimmed = (T // ticks_per_unit) * ticks_per_unit
    piano_roll = piano_roll[:T_trimmed]
    piano_roll = piano_roll.reshape(-1, ticks_per_unit, N)
    downsampled = piano_roll.max(axis=1)  # preserve velocity if any note was on
    return downsampled

def mid2matrix(mid, unit_fraction = 1/8):
    result_array = mid2arry(mid)
    ticks_per_beat = mid.ticks_per_beat
    downsampled_array = downsample_piano_roll(result_array, ticks_per_beat, unit_fraction=unit_fraction)
    melody = extract_melody(downsampled_array)
    X = np.zeros((np.max(melody), len(melody)))
    for i in np.arange(len(melody)):
        X[melody[i]-1,i]=1
    return X

def tokenize(X, lentoken):
    tokens = []
    num_tokens = X.shape[1] % lentoken
    for i in np.arange(num_tokens):
        tokens.append(X[:,i*lentoken : (i+1)*lentoken ])
    return tokens