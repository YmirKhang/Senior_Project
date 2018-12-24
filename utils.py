import numpy as np
from keras.models import Sequential, load_model
MODE_MAJ = 1 
MODE_MIN = 0

note_dict = {
        'Cb':11,
        'C':0,
        'C#':1,
        'Db':1,
        'D':2,
        'D#':3,
        'Eb':3,
        'E':4,
        'E#':5,
        'Fb':4,
        'F':5,
        'F#':6,
        'Gb':6,
        'G':7,
        'G#':8,
        'Ab':8,
        'A':9,
        'A#':10,
        'Bb':10,
        'B':11,
        'B#':0
}

number_dict = {
        0:'C',
        1:'C#',
        2:'D',
        3:'D#',
        4:'E',
        5:'F',
        6:'F#',
        7:'G',
        8:'G#',
        9:'A',
        10:'A#',
        11:'B'
}

chroma_method = 0

Major_Chroma = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0] if chroma_method == 0 else [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
Minor_Chroma = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0] if chroma_method == 0 else [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

Major_Chroma = [i/sum(Major_Chroma) for i in Major_Chroma]
Minor_Chroma = [i/sum(Minor_Chroma) for i in Minor_Chroma]

Chroma_Templates = []
for i in range(12):
    template = Major_Chroma[-i:] +  Major_Chroma[:-i]
    Chroma_Templates.append(template)
    template = Minor_Chroma[-i:] +  Minor_Chroma[:-i]
    Chroma_Templates.append(template)    

Chroma_Templates = np.matrix(Chroma_Templates)

def number_to_chord(num):
    if num == 48:
        return 'N'
    key = number_dict[num//4]
    offset = num % 4
    if offset == 0:
        return key + "MAJ"
    if offset == 1:
        return key + "MIN"
    if offset == 2:
        return key + "AUG" # Can be maj
    if offset == 3:
        return key + "DIM" # Can be min
    

def parse_chord(text, offset):
    if text[0].upper() == 'N':
        return 24
    else:
        tokens = text.split('_')
        key = (note_dict[tokens[0]] + offset) % 12
        mode = tokens[1].upper()
        if mode == 'MAJ':
            return key * 2
        elif mode == 'MIN':
            return key * 2 + 1
        elif mode == 'AUG':
            return key * 2 
        elif mode == 'DIM':
            return key * 2 + 1 
        
def parse_chord_restricted(text, offset):
    if text[0].upper() == 'N':
        return 24
    else:
        tokens = text.split('_')
        key = (note_dict[tokens[0]] + offset) % 12
        mode = tokens[1].upper()
        if mode == 'MAJ':
            return key * 2
        elif mode == 'MIN':
            return key * 2 + 1
        elif mode == 'AUG':
            return key * 2 
        elif mode == 'DIM':
            return key * 2 + 1 
        
def chroma_from_slice(note_slice):
    vector = [0,0,0,0,0,0,0,0,0,0,0,0]
    for timestep in note_slice:
        for note in range(73):
            if(timestep[note]!=0):
                vector[note%12] += 1
    if sum(vector) != 0:
        return np.array([note/sum(vector) for note in vector])
    else:
        return np.array(vector)
    
def max_correl_chord(chord,one_hot=False):
    chord = chord[:12]
    max_correl = 0 
    max_index = 0
    for i in range(24):
        if np.corrcoef(chord,Chroma_Templates[i])[0][1] > max_correl:
            max_index = i
            max_correl = np.corrcoef(chord,Chroma_Templates[i])[0][1]
    mode = "Maj" if max_index % 2 == 0 else "Min"
    note = number_dict[int(max_index / 2)]
    if sum(chord) == 0:
        max_index=24
    if not one_hot:
        return note + mode
    else:
        result = np.zeros((25))
        result[max_index] = 1
        return result

def chromas_from_midi(midi):
    chromas = []
    for i in range(0,len(midi),16):
        chromas.append(chroma_from_slice(midi[i:i+16]))
    return chromas

def get_none_chord_array():
    arr = np.zeros(25)
    arr[24] = 1
    return arr

def get_chord_progressions_from_song(notes, mode):
    cp_model = load_model('./training_checkpoints/final_model.hdf5')
    if mode == MODE_MIN:
        mode_feature= np.array([1,0])
    else:
        mode_feature= np.array([0,1])
    chroma_progression = []
    for i in range(0,len(notes),16):
        chroma_progression.append(np.concatenate((chroma_from_slice(notes[i:i+16]),mode_feature),axis=1))
    chord_progression = np.array([max_correl_chord(chord,False) for chord in chroma_progression[:3]])
    predicted_chords = cp_model.predict(chroma_progression)
    chord_progression.append(predicted_chords)
    chord_progression.append(get_none_chord_array())
    return chord_progression
    
    
    
    
        
    