from music21 import converter, instrument, note, chord
import numpy as np
from math import ceil
num_threads = 8
import os
from keras.models import  load_model
import keras.backend as K

note_dict = {
        'C':0,
        'D':2,
        'E':4,
        'F':5,
        'G':7,
        'A':9,
        'B':11
    }

#To extract genres from a list of genres
def genre_extractor(row, lookup_genre):
    for genre in row['genres']:
        if lookup_genre in genre:
            return 1
    return 0

# Custom binary crossentrophy loss
def get_weighted_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)

#Helper function
def string_to_list(row):
    return [s.strip()[1:-1] for s in row['genres'][1:-1].split(",")]

# Function to get numerical note value from text
# This is between C3 and c6
def get_notes_restricted(text, chord_key = 0):
    val = (int(text[-1]) - 3) * 12
    val = val + note_dict[text[0]]
    val = val - chord_key
    if text[1] == '-':
        val = val -1
    elif text[1] == '#':
        val = val + 1
    if val < 0:
        val = val % 12
    while val > 36:
        val = val -12
    return val
    
#This is between C1 and C7
def get_notes(text):
    val = (int(text[-1]) - 1) * 12
    val = val + note_dict[text[0]]
    if text[1] == '-':
        val = val -1
    elif text[1] == '#':
        val = val + 1
    while val > 72:
        val = val - 12
    return val

def find_complete_song_name(artist, song):
    for songname in os.listdir('./clean_midi/' + artist+'/'):
        if song in songname:
            return songname

def midi_to_input(artist, song, key, mode):
    try:
        midi = converter.parse('./clean_midi/' + artist+'/'+ song + '.mid')
    except:
        try:
            song = find_complete_song_name(artist,song)
            midi = converter.parse('./clean_midi/' + artist+'/'+ song )
        except:
            return None
    try:
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        print(song)
        
        if parts: 
            # get the track with the most notes from the instruments
            notes_to_parse = max(parts.parts, key=lambda p: p.__len__()).flat.notes
        else: 
            #single instrument
            notes_to_parse = midi.flat.notes
        transposed = transpose(notes_to_parse, key, mode, notes_to_parse.analyze('ambitus').noteStart, notes_to_parse.analyze('ambitus').noteEnd)
        duration = notes_to_parse.duration.quarterLength
        notes = np.zeros((ceil(duration*4) + 1, 74))
        for element in transposed:
            if isinstance(element, note.Note):
                timestep = int(round(element.offset*4)) 
                notes[timestep, get_notes(element.pitch.nameWithOctave)] = 1
                notes[timestep, 73] = max(notes[timestep, 73], element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                timestep = int(round(element.offset*4)) 
                for part in element:
                    notes[timestep, get_notes(part.pitch.nameWithOctave)] = 1
                notes[timestep, 73] = max(notes[timestep, 73], element.duration.quarterLength)     
        return notes
    except:
        return None
    
def midi_to_input_restricted(artist, song, key, mode, chord_progression=None):
    try:
        midi = converter.parse('./clean_midi/' + artist+'/'+ song + '.mid')
    except:
        try:
            song = find_complete_song_name(artist,song)
            midi = converter.parse('./clean_midi/' + artist+'/'+ song )
        except:
            return None
    try:
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        print(song)
        
        if parts: 
            # get the track with the most notes from the instruments
            notes_to_parse = max(parts.parts, key=lambda p: p.__len__()).flat.notes
        else: 
            #single instrument
            notes_to_parse = midi.flat.notes
        transposed = transpose(notes_to_parse, key, mode, notes_to_parse.analyze('ambitus').noteStart, notes_to_parse.analyze('ambitus').noteEnd)
        duration = notes_to_parse.duration.quarterLength
        notes = np.zeros((ceil(duration*4) + 1, 38))
        for element in transposed:
            if isinstance(element, note.Note):
                timestep = int(round(element.offset*4))
                if chord_progression is not None:
                    predicted_chord = chord_progression[timestep//16]
                    chord_offset = next((i for i, x in enumerate(predicted_chord) if x), None)
                    chord_offset = chord_offset // 2
                    chord_offset %= 12
                else:
                    chord_offset = 0
                notes[timestep, get_notes_restricted(element.pitch.nameWithOctave, chord_offset)] = 1
                notes[timestep, 37] = max(notes[timestep, 37], element.duration.quarterLength)
                if element.duration.quarterLength == 0:
                    print("found 0")
            elif isinstance(element, chord.Chord):
                timestep = int(round(element.offset*4)) 
                for part in element:
                    if chord_progression is not None:
                        predicted_chord = chord_progression[timestep//16]
                        chord_offset = next((i for i, x in enumerate(predicted_chord) if x), None)
                        chord_offset = chord_offset // 2
                        chord_offset %= 12
                    else:
                        chord_offset = 0
                    notes[timestep, get_notes_restricted(part.pitch.nameWithOctave, chord_offset)] = 1
                notes[timestep, 37] = max(notes[timestep, 37], element.duration.quarterLength)     
        return notes
    except Exception as ex:
        print(ex)
        return None
    

def midi_to_input_restricted_time_distributed(artist, song, key, mode, chord_progression=None):
    try:
        midi = converter.parse('./clean_midi/' + artist+'/'+ song + '.mid')
    except:
        try:
            song = find_complete_song_name(artist,song)
            midi = converter.parse('./clean_midi/' + artist+'/'+ song )
        except:
            return None
    try:
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        print(song)
        
        if parts: 
            # get the track with the most notes from the instruments
            notes_to_parse = max(parts.parts, key=lambda p: p.__len__()).flat.notes
        else: 
            #single instrument
            notes_to_parse = midi.flat.notes
        transposed = transpose(notes_to_parse, key, mode, notes_to_parse.analyze('ambitus').noteStart, notes_to_parse.analyze('ambitus').noteEnd)
        duration = notes_to_parse.duration.quarterLength
        notes = np.zeros((ceil(duration*4) + 1, 37))
        for element in transposed:
            if isinstance(element, note.Note):
                timestep = int(round(element.offset*4))
                if chord_progression is not None:
                    predicted_chord = chord_progression[timestep//16]
                    chord_offset = next((i for i, x in enumerate(predicted_chord) if x), None)
                    chord_offset = chord_offset // 2
                    chord_offset %= 12
                else:
                    chord_offset = 0
                for t in range(round(4*element.duration.quarterLength)):
                    if timestep + t < ceil(duration*4) :
                        notes[timestep + t, get_notes_restricted(element.pitch.nameWithOctave, chord_offset)] = 1
                
            elif isinstance(element, chord.Chord):
                timestep = int(round(element.offset*4)) 
                for part in element:
                    if chord_progression is not None:
                        predicted_chord = chord_progression[timestep//16]
                        chord_offset = next((i for i, x in enumerate(predicted_chord) if x), None)
                        chord_offset = chord_offset // 2
                        chord_offset %= 12
                    else:
                        chord_offset = 0
                    for t in range(round(4*element.duration.quarterLength)):
                        if timestep + t < ceil(duration*4):
                            notes[timestep + t, get_notes_restricted(part.pitch.nameWithOctave, chord_offset)] = 1
        return notes
    except Exception as ex:
        print(ex)
        return None

def transpose(note_stream, key, mode,spectrumStart,spectrumEnd):
    start = get_notes(spectrumStart.nameWithOctave)
    if (key < 6) and (key<=start):
        offset = -1 * key 
    else:
        offset = 12 - key
    print(offset)
    return note_stream.transpose(offset)

def lstm_input_from_df(df):
    df['LSTM_input'] = df.apply(lambda row: midi_to_input(row.artist_name, row.song_name, row.key, row.mode), axis = 1)
    
    
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

def get_chord_progressions_with_model(notes, mode):
    cp_model = load_model('./training_checkpoints/model_final.hdf5')
    if mode == MODE_MIN:
        mode_feature= np.array([1,0])
    else:
        mode_feature= np.array([0,1])
    chroma_progression = []
    for i in range(0,len(notes),16):
        chroma = chroma_from_slice(notes[i:i+16])
        chroma_with_feature = np.concatenate((chroma,mode_feature),axis=0)
        chroma_progression.append(chroma_with_feature)
    chord_progression = [max_correl_chord(chord, True) for chord in chroma_progression[:16 - 1]]
    for i in range(0,len(chroma_progression)-16 + 1):
        raw_prediction = cp_model.predict(np.array(chroma_progression[i:i + 16]).reshape(1,16,14))
        prediction = (raw_prediction[0][3] == raw_prediction[0][3].max(axis=0)).astype(int)
        chord_progression.append(prediction)
    return np.array(chord_progression)[:, 16-1, :]

def get_chord_progressions_from_song(notes):
    chroma_progression = []
    for i in range(0,len(notes),16):
        chroma = chroma_from_slice(notes[i:i+16])
        chroma_progression.append(chroma)
    chord_progressions = [max_correl_chord(chord,True) for chord in chroma_progression]
    return chord_progressions
        
    