import csv
import numpy as np
import pandas as pd
import re
from keras.utils import to_categorical
from keras.layers import LSTM, Dropout, Dense, Activation, TimeDistributed, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

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

def number_to_chord(num):
    if num == 24:
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
            

def get_inputs_for_rnn(artist, song, transpose_offset):
    chords = []
    print("Processing: " + song)
    with open('chord_features/chords-chordino_' + artist + '.csv', newline='') as csvfile:
        chordreader = csv.reader(csvfile, delimiter = ',')
        song_found = False
        for row in chordreader:
            if song_found == True:
                if row[0] == '':
                    if float(row[1]) == chords[-1:][0]:
                        continue
                    chords.append([float(row[1]), parse_chord(row[2], transpose_offset), np.zeros(12)])
                else:
                    break        
            else:
                if row[0] == artist + '/' + song:
                    song_found = True
                    chords.append([float(row[1]), parse_chord(row[2], transpose_offset), np.zeros(12)])
                    
    with open('chroma_features/chroma-nnls_' + artist + '.csv', newline='') as csvfile:
        chordreader = csv.reader(csvfile, delimiter = ',')
        song_found = False
        index = 0
        for row in chordreader:
            if song_found == True:
                if row[0] == '':
                    if index == len(chords) - 1:
                        chords[index][2] = [val/np.sum(chords[index][2]) for val in chords[index][2]]
                        chords[index][2] = np.roll(chords[index][2] , transpose_offset)
                    elif float(row[1]) >= chords[index + 1][0]:
                        if (np.sum(chords[index][2])) != 0:
                            chords[index][2] = [val/np.sum(chords[index][2]) for val in chords[index][2]]
                            chords[index][2] = np.roll(chords[index][2] , transpose_offset)
                        index += 1
                    chords[index][2] += np.array(row[2:]).astype(np.float)
                else:
                    break        
            else:
                if row[0] == artist + '/' + song:
                    song_found = True
                    chords[index][2] += np.array(row[2:]).astype(np.float)
        if (np.sum(chords[index][2])) != 0:
            chords[index][2] = [val/np.sum(chords[index][2]) for val in chords[index][2]]       
            chords[index][2] = np.roll(chords[index][2] , transpose_offset)
    return chords

def key_number_extractor(row):
    key = row.key_string[0].upper()
    key = note_dict[key]
    flat = re.search('flat', row.key_string)
    sharp = re.search('sharp', row.key_string)
    
    if flat is not None:
        key = key -1
    elif sharp is not None:
        key = key +1
    
    if key < 0:
        key += 12
    elif key > 11:
        key -= 12
    
    return key

def key_mode_extractor(row):
    m = re.search('min', row.key_string)
    if m is None:
        return 1
    else:
        return 0
    
def key_name_extractor(row):
    m = re.search('(a|b|c|d|e|f|g)_*(sharp_*|flat_*)*(min|maj)', row.Filename)
    if m is None:
        return ""
    else:
        return m.group(0)
    
songs = pd.read_csv('./cross-composer_annotations.csv')
songs['key_string'] = songs.apply(lambda row: key_name_extractor(row),axis=1)
songs = songs[songs['key_string'] != ""]
songs['mode'] = songs.apply(lambda row: key_mode_extractor(row),axis=1)
songs['key'] = songs.apply(lambda row: key_number_extractor(row),axis=1)
songs['features'] = songs.apply(lambda row: get_inputs_for_rnn(row.Class, row.Filename, -1 * row.key), axis = 1)
songs['length'] = songs.apply(lambda row: len(row.features), axis = 1)
songs = songs[songs['length'] > 10 ]
songs.to_pickle("./cross_composer_features_restricted.pkl")
