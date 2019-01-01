from music21 import converter, instrument, note, chord
import pandas as pd
import numpy as np
from math import ceil
from threading import Thread
num_threads = 8
import os
from keras.utils import to_categorical
from keras.layers import LSTM, Dropout, Dense, Activation, TimeDistributed, BatchNormalization, Concatenate, Input, Lambda
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import sys


DURATION_OFFSET = 73
num_steps = 4

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
                    chord = chord_progression[timestep//16]
                    chord_offset = next((i for i, x in enumerate(chord) if x), None)
                    chord_offset %= 12
                else:
                    chord_offset = 0
                notes[timestep, get_notes_restricted(element.pitch.nameWithOctave, chord_offset)] = 1
                notes[timestep, 37] = max(notes[timestep, 37], element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                timestep = int(round(element.offset*4)) 
                for part in element:
                    notes[timestep, get_notes_restricted(part.pitch.nameWithOctave)] = 1
                notes[timestep, 37] = max(notes[timestep, 37], element.duration.quarterLength)     
        return notes
    except:
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


songs = pd.read_csv('./statistics.csv')
songs['genres'] = songs.apply(lambda row: string_to_list(row),axis=1)
genre_list = ['pop','rock','jazz','soul','classical','electronic','dance','metal','disco', 'funk']

for genre in genre_list:
    songs['is_' + genre] = songs.apply(lambda row: genre_extractor(row, genre), axis = 1)
    
classical_songs = songs[songs['is_classical']==1]
classical_songs = classical_songs[classical_songs['is_rock'] == 0]
classical_songs = classical_songs[classical_songs['time_signature'] == 4]

#%% Helper functions

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
    chord_progression = [max_correl_chord(chord, True) for chord in chroma_progression[:num_steps - 1]]
    for i in range(0,len(chroma_progression)-num_steps + 1):
        raw_prediction = cp_model.predict(np.array(chroma_progression[i:i + num_steps]).reshape(1,num_steps,14))
        prediction = (raw_prediction[0][3] == raw_prediction[0][3].max(axis=0)).astype(int)
        chord_progression.append(prediction)
    return np.array(chord_progression)[:, num_steps-1, :]

def get_chord_progressions_from_song(notes):
    chroma_progression = []
    for i in range(0,len(notes),16):
        chroma = chroma_from_slice(notes[i:i+16])
        chroma_progression.append(chroma)
    chord_progressions = [max_correl_chord(chord,True) for chord in chroma_progression]
    return chord_progressions

#%%

def get_chord_progression(feature_matrix, mode):
    return get_chord_progressions_from_song(feature_matrix,mode)

#%%
try:
    classical_songs = pd.read_pickle("./classical_songs_as_input.pkl")
except FileNotFoundError:
    
    classical_songs['input_features'] = classical_songs.apply(lambda row: midi_to_input(row.artist_name, row.song_name, row.key, row.mode), axis = 1)
    classical_songs.to_pickle("./classical_songs_as_input.pkl")
                       
try:
    restricted_classical_songs = pd.read_pickle("./restricted_classical_songs_as_input.pkl")
except FileNotFoundError:
    restricted_classical_songs = classical_songs.copy()
    restricted_classical_songs.drop(['input_features'] , axis = 1)
    restricted_classical_songs['input_features'] = restricted_classical_songs.apply(lambda row: midi_to_input_restricted(row.artist_name, row.song_name, row.key, row.mode), axis = 1)
    restricted_classical_songs.to_pickle("./restricted_classical_songs_as_input.pkl")
print("Finished extracting song input matrices")     
    
#%%
classical_songs = classical_songs[classical_songs['input_features'].isnull() != True] 
classical_songs['length'] = classical_songs.apply(lambda row: len(row.input_features), axis = 1)

restricted_classical_songs = restricted_classical_songs[restricted_classical_songs['input_features'].isnull() != True] 
restricted_classical_songs['length'] = restricted_classical_songs.apply(lambda row: len(row.input_features), axis = 1)



#%%
#songs_as_input[songs_as_input['artist_name']=='Ludwig van Beethoven'] = songs_as_input[songs_as_input['artist_name']=='Ludwig van Beethoven'].apply(lambda row: midi_to_input(row.artist_name, row.song_name, row.key, row.mode), axis = 1)

# Custom binary crossentrophy loss
def get_weighted_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)

class BatcSongGenerator(object):

    def __init__(self, df, num_steps, batch_size, skip_step=5):
        self.df = df
        self.metadata = df.iloc[0]
        self.data = df.iloc[0].input_features
        self.spotify_features = self.get_spotify_features()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        
    def get_duration_feature(self,duration):
        # To represent the duration in a categorical format
        duration = int(min(round(4*duration),16))
        feature = np.zeros(16)
        if duration != 0:
            feature[duration-1] = 1
        return feature
    
    def get_spotify_features(self):
        features = np.zeros(9)
        features[0] = self.metadata.danceability
        features[1] = self.metadata.energy
        features[2] = self.metadata.loudness
        features[3] = self.metadata.speechiness
        features[4] = self.metadata.acousticness
        features[5] = self.metadata.instrumentalness
        features[6] = self.metadata.liveness
        features[7] = self.metadata.valence
        features[8] = self.metadata.tempo
        return features
        
    def generate(self):
        # input 73 for notes, 16 for duration, 9 for spotify features
        x = np.zeros((self.batch_size, self.num_steps, 98))
        # output 73 for notes, 16 for duration 
        y = np.zeros((self.batch_size, self.num_steps, 89))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0
                    self.song_idx += 1
                    if(len(self.df) == self.song_idx):
                        self.song_idx = 0
                    self.data = self.df.iloc[self.song_idx].input_features
                    self.metadata = self.df.iloc[self.song_idx]
                    self.spotify_features = self.get_spotify_features()
                temp_x = [np.concatenate((x_samp[:73], self.get_duration_feature(x_samp[73]),self.spotify_features),axis=0) for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]]
                x[i, :, :] = temp_x
                temp_y = [np.concatenate((y_samp[:73], self.get_duration_feature(y_samp[73])),axis=0) for y_samp in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = temp_y
                self.current_idx += self.skip_step
            yield x, y   
#%% Split Songs

restricted_classical_songs = restricted_classical_songs[restricted_classical_songs.artist_name == "Bach Johann Sebastian"]

#%%

msk = np.random.rand(len(classical_songs)) 

train = classical_songs[msk < 0.8]
valid = classical_songs[msk >= 0.8]

#%% Build the songs model

num_steps = 32
hidden_size = 256
batch_size = 20
num_epochs = 5
results=[]
dropout = 0.5
#%%

train_data_generator = BatcSongGenerator(train, num_steps, batch_size, skip_step=1)

validation_data_generator = BatcSongGenerator(valid, num_steps, batch_size, skip_step=1)
#%%
model = Sequential()
model.add(Dense(hidden_size,input_shape=(num_steps,98)))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(89)))
model.add(Activation('sigmoid'))

#%% Start the training with evaluation after each epoch

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
checkpointer = ModelCheckpoint(filepath='./song_checkpoints/no_chord_progression.hdf5', verbose=1, period=10)

model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                    validation_data=validation_data_generator.generate(),
                    validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                    callbacks=[checkpointer],verbose=1)

# Disble model evaluation for this time as we will be doing model evaluation between epochs
#scores = model.evaluate_generator(test_data_generator.generate(), steps=(sum(test['length']) - test.shape[0]*num_steps)//(batch_size), verbose=1)


#%% Tester
model = load_model('./song_checkpoints/no_chord_progression.hdf5')
#%%
threshold = 0.1
validation_data_generator = BatcSongGenerator(valid, num_steps, 1, skip_step=1)

dummy_iters = 256

for i in range(dummy_iters):
    dummy = next(validation_data_generator.generate())
    
    predictions = []
    true_outputs = []
for i in range(16):
    data = next(validation_data_generator.generate())
    prediction = model.predict(data[0])
    predict_note = prediction[0][num_steps - 1]
    true_note = data[1][0][num_steps-1]
    predictions.append(np.argwhere(predict_note > threshold))
    true_outputs.append(np.argwhere(true_note > threshold))
#%%

class BatchSongGeneratorRestricted(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=5):
        self.df = df
        self.metadata = df.iloc[0]
        self.data = df.iloc[0].input_features
        self.spotify_features = self.get_spotify_features()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        
    def get_duration_feature(self,duration):
        # To represent the duration in a categorical format
        duration = int(min(round(4*duration),16))
        feature = np.zeros(17)
        feature[duration] = 1
        return feature
    
    def get_spotify_features(self):
        features = np.zeros(9)
        features[0] = self.metadata.danceability
        features[1] = self.metadata.energy
        features[2] = self.metadata.loudness
        features[3] = self.metadata.speechiness
        features[4] = self.metadata.acousticness
        features[5] = self.metadata.instrumentalness
        features[6] = self.metadata.liveness
        features[7] = self.metadata.valence
        features[8] = self.metadata.tempo
        return features
        
    def generate(self):
        # input 37 for notes, 17 for duration, 9 for spotify features
        x = np.zeros((self.batch_size, self.num_steps, 63))
        # output 37 for notes, 17 for duration 
        y1 = np.zeros((self.batch_size, self.num_steps, 37))
        y2 = np.zeros((self.batch_size, self.num_steps, 17))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0
                    self.song_idx += 1
                    if(len(self.df) == self.song_idx):
                        self.song_idx = 0
                    self.data = self.df.iloc[self.song_idx].input_features
                    self.metadata = self.df.iloc[self.song_idx]
                    self.spotify_features = self.get_spotify_features()
                temp_x = [np.concatenate((x_samp[:37], self.get_duration_feature(x_samp[37]),self.spotify_features),axis=0) for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]]
                x[i, :, :] = temp_x
                temp_y1 = [ y_samp[:37] for y_samp in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                temp_y2 = [ self.get_duration_feature(y_samp[37]) for y_samp in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                # convert all of temp_y into a one hot representation
                y1[i, :, :] = temp_y1
                y2[i, :, :] = temp_y2
                self.current_idx += self.skip_step
            yield x, [y1,y2]   
#%% Split Songs

msk = np.random.rand(len(restricted_classical_songs)) 

train = restricted_classical_songs[msk < 0.8]
valid = restricted_classical_songs[msk >= 0.8]

#%% Build the songs model

num_steps = 32
hidden_size = 256
batch_size = 20
num_epochs = 5
results=[]
dropout = 0.5
 
train_data_generator = BatchSongGeneratorRestricted(train, num_steps, batch_size, skip_step=1)
validation_data_generator = BatchSongGeneratorRestricted(valid, num_steps, batch_size, skip_step=1)

a = Input(shape = (num_steps, 63))
d1 = Dense(hidden_size,input_shape=(num_steps,63))(a)
l1 = LSTM(hidden_size, return_sequences=True)(d1)
bn = BatchNormalization() (l1)
do1 = Dropout(dropout)(bn)
l2 = LSTM(hidden_size, return_sequences=True)(do1)
do2 = Dropout(dropout)(l2)
td1 = TimeDistributed(Dense(37, activation = 'sigmoid'))(do2)
td2 = TimeDistributed(Dense(17, activation = 'softmax'))(do2)
out1 = Lambda(lambda x:x, name = "notes")(td1)
out2 = Lambda(lambda x:x, name = "times")(td2)
model = Model(inputs =a, outputs = [out1 , out2])

#%% Start the training with evaluation after each epoch

model.compile(loss=['binary_crossentropy','categorical_crossentropy'], loss_weights = [1, 0.2], optimizer='adam', metrics={'notes':'binary_accuracy','times':'categorical_accuracy'})
checkpointer = ModelCheckpoint(filepath='./song_checkpoints/no_chord_progression.hdf5', verbose=2, period=10)

model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                    validation_data=validation_data_generator.generate(),
                    validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                    callbacks=[checkpointer],verbose=1)

#%%

threshold = 0.1
validation_data_generator = BatchSongGeneratorRestricted(valid, num_steps, 1, skip_step=1)

dummy_iters = 512

for i in range(dummy_iters):
    dummy = next(validation_data_generator.generate())
    
    predictions = []
    true_outputs = []
for i in range(16):
    data = next(validation_data_generator.generate())
    prediction = model.predict(data[0])
    predict_note = prediction[0][0][num_steps - 1]
    true_note = data[1][0][0][num_steps-1]
    predictions.append(np.argwhere(predict_note > threshold))
    true_outputs.append(np.argwhere(true_note > threshold))

# TODO will try binary crossentropy
# TODO will try custom crossentopy
# TODO add musescore to project installation
# TODO test if duration is syntactically correct

