from music21 import converter, instrument, note, chord
import pandas as pd
import numpy as np
from math import ceil
from threading import Thread
num_threads = 8
import os
from keras.utils import to_categorical
from keras.layers import LSTM, Dropout, Dense, Activation, TimeDistributed, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K


DURATION_OFFSET = 73

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

def get_chord_progressions_from_song(notes, mode):
    cp_model = load_model('./training_checkpoints/model_final.hdf5')
    if mode == MODE_MIN:
        mode_feature= np.array([1,0])
    else:
        mode_feature= np.array([0,1])
    chroma_progression = []
    for i in range(0,len(notes),16):
        chroma_progression.append(np.concatenate((chroma_from_slice(notes[i:i+16]),mode_feature),axis=1))
    chord_progression = np.array([max_correl_chord(chord,False) for chord in chroma_progression[:3]])
    predicted_chords = cp_model.predict(chroma_progression,batch_size = len(chroma_progression) - 1)
    chord_progression.append(predicted_chords)
    chord_progression.append()
    return chord_progression

#%%

def get_chord_progression(feature_matrix, mode):
    return get_chord_progressions_from_song(feature_matrix,mode)

#%%
try:
    classical_songs = pd.read_pickle("./classical_songs_as_input.pkl")
except FileNotFoundError:
    
    classical_songs['input_features'] = classical_songs.apply(lambda row: midi_to_input(row.artist_name, row.song_name, row.key, row.mode), axis = 1)
    classical_songs.to_pickle("./classical_songs_as_input.pkl")
                       
print("Finished extracting song input matrices")     
    
#%%
classical_songs = classical_songs[classical_songs['input_features'].isnull() != True] 
classical_songs['length'] = classical_songs.apply(lambda row: len(row.input_features), axis = 1)
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
        return np.array([int(char) for char in bin(duration)[2:].zfill(5)])
    
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
        # input 73 for notes, 5 for duration, 9 for spotify features
        x = np.zeros((self.batch_size, self.num_steps, 87))
        # output 73 for notes, 5 for duration 
        y = np.zeros((self.batch_size, self.num_steps, 78))
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

msk = np.random.rand(len(classical_songs)) 

train = classical_songs[msk < 0.8]
valid = classical_songs[msk >= 0.8]

#%% Build the songs model

num_steps = 16
hidden_size = 128
batch_size = 20
num_epochs = 20
results=[]
dropout = 0.5
 
train_data_generator = BatcSongGenerator(train, num_steps, batch_size, skip_step=1)
validation_data_generator = BatcSongGenerator(valid, num_steps, batch_size, skip_step=1)

model = Sequential()
model.add(Dense(hidden_size,input_shape=(num_steps,87)))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(78)))
model.add(Activation('softmax'))

#%% Start the training with evaluation after each epoch

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
checkpointer = ModelCheckpoint(filepath='./song_checkpoints/no_chord_progression.hdf5', verbose=2, period=10)

model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                    validation_data=validation_data_generator.generate(),
                    validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                    callbacks=[checkpointer],verbose=1)

# Disble model evaluation for this time as we will be doing model evaluation between epochs
#scores = model.evaluate_generator(test_data_generator.generate(), steps=(sum(test['length']) - test.shape[0]*num_steps)//(batch_size), verbose=1)

#%%

class BatcSongGeneratorWithCP(object):
    
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
        return np.array([int(char) for char in bin(duration)[2:].zfill(5)])
    
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
        features[8] = self.metadata.mode
        return features
        
    def generate(self):
        # input 73 for notes, 5 for duration, 10 for spotify features, and 25 for chord progressions
        x = np.zeros((self.batch_size, self.num_steps, 113))
        # output 73 for notes, 5 for duration 
        y = np.zeros((self.batch_size, self.num_steps, 78))
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

msk = np.random.rand(len(classical_songs)) 

train = classical_songs[msk < 0.8]
valid = classical_songs[msk >= 0.8]

#%% Build the songs model

num_steps = 16
hidden_size = 128
batch_size = 20
num_epochs = 20
results=[]
dropout = 0.5
 
train_data_generator = BatcSongGeneratorWithCP(train, num_steps, batch_size, skip_step=1)
validation_data_generator = BatcSongGeneratorWithCP(valid, num_steps, batch_size, skip_step=1)

model = Sequential()
model.add(Dense(hidden_size,input_shape=(num_steps,87)))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(78)))
model.add(Activation('softmax'))

#%% Start the training with evaluation after each epoch

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
checkpointer = ModelCheckpoint(filepath='./song_checkpoints/no_chord_progression.hdf5', verbose=2, period=10)

model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                    validation_data=validation_data_generator.generate(),
                    validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                    callbacks=[checkpointer],verbose=1)

