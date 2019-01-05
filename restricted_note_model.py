# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

num_threads = 8

from keras.layers import LSTM, Dropout, Dense, TimeDistributed, BatchNormalization, Input, Lambda
from keras.models import  load_model, Model
from keras.callbacks import ModelCheckpoint
from Utils.Utils import  midi_to_input, get_chord_progressions_from_song, midi_to_input_restricted_time_distributed, midi_to_input_restricted
from DataGenerators.RestrictedNotes import NoteGenerator3Octaves, NoteGenereatorTimeDistributed
from models.NoteModel2Output import get_model_2_output
from models.NoteModel import get_restricted_model

DURATION_OFFSET = 73

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

restricted_classical_songs = restricted_classical_songs[restricted_classical_songs.artist_name == "Claude Debussy"]

try:
    rcs = pd.read_pickle("./restricted_relative_classical_songs_as_input.pkl")
except:
    restricted_classical_songs['chroma_progressions'] = restricted_classical_songs.apply(lambda row: get_chord_progressions_from_song(row.input_features),axis = 1)
    rcs = restricted_classical_songs   
    rcs['relative_features'] = rcs.apply(lambda row: midi_to_input_restricted(row.artist_name, row.song_name, row.key, row.mode, row.chroma_progressions), axis = 1)
    rcs.to_pickle("./restricted_relative_classical_songs_as_input.pkl")
#%% Split Songs

msk = np.random.rand(len(rcs)) 

train = rcs[msk < 0.8]
valid = rcs[msk >= 0.8]

#%% Build the songs model


num_steps = 64
hidden_size = 256
batch_size = 20
num_epochs = 20
results=[]
dropout = 0.5
relative = True
input_size = 88 if relative else 63
#%%
 
train_data_generator = NoteGenerator3Octaves(train, num_steps, batch_size, skip_step=1,relative=True)
validation_data_generator = NoteGenerator3Octaves(valid, num_steps, batch_size, skip_step=1,relative=True)

model = get_model_2_output(num_steps, input_size, hidden_size, dropout)
#%% Start the training with evaluation after each epoch

model.compile(loss=['binary_crossentropy','categorical_crossentropy'], loss_weights = [30, 1], optimizer='adam', metrics={'notes':'binary_accuracy','times':'categorical_accuracy'})
checkpointer = ModelCheckpoint(filepath='./song_checkpoints/relative_model.hdf5', verbose=2, period=10)

model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                    validation_data=validation_data_generator.generate(),
                    validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                    callbacks=[checkpointer],verbose=1)

#%%
model = load_model('./song_checkpoints/relative_model.hdf5')
#%%


threshold = 0.07
validation_data_generator = NoteGenerator3Octaves(valid, num_steps, 1, skip_step=1,relative=True)

dummy_iters = 1400

for i in range(dummy_iters):
    dummy = next(validation_data_generator.generate())
    
    predictions = []
    true_outputs = []
for i in range(64):
    data = next(validation_data_generator.generate())
    prediction = model.predict(data[0])
    predict_note = prediction[0][0][num_steps - 1]
    true_note = data[1][0][0][num_steps-1]
    predictions.append(np.argwhere(predict_note > threshold))
    true_outputs.append(np.argwhere(true_note > threshold))
    
#%%
rcs['time_distributed_input'] = rcs.apply(lambda row: midi_to_input_restricted_time_distributed(row.artist_name, row.song_name, row.key, row.mode), axis=1)

#%% Split Songs

msk = np.random.rand(len(rcs)) 

train = rcs[msk < 0.8]
valid = rcs[msk >= 0.8]

#%% Build the songs model


num_steps = 64
hidden_size = 256
batch_size = 20
num_epochs = 20
results=[]
dropout = 0.5
relative = True
input_size = 71
#%%
 
train_data_generator = NoteGenereatorTimeDistributed(train, num_steps, batch_size, skip_step=1,relative=True)
validation_data_generator = NoteGenereatorTimeDistributed(valid, num_steps, batch_size, skip_step=1,relative=True)

model = get_restricted_model(num_steps,input_size,hidden_size,dropout)

#%% Start the training with evaluation after each epoch

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
checkpointer = ModelCheckpoint(filepath='./song_checkpoints/time_distributed.hdf5', verbose=2, period=10)

model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                    validation_data=validation_data_generator.generate(),
                    validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                    callbacks=[checkpointer],verbose=1)

#%%
model = load_model('./song_checkpoints/time_distributed.hdf5')
#%%


threshold = 0.12
validation_data_generator = NoteGenereatorTimeDistributed(valid, num_steps, 1, skip_step=1,relative=True)

dummy_iters = 1400

for i in range(dummy_iters):
    dummy = next(validation_data_generator.generate())
    
    predictions = []
    true_outputs = []
for i in range(64):
    data = next(validation_data_generator.generate())
    prediction = model.predict(data[0])
    predict_note = prediction[0][num_steps - 1]
    true_note = data[1][0][num_steps-1]
    predictions.append(np.argwhere(predict_note > threshold))
    true_outputs.append(np.argwhere(true_note > threshold))
    