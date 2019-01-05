import pandas as pd
import numpy as np
num_threads = 8
from keras.models import  load_model
from keras.callbacks import ModelCheckpoint
from Utils.Utils import genre_extractor, string_to_list, midi_to_input, midi_to_input_restricted, get_chord_progressions_from_song
from DataGenerators.Notes import BatchNoteGenerator
from models.NoteModel import get_lstm_model_for_notes

DURATION_OFFSET = 73
num_steps = 4

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

train_data_generator = BatchNoteGenerator(train, num_steps, batch_size, skip_step=1)

validation_data_generator = BatchNoteGenerator(valid, num_steps, batch_size, skip_step=1)
#%%
model = get_lstm_model_for_notes(hidden_size,num_steps,dropout)

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
validation_data_generator = BatchNoteGenerator(valid, num_steps, 1, skip_step=1)

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
