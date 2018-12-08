#%% -*- coding: utf-8 -*-

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
                    if float(row[1]) >= chords[index][0] and index != len(chords) - 1:
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
    

    
#%%
try:
    songs = pd.read_pickle("./cross_composer_features_restricted.pkl")
except FileNotFoundError:
    songs = pd.read_csv('./cross-composer_annotations.csv')
    songs['key_string'] = songs.apply(lambda row: key_name_extractor(row),axis=1)
    songs = songs[songs['key_string'] != ""]
    songs['mode'] = songs.apply(lambda row: key_mode_extractor(row),axis=1)
    songs['key'] = songs.apply(lambda row: key_number_extractor(row),axis=1)
    songs['features'] = songs.apply(lambda row: get_inputs_for_rnn(row.Class, row.Filename, -1 * row.key), axis = 1)
    songs['length'] = songs.apply(lambda row: len(row.features), axis = 1)
    songs = songs[songs['length'] > 10 ]
    songs.to_pickle("./cross_composer_features_restricted.pkl")
#%%
class KerasBatchGenerator(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=5):
        self.df = df
        self.data = df.iloc[0].features
        self.mode = df.iloc[0]['mode']
        self.class_num = int(df.iloc[0]['Class'][:2]) - 1
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        self.mode_features = self.get_mode_array()
        #self.class_features = self.get_class_array()
        
    def get_mode_array(self):
        if self.mode == 1:
            mode_arr = np.array([[0,1]])
        else:
            mode_arr = np.array([[1,0]])
        result = np.array(mode_arr)
        for i in range(self.num_steps - 1):
            result = np.concatenate((result,mode_arr),axis=0)
        return result
    
    def get_class_array(self):
        class_arr = np.zeros((1,11))
        class_arr[0][self.class_num] = 1
        result = np.array(class_arr)
        for i in range(self.num_steps - 1):
            result = np.concatenate((result,class_arr),axis=0)
        return result
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, 14))
        y = np.zeros((self.batch_size, 25))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    self.song_idx += 1
                    if(len(self.df) == self.song_idx):
                        self.song_idx = 0
                    self.data = self.df.iloc[self.song_idx].features
                    self.mode = self.df.iloc[self.song_idx]['mode']
                    self.class_num = int(self.df.iloc[self.song_idx]['Class'][:2]) - 1
                    self.mode_features = self.get_mode_array()
                    #self.class_features = self.get_class_array()
                #temp_x = np.concatenate((np.concatenate(((np.array([x_samp[2] for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]])>0.15).astype(int),self.mode_features), axis = 1),self.class_features), axis = 1)
                temp_x = np.concatenate(([x_samp[2] for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]],self.mode_features), axis = 1)
                x[i, :, :] = temp_x
                temp_y = self.data[self.current_idx + self.num_steps][1]
                # convert all of temp_y into a one hot representation
                y[i, :] = to_categorical(temp_y, num_classes = 25)
                self.current_idx += self.skip_step
            yield x, y
            
#%%
#Create training and testing samplesfor dropout in [0]
msk = np.random.rand(len(songs)) 

train = songs[msk < 0.7]
valid = songs[np.logical_and(msk < 0.85, msk >= 0.7)]
test = songs[msk >= 0.85]
#%%
num_steps = 4
hidden_size = 512
batch_size = 25
num_epochs = 80
results=[]
dropout = 0.45
#%%

for dropout in [0.2,0.35,0.5]:
    for hidden_size in [128,256,512]:
    
        print('training for hidden: ' +str(hidden_size) +' dropout: ' + str(dropout) + ' num_steps: '+ str(num_steps))
        train_data_generator = KerasBatchGenerator(train, num_steps, batch_size, skip_step=1)
        validation_data_generator = KerasBatchGenerator(valid, num_steps, batch_size, skip_step=1)
        test_data_generator = KerasBatchGenerator(test, num_steps, batch_size, skip_step=1)
        
        model = Sequential()
        model.add(Dense(hidden_size,input_shape=(num_steps,14)))
        model.add(LSTM(hidden_size, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(LSTM(hidden_size))
        model.add(Dropout(dropout))
        model.add(Dense(25))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        checkpointer = ModelCheckpoint(filepath='./training_checkpoints' + '/model-{epoch:02d}'+'steps=' + str(num_steps)+'hidden='+str(hidden_size) +'dropout='+ str(dropout)+ '.hdf5', verbose=2)
        
        model.fit_generator(train_data_generator.generate(), (sum(train['length'])-train.shape[0] * num_steps)//(batch_size), num_epochs,
                            validation_data=validation_data_generator.generate(),
                            validation_steps=(sum(valid['length'])-valid.shape[0] * num_steps)//(batch_size),
                            callbacks=[checkpointer],verbose=1)
        
        scores = model.evaluate_generator(test_data_generator.generate(), steps=(sum(test['length']) - test.shape[0]*num_steps)//(batch_size), verbose=1)
        results.append(scores[1])


#%%
#model = load_model('./training_checkpoints/model-50.hdf5')
#test_data_generator = KerasBatchGenerator(test, num_steps, batch_size, skip_step=1)
#scores = model.evaluate_generator(test_data_generator.generate(), steps=(sum(test['length']) - test.shape[0]*num_steps)//(batch_size), verbose=2)
#print(scores)
#test_data_generator = KerasBatchGenerator(test, num_steps, batch_size, skip_step=1)
#preds = model.predict_generator(test_data_generator.generate(), steps=sum(test['length'])//(batch_size*num_steps), verbose=2)

