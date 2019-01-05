# -*- coding: utf-8 -*-
import numpy as np
from keras.utils import to_categorical

class CrossComposerGenerator(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=5):
        self.df = df
        self.data = df.iloc[0].features
        self.mode = df.iloc[0]['mode']
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        self.mode_features = self.get_mode_array()
        
    def get_mode_array(self):
        if self.mode == 1:
            mode_arr = np.array([[0,1]])
        else:
            mode_arr = np.array([[1,0]])
        result = np.array(mode_arr)
        for i in range(self.num_steps - 1):
            result = np.concatenate((result,mode_arr),axis=0)
        return result
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, 14))
        y = np.zeros((self.batch_size, self.num_steps, 49))
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
                    self.mode_features = self.get_mode_array()
                temp_x = np.concatenate(([x_samp[2] for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]],self.mode_features), axis = 1)
                x[i, :, :] = temp_x
                temp_y = [y_samp[1] for y_samp in self.data[self.current_idx :self.current_idx + self.num_steps ]]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes = 49)
                self.current_idx += self.skip_step
            yield x, y