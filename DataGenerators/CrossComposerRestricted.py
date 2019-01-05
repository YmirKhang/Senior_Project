# -*- coding: utf-8 -*-
import numpy as np
from keras.utils import to_categorical


class CrossComposerRestrictedModeless(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=1):
        self.df = df
        self.data = df.iloc[0].features
        self.class_num = int(df.iloc[0]['Class'][:2]) - 1
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, 12))
        y = np.zeros((self.batch_size, self.num_steps, 25))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0
                    self.song_idx += 1
                    if(len(self.df) == self.song_idx):
                        self.song_idx = 0
                    self.data = self.df.iloc[self.song_idx].features
                temp_x = [x_samp[2] for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]]
                x[i, :, :] = temp_x
                temp_y = [y_samp[1] for y_samp in self.data[self.current_idx + 1 :self.current_idx + self.num_steps + 1 ]]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes = 25)
                self.current_idx += self.skip_step
            yield x, y


class CrossComposerRestricted(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=1):
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
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, 14))
        y = np.zeros((self.batch_size, self.num_steps, 25))
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
                    #self.class_features = self.get_class_array()
                #temp_x = np.concatenate((np.concatenate(((np.array([x_samp[2] for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]])>0.15).astype(int),self.mode_features), axis = 1),self.class_features), axis = 1)
                modeless_x = np.array([x_samp[2] for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]])
                modeless_x[modeless_x < 0.08] = 0
                print(modeless_x.shape)
                if sum(modeless_x) != 0:
                    modeless_x = [j/sum(modeless_x) for j in modeless_x]
                temp_x = np.concatenate((modeless_x,self.mode_features), axis = 1)
                x[i, :, :] = temp_x
                temp_y = [y_samp[1] for y_samp in self.data[self.current_idx +1 :self.current_idx + self.num_steps +1 ]]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes = 25)
                self.current_idx += self.skip_step
            yield x, y