# -*- coding: utf-8 -*-
import numpy as np

class BatchNoteGenerator(object):

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