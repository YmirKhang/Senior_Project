# -*- coding: utf-8 -*-

import numpy as np

class NoteGenereatorTimeDistributed(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=5, relative = False):
        self.df = df
        self.metadata = df.iloc[0]
        self.data = df.iloc[0].time_distributed_input
        self.spotify_features = self.get_spotify_features()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        self.relative = relative
        
    
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
        # input 37 for notes, 9 for spotify features, 25 for chord progressions
        x = np.zeros((self.batch_size, self.num_steps, 71))
        # output 37 for notes, 17 for duration 
        y1 = np.zeros((self.batch_size, self.num_steps, 37))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0
                    self.song_idx += 1
                    if(len(self.df) == self.song_idx):
                        self.song_idx = 0
                    self.data = self.df.iloc[self.song_idx].time_distributed_input    
                    self.metadata = self.df.iloc[self.song_idx]
                    self.spotify_features = self.get_spotify_features()
                temp_x = [np.concatenate((x_samp[:37], self.spotify_features,self.metadata.chroma_progressions[self.current_idx//16]),axis=0) for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]]
                x[i, :, :] = temp_x
                temp_y1 = [ y_samp[:37] for y_samp in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                # convert all of temp_y into a one hot representation
                y1[i, :, :] = temp_y1
                self.current_idx += self.skip_step
            yield x, y1 
            
class NoteGenerator3Octaves(object):
    
    def __init__(self, df, num_steps, batch_size, skip_step=5, relative = False):
        self.df = df
        self.metadata = df.iloc[0]
        if relative:
            self.data = df.iloc[0].relative_features    
        else:
            self.data = df.iloc[0].input_features
        self.spotify_features = self.get_spotify_features()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.song_idx = 0
        self.current_idx = 0
        self.skip_step = skip_step
        self.input_size = 88 if relative else 63
        self.relative = relative
        
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
        x = np.zeros((self.batch_size, self.num_steps, self.input_size))
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
                    if self.relative:
                        self.data = self.df.iloc[self.song_idx].relative_features    
                    else:
                        self.data = self.df.iloc[self.song_idx].input_features
                    self.metadata = self.df.iloc[self.song_idx]
                    self.spotify_features = self.get_spotify_features()
                if self.relative:
                    temp_x = [np.concatenate((x_samp[:37], self.get_duration_feature(x_samp[37]),self.spotify_features,self.metadata.chroma_progressions[self.current_idx//16]),axis=0) for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]]
                else:
                    temp_x = [np.concatenate((x_samp[:37], self.get_duration_feature(x_samp[37]),self.spotify_features),axis=0) for x_samp in self.data[self.current_idx:self.current_idx + self.num_steps]]
                x[i, :, :] = temp_x
                temp_y1 = [ y_samp[:37] for y_samp in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                temp_y2 = [ self.get_duration_feature(y_samp[37]) for y_samp in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                # convert all of temp_y into a one hot representation
                y1[i, :, :] = temp_y1
                y2[i, :, :] = temp_y2
                self.current_idx += self.skip_step
            yield x, [y1,y2]   