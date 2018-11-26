#%%

import pandas as pd
from main import midi_to_input
import numpy as np
from transition_handler import get_hmm_parameters 
from hmmlearn.hmm import GaussianHMM

num_chords = 24
num_emissions = 12

chroma_method = 0

Major_Chroma = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0] if chroma_method == 0 else [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
Minor_Chroma = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0] if chroma_method == 0 else [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


Major_Chroma = [i/sum(Major_Chroma) for i in Major_Chroma]
Minor_Chroma = [i/sum(Minor_Chroma) for i in Minor_Chroma]
Major_Covar_Mat = np.zeros((num_emissions,num_emissions))
Minor_Covar_Mat = np.zeros((num_emissions,num_emissions))

for i in range(12):
    Major_Covar_Mat[i][i] = 0.2
    Minor_Covar_Mat[i][i] = 0.2

Major_Covar_Mat[0][0] = 1
Major_Covar_Mat[4][4] = 1
Major_Covar_Mat[7][7] = 1
Major_Covar_Mat[0][4] = 0.6
Major_Covar_Mat[4][0] = 0.6
Major_Covar_Mat[0][7] = 0.8
Major_Covar_Mat[7][0] = 0.8
Major_Covar_Mat[4][7] = 0.8
Major_Covar_Mat[7][4] = 0.8

Minor_Covar_Mat[0][0] = 1
Minor_Covar_Mat[3][3] = 1
Minor_Covar_Mat[7][7] = 1
Minor_Covar_Mat[0][3] = 0.6
Minor_Covar_Mat[3][0] = 0.6
Minor_Covar_Mat[0][7] = 0.8
Minor_Covar_Mat[7][0] = 0.8
Minor_Covar_Mat[3][7] = 0.8
Minor_Covar_Mat[7][3] = 0.8

covariance_matrix = np.zeros((num_chords,num_emissions,num_emissions))

for i in range(12):
    covariance_matrix[2*i] = np.roll(np.roll(Major_Covar_Mat,i,axis=0),i,axis=1)
    covariance_matrix[2*i+1] = np.roll(np.roll(Minor_Covar_Mat,i,axis=0),i,axis=1)


Chroma_Templates = []
for i in range(12):
    template = Major_Chroma[-i:] +  Major_Chroma[:-i]
    Chroma_Templates.append(template)
    template = Minor_Chroma[-i:] +  Minor_Chroma[:-i]
    Chroma_Templates.append(template)    

Chroma_Templates = np.matrix(Chroma_Templates)

#%%


note_dict = {
        'C':0,
        'D':2,
        'E':4,
        'F':5,
        'G':7,
        'A':9,
        'B':11
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

def max_correl_chord(chord):
    chord = chroma_from_slice(chord)
    max_correl = 0 
    max_index = 0
    for i in range(24):
        if np.corrcoef(chord,Chroma_Templates[i])[0][1] > max_correl:
            max_index = i
            max_correl = np.corrcoef(chord,Chroma_Templates[i])[0][1]
    mode = "Maj" if max_index % 2 == 0 else "Min"
    note = number_dict[int(max_index / 2)]
    return note + mode

def index_to_chord(index):
    mode = "Maj" if index % 2 == 0 else "Min"
    note = number_dict[int(index / 2)]
    return note + mode

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

def chroma_from_slice(note_slice):
    vector = [0,0,0,0,0,0,0,0,0,0,0,0]
    for timestep in note_slice:
        for note in range(73):
            if(timestep[note]!=0):
                vector[note%12] += 1
    if sum(vector) != 0:
        return [note/sum(vector) for note in vector]
    else:
        return vector
    
def most_likely_from_midi(midi):
    likely_chords = []
    for i in range(0,len(midi),16):
        likely_chords.append(max_correl_chord(midi[i:i+16]))
    return likely_chords

def chromas_from_midi(midi):
    chromas = []
    for i in range(0,len(midi),16):
        chromas.append(chroma_from_slice(midi[i:i+16]))
    return chromas
#%%

#%%

start_probs, transition_matrix = get_hmm_parameters()


markov_model = GaussianHMM(n_components=24,
                           covariance_type="full",
                           init_params="stmc")

markov_model.startprob_ = start_probs
markov_model.transmat_ = transition_matrix
markov_model.n_features = 12
markov_model.means_ = Chroma_Templates
markov_model.covars_ = covariance_matrix

path = markov_model.predict(emissions,[len(emissions)])

chords = [index_to_chord(chord) for chord in path]
probable_chords = most_likely_from_midi(notes)
#%%
