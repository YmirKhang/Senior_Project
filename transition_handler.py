#%% -*- coding: utf-8 -*-
import numpy as np


def get_hmm_parameters():
    lines = [line.rstrip('\n') for line in open('major_transitions')]
    del lines[0]
    start_probs = []
    
    for i in range(60):
        if i%5==0 or i%5==1:
            start_probs.append(float(lines[i]))
    
    del lines[:61]
    start_probs = [prob/sum(start_probs) for prob in start_probs] 
    
    trans_probs = []
    real_trans_probs = []
    for i in range(60):
        trans_probs += lines[60*5*i:60*5*i+120]
        
    for i in range(24):
        for j in range(60):
            if j%5==0 or j%5==1:
                real_trans_probs.append(trans_probs[i*60+j])
                
    real_trans_probs = np.array(real_trans_probs)
    real_trans_probs = real_trans_probs.reshape((24,24))
    real_trans_probs = real_trans_probs.astype(np.float)
    
    for i in range(24):
        real_trans_probs[i] = [trans/sum(real_trans_probs[i]) for trans in real_trans_probs[i]]
    
    return start_probs, real_trans_probs
#%% 