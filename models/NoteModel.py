# -*- coding: utf-8 -*-

from keras.layers import LSTM, Dropout, Dense, Activation, TimeDistributed, BatchNormalization, Lambda, Input
from keras.models import Sequential, Model

def get_lstm_model_for_notes(hidden_size,num_steps,dropout):
    model = Sequential()
    model.add(Dense(hidden_size,input_shape=(num_steps,98)))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(89)))
    model.add(Activation('sigmoid'))
    return model

def get_restricted_model(num_steps, input_size, hidden_size, dropout):
    a = Input(shape = (num_steps, 71))
    d1 = Dense(hidden_size,input_shape=(num_steps,input_size))(a)
    l1 = LSTM(hidden_size, return_sequences=True)(d1)
    bn = BatchNormalization() (l1)
    do1 = Dropout(dropout)(bn)
    l2 = LSTM(hidden_size, return_sequences=True)(do1)
    do2 = Dropout(dropout)(l2)
    td1 = TimeDistributed(Dense(37, activation = 'sigmoid'))(do2)
    out1 = Lambda(lambda x:x, name = "notes")(td1)    
    model = Model(inputs =a, outputs = out1)
    return model