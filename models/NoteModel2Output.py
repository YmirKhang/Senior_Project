# -*- coding: utf-8 -*-


from keras.layers import LSTM, Dropout, Dense, TimeDistributed, BatchNormalization, Input, Lambda
from keras.models import  Model

def get_model_2_output(num_steps, input_size,hidden_size, dropout):
    a = Input(shape = (num_steps, input_size))
    d1 = Dense(hidden_size,input_shape=(num_steps,input_size))(a)
    l1 = LSTM(hidden_size, return_sequences=True)(d1)
    bn = BatchNormalization() (l1)
    do1 = Dropout(dropout)(bn)
    l2 = LSTM(hidden_size, return_sequences=True)(do1)
    do2 = Dropout(dropout)(l2)
    td1 = TimeDistributed(Dense(37, activation = 'sigmoid'))(do2)
    td2 = TimeDistributed(Dense(17, activation = 'softmax'))(td1)
    out1 = Lambda(lambda x:x, name = "notes")(td1)
    out2 = Lambda(lambda x:x, name = "times")(td2)
    model = Model(inputs =a, outputs = [out1 , out2])
    return model