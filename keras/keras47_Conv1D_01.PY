import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional

#1. DATA

#2. MODEL
model=Sequential()
# model.add(LSTM(10, input_shape = (3,3)))  
model.add(Conv1D(10,2,input_shape = (3,3))) 
model.add(Conv1D(10,2))                    
model.add(Conv1D(10,2, padding='same'))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))