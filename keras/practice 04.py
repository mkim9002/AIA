from tensorflow.keras.datasets import reuters
import numpy as np
import  pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#1. data
(x_train, y_train),(x_test,y_test) = reuters.load_data(num_words=10000, test_split=0.2)

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(len(x_train[0]), len(x_train[1]),len(x_train[8981]))
print("뉴스기사의 평균길이 :", max(len(i) for i in x_train))
print("뉴스기사의 평균길이 :", sum(map(len,x_train))/ len(x_train))

print(np.unique(y_train))
print(np.unique(y_test))
print(type(x_train), type(x_test))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
