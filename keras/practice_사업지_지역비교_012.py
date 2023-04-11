
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional
from sklearn.metrics import r2_score, mean_squared_error


# 1. 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '목적지검색건수.csv', encoding='CP949')
df2 = pd.read_csv(path + '관광지출액.csv', encoding='CP949')

print(df1)
print(df2)

# 1-1. 데이터 merge
merged_df = pd.merge(df1, df2, on='시도명')

# 데이터 불러오기
df2 = pd.read_csv(path + '관광지출액.csv', encoding='CP949')

# 필요한 컬럼 추출
df2 = df2[['시도명', '2018관광지출액', '2019관광지출액', '2020관광지출액', '2021관광지출액', '2022관광지출액']]

# 시도명을 인덱스로 설정
df2 = df2.set_index('시도명')

# 증감 계산
df2['2018-2019 증감'] = df2['2019관광지출액'] - df2['2018관광지출액']
df2['2019-2020 증감'] = df2['2020관광지출액'] - df2['2019관광지출액']
df2['2020-2021 증감'] = df2['2021관광지출액'] - df2['2020관광지출액']
df2['2021-2022 증감'] = df2['2022관광지출액'] - df2['2021관광지출액']

# 결과 출력
print(df2)



# 데이터 불러오기
df1 = pd.read_csv(path + '목적지검색건수.csv', encoding='CP949')

# 필요한 컬럼 추출
df1 = df1[['시도명', '2018검색건수', '2019검색건수', '2020검색건수', '2021검색건수', '2022검색건수']]

# 시도명을 인덱스로 설정
df1 = df1.set_index('시도명')

# 증감 계산
df1['2018-2019 증감'] = df1['2019검색건수'] - df1['2018검색건수']
df1['2019-2020 증감'] = df1['2020검색건수'] - df1['2019검색건수']
df1['2020-2021 증감'] = df1['2021검색건수'] - df1['2020검색건수']
df1['2021-2022 증감'] = df1['2022검색건수'] - df1['2021검색건수']

# 결과 출력
print(df1)

