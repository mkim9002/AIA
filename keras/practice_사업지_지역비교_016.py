import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pylab 


# 1. data
# 1.1 path, path_save, read_csv
path = './_data/개인프로젝트/'
path_save = './_save/개인프로젝트/'

datasets_검색수 = pd.read_csv(path + '한국관광공사_검색수.csv', index_col=0, encoding='cp949')
datasets_방문자수 = pd.read_csv(path + '한국관광공사_방문자수.csv', index_col=0, encoding='cp949')

print(datasets_검색수.shape, datasets_방문자수.shape)  #(17, 5) (17, 5)
print(datasets_검색수.columns, datasets_방문자수.columns) #Index(['2018', '2019', '2020', '2021', '2022'], dtype='object') Index(['2018', '2019', '2020', '2021', '2022'], dtype='object')
print(datasets_검색수.info(), datasets_방문자수.info())
print(datasets_검색수.describe(), datasets_방문자수.describe())
print(type(datasets_검색수), type(datasets_방문자수))

#2. 전처리
#2.1. 두 데이터프레임을 시도명을 기준으로 병합
df = pd.merge(datasets_검색수, datasets_방문자수, on='시도명')

#2.2. 컬럼명 변경 (년도는 뒤에 두 자리만 사용)
df.columns = ['시도명', 'search_19', 'search_20', 'search_21', 'search_22',
'visit_18', 'visit_19', 'visit_20', 'visit_21', 'visit_22']

#2.3. 시도명을 인덱스로 변경
df = df.set_index('시도명')

#3. 모델 구성

#모델 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(9,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#4. 컴파일, 훈련
#컴파일
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

#훈련 데이터 준비
X = df.values[:, :10] # 검색수와 방문자수
y = df.values[:, -1] # 2022 방문자수

print(X.shape) #(17, 9)
print(y.shape) #(17,)


#훈련
model.fit(X, y, epochs=200)

#5. 예측 값 만들기 및 도표 만들기
#예측값 구하기
y_pred = model.predict(X)
print(y_pred)
