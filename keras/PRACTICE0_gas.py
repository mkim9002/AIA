import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Flatten, MaxPooling2D, Input, Concatenate,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. data
# 1.1 path, path_save, read_csv
path = './_data/개인프로젝트/'
path_save = './_save/개인프로젝트/'

datasets_검색수 = pd.read_csv(path + '한국관광공사_검색수.csv', index_col=0, encoding='cp949')
datasets_방문자수 = pd.read_csv(path + '한국관광공사_방문자수.csv', index_col=0, encoding='cp949')

# 2. 전처리
# 2.1. 두 데이터프레임을 시도명을 기준으로 병합
df = pd.merge(datasets_검색수, datasets_방문자수, on='시도명')

# 2.2. 컬럼명 변경 (년도는 뒤에 두 자리만 사용)
df.columns = ['시도명', 'search_19', 'search_20', 'search_21', 'search_22', 'visit_18', 'visit_19', 'visit_20', 'visit_21', 'visit_22']

# 2.3. 시도명을 인덱스로 변경
df = df.set_index('시도명')

# 3. 모델 구성
# 모델 구성을 위해 필요한 라이브러리 import
from tensorflow.keras.layers import Input, Dense, Concatenate

# 모델 구성
input_ = Input(shape=(10,))
x1 = Dense(64, activation='relu')(input_)
x1 = Dense(32, activation='relu')(x1)
x1 = Dense(16, activation='relu')(x1)

x2 = Conv1D(32, 2, activation='relu', padding='same')(input_)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Conv1D(64, 2, activation='relu', padding='same')(x2)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Flatten()(x2)

concat = Concatenate()([x1, x2])
output = Dense(1)(concat)
model = Model(inputs=input_, outputs=output)

# 4. 컴파일, 훈련
# 컴파일
model.compile(loss='mse', optimizer='adam')

# 훈련 데이터 준비
X = df.values[:, :10] # 검색수와 방문자수
y = df.values[:, -1] # 2022 방문자수

# 훈련
history = model.fit(X, y, epochs=200, verbose=0)

#예측값 구하기
y_pred = model.predict(X)

#도표 만들기
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family']='Malgun Gothic'
plt.figure(figsize=(15,5))
plt.plot(df.index, y, 'o-', label='실제값')
plt.plot(df.index, y_pred, 'o-', label='예측값')
plt.legend()
plt.title('2023년 방문자수 예측')
plt.xlabel('시도명')
plt.ylabel('방문자수')
plt.xticks(rotation=45)
plt.show()

#각 시도별 2023년 방문자수 예측
print(model.predict(X)[-1]) # 제주도의 2023년 방문자수 예측 값 출력