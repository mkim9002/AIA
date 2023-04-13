import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score



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

#6. 모델 평가
#테스트 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#정규화
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#모델 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(9,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#컴파일
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

#훈련
early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
model_check = ModelCheckpoint(path_save + 'best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1, callbacks=[early_stop, model_check], validation_split=0.2)

#예측
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#R2, RMSE 계산
print('Train R2 score :', r2_score(y_train, y_pred_train))
print('Test R2 score :', r2_score(y_test, y_pred_test))

print('Train RMSE score :', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('Test RMSE score :', np.sqrt(mean_squared_error(y_test, y_pred_test)))

#accuracy score 계산
y_pred_train_round = np.round(y_pred_train)
y_pred_test_round = np.round(y_pred_test)

print('Train Accuracy score :', accuracy_score(y_train, y_pred_train_round))
print('Test Accuracy score :', accuracy_score(y_test, y_pred_test_round))

#도표 만들기
plt.figure(figsize=(15,5))
plt.plot(df.index, y, 'o-', label='실제값')
plt.plot(df.index, y_pred_train, 'o-', label='훈련 예측값')
plt.plot(df.index, y_pred_test, 'o-', label='테스트 예측값')
plt.legend()
plt.title('2023년 방문자수 예측')
plt.xlabel('시도명')
plt.ylabel('방문자수')
plt.xticks(rotation=45)
plt.show()