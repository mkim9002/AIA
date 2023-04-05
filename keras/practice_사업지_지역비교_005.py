import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 불러오기
df1 = pd.read_csv('관광지출액.csv', encoding='CP949')
df2 = pd.read_csv('목적지검색건수.csv', encoding='CP949')
df3 = pd.read_csv('관광펜션업.csv', encoding='CP949')
df4 = pd.read_csv('방문자수.csv', encoding='CP949')

# 결측치 처리
df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df4 = df4.dropna()

# 데이터 정규화
scaler = MinMaxScaler()
df1[['관광지출액']] = scaler.fit_transform(df1[['관광지출액']])
df2[['검색량']] = scaler.fit_transform(df2[['검색량']])
df3[['숙박시설수']] = scaler.fit_transform(df3[['숙박시설수']])
df4[['방문자수']] = scaler.fit_transform(df4[['방문자수']])

# 데이터 합치기
df = pd.concat([df1['관광지출액'], df2['검색량'], df3['숙박시설수'], df4['방문자수']], axis=1)

# 학습 데이터 생성
train = df.iloc[:-1,:]
train_label = df.iloc[1:,:]

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(4,1)))
model.add(Dense(4))

# 모델 컴파일 및 학습
model.compile(loss='mse', optimizer='adam')
model.fit(train.values.reshape(-1,4,1), train_label.values, epochs=100, batch_size=1, verbose=2)

# 마지막 데이터로 추론하여 최적의 투자 지역 도출
prediction = model.predict(np.array(df.iloc[-1,:]).reshape(1,4,1))
investment_area = df.index[np.argmax(prediction)]

print("최적의 투자 지역은 {} 입니다.".format(investment_area))