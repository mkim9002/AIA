import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, concatenate, Dropout, Bidirectional
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
df2 = pd.read_csv(path + '전국관광지데이터.csv', encoding='CP949')
print(df1)

# 연도별 폐업 건수 계산
폐업추세 = df1.groupby('폐업일자').size().reset_index(name='건수')
print(폐업추세)

# 연도별 폐업 비율 계산
total = 폐업추세['건수'].sum()
폐업추세['비율(%)'] = round(폐업추세['건수'] / total * 100, 2)
# print(폐업추세)

# 2. 전국 관광지 데이터 가공
# 2021-2022 관광지 출액, 검색건수, 방문자수 증가율 계산

# 전체 관광지 데이터에서 2021, 2022 데이터 추출
df2_21 = df2[['2021관광지출액', '2021검색건수', '2021방문자수']].dropna()
df2_22 = df2[['2022관광지출액', '2022검색건수', '2022방문자수']].dropna()

# 2021, 2022 관광지 출액 합계 계산 및 증가율 계산
total_rev_21 = df2_21['2021관광지출액'].sum()
total_rev_22 = df2_22['2022관광지출액'].sum()
rev_increase = round((total_rev_22 - total_rev_21) / total_rev_21 * 100, 2)

# 2021, 2022 검색건수 합계 계산 및 증가율 계산
total_search_21 = df2_21['2021검색건수'].sum()
total_search_22 = df2_22['2022검색건수'].sum()
search_increase = round((total_search_22 - total_search_21) / total_search_21 * 100, 2)

# 2021, 2022 방문자수 합계 계산 및 증가율 계산
total_visit_21 = df2_21['2021방문자수'].sum()
total_visit_22 = df2_22['2022방문자수'].sum()
visit_increase = round((total_visit_22 - total_visit_21) / total_visit_21 * 100, 2)

# 결과 출력
# print('2021-2022 관광지 출액 증가율: {}%'.format(rev_increase))
# print('2021-2022 검색건수 증가율: {}%'.format(search_increase))
# print('2021-2022 방문자수 증가율: {}%'.format(visit_increase))

# 데이터 범위
# 데이터 범위
df1_x = np.array(df1.drop(['지역명'], axis=1))
df1_y = np.array(df1['폐업추세']).astype(np.float64)
df2_x = np.array(df2.drop(['지역명'], axis=1))
df2_y = np.array(df2['폐업추세']).astype(np.float64)

# 데이터 결합
df = pd.concat([df1, df2], axis=1)
df_x = np.array(df.drop(['지역명', '폐업추세'], axis=1))
df_y = np.array(df['폐업추세']).astype(np.float64)
print(df_y)


# train, test 분리
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, train_size=0.7, shuffle=False)

# timesteps
timesteps = 3

# split_x 
def split_x(dt, st):
    a = []
    for i in range(len(dt)-st+1):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1.2.8 split_x 에 timestep 적용
df_x_train_split = split_x(df_x_train, timesteps)
df_x_test_split = split_x(df_x_test, timesteps)
print(df_x_train.shape)
print(df_x_test.shape)
print(df_y_train.shape)
print(df_y_test.shape)


print(df)
df2_y_train_split = split_x(df2_y_train, timesteps)
df2_y_test_split = split_x(df2_y_test, timesteps)


print(df1_x_train_split.shape)      # (679, 3, 15)
print(df2_x_train_split.shape)      # (679, 3, 0)

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 15))
dense1 = LSTM(90, activation='relu', name='ss1')(input1)
dense2 = Dense(80, activation='relu', name='ss2')(dense1)
dense3 = Dense(70, activation='relu', name='ss3')(dense2)
output1 = Dense(50, activation='relu', name='ss4')(dense3)

# 2.2 모델2
input2 = Input(shape=(timesteps, 15))
dense11 = LSTM(90, name='hd1')(input2)
dense12 = Dense(80, name='hd2')(dense11)
dense13 = Dense(70, name='hd3')(dense12)
dense14 = Dense(60, name='hd4')(dense13)
output2 = Dense(50, name='output2')(dense14)

# 2.3 머지
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(50, activation='relu', name='mg2')(merge1)
merge3 = Dense(50, activation='relu', name='mg3')(merge2)
hidden_output = Dense(50, name='last')(merge3)

# 2.5 분기1
bungi1 = Dense(30, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

# 2.6 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)
model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

print(df1_x_train_split)
print(df2_x_train_split)
print(df1_x_train_split)
print(df1_x_train_split)
print(df1_x_train_split)
print(df1_x_train_split)
print(df1_x_train_split)
print(df1_x_train_split)

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
hist = model.fit([df1_x_train_split, df2_x_train_split], 
                 [df1_y_train_split, df2_y_train_split], 
                 epochs=20, batch_size=128, validation_split=0.2, callbacks=[es])

model.save(path_save + '폐업율전망.h5') 

# 4. 평가, 예측

loss = model.evaluate([df1_x_test_split, df2_x_test_split], [df2_y_test_split, df2_y_test_split])
print('loss : ', loss)







