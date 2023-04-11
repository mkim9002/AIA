import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. data
# 1.1 path, path_save, read_csv
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

관광펜션업 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
전국관광지데이터 = pd.read_csv(path + '전국관광지데이터.csv', encoding='CP949')



# 1.2 데이터 범위 설정,전처리
# 1.2.1 drop 설정
관광펜션업_x = np.array(관광펜션업.drop(['지역명','지역명1', '영업상태구분코드'], axis=1))
관광펜션업_y = np.array(관광펜션업['영업상태구분코드'])
전국관광지데이터_x = np.array(전국관광지데이터.drop(['지역명', '지역명1', '영업상태구분코드'], axis=1))
전국관광지데이터_y = np.array(전국관광지데이터['영업상태구분코드'])




# 1.2.4 np.char.replace   astype(str)    .astype(np.float64) 문자를 숫자로 변경
관광펜션업_x = np.char.replace(관광펜션업_x.astype(str), ',', '').astype(np.float64)
관광펜션업_y = np.char.replace(관광펜션업_y.astype(str), ',', '').astype(np.float64)
전국관광지데이터_x = np.char.replace(전국관광지데이터_x.astype(str), ',', '').astype(np.float64)
전국관광지데이터_y = np.char.replace(전국관광지데이터_y.astype(str), ',', '').astype(np.float64)

# 1.2.4 train, test 분리
관광펜션업_x_train, 관광펜션업_x_test, 관광펜션업_y_train, 관광펜션업_y_test,\
전국관광지데이터_x_train, 전국관광지데이터_x_test, 전국관광지데이터_y_train, 전국관광지데이터_y_test \
= train_test_split(관광펜션업_x, 관광펜션업_y, 전국관광지데이터_x, 전국관광지데이터_y,
                    train_size=0.7, shuffle=False)





# # 1.2.5 scaler (0,1로 분리)
# scaler = MinMaxScaler()
# 관광펜션업_x_train = scaler.fit_transform(관광펜션업_x_train)
# 관광펜션업_x_test= scaler.transform(관광펜션업_x_test)
# 전국관광지데이터_x_train = scaler.transform(전국관광지데이터_x_train)
# 전국관광지데이터_x_test = scaler.transform(전국관광지데이터_x_test)

# 1.2.6 timesteps
timesteps = 2

# 1.2.7  split_x 
def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1.2.8 split_x 에 timestep 적용
관광펜션업_x_train_split = split_x(관광펜션업_x_train, timesteps)
관광펜션업_x_test_split = split_x(관광펜션업_x_test, timesteps)
전국관광지데이터_x_train_split = split_x(전국관광지데이터_x_train, timesteps)
전국관광지데이터_x_test_split = split_x(전국관광지데이터_x_test, timesteps)

# 1.2.9 timestep 의 범위 설정 (버려지는 범위 설정 위해 [timesteps:])
관광펜션업_y_train_split = 관광펜션업_y_train[timesteps:]
관광펜션업_y_test_split = 관광펜션업_y_test[timesteps:]
전국관광지데이터_y_train_split = 전국관광지데이터_y_train[timesteps:]
전국관광지데이터_y_test_split = 전국관광지데이터_y_test[timesteps:]

print(관광펜션업_x_train_split.shape)      # (673, 9, 14)
print(전국관광지데이터_x_train_split.shape)      # (673, 9, 13)


# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 14))
dense1 = LSTM(100, activation='relu', name='ss1')(input1)
dense2 = Dense(90, activation='relu', name='ss2')(dense1)
dense3 = Dense(80, activation='relu', name='ss3')(dense2)
output1 = Dense(70, activation='relu', name='ss4')(dense3)

# 2.2 모델2
input2 = Input(shape=(timesteps, 17))
dense11 = LSTM(100, name='hd1')(input2)
dense12 = Dense(90, name='hd2')(dense11)
dense13 = Dense(70, name='hd3')(dense12)
dense14 = Dense(60, name='hd4')(dense13)
output2 = Dense(50, name='output2')(dense14)

# 2.3 머지
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(50, activation='relu', name='mg2')(merge1)
merge3 = Dense(30, activation='relu', name='mg3')(merge2)
hidden_output = Dense(100, name='last')(merge3)

# 2.5 분기1
bungi1 = Dense(30, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

# 2.6 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)
model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, restore_best_weights=True)
hist = model.fit([관광펜션업_x_train_split, 전국관광지데이터_x_train_split], 
                 [관광펜션업_y_train_split, 전국관광지데이터_y_train_split], 
                 epochs=10, batch_size=128, validation_split=0.02, callbacks=[es])

model.save(path_save + 'keras53_관광펜션업2_kmg.h5')

# 4. 평가, 예측

loss = model.evaluate([관광펜션업_x_test_split, 전국관광지데이터_x_test_split], [관광펜션업_y_test_split, 전국관광지데이터_y_test_split])
print('loss : ', loss)

# for_r2 = model.predict([관광펜션업_x_test_split, 전국관광지데이터_x_test_split])
# print(f'결정계수 : {r2_score(관광펜션업_y_test_split,for_r2[0])/2+r2_score(전국관광지데이터_y_test_split,for_r2[1])/2}')

관광펜션업_x_predict = 관광펜션업_x_test[-timesteps:]
관광펜션업_x_predict = 관광펜션업_x_predict.reshape(1, timesteps, 14)
전국관광지데이터_x_predict = 전국관광지데이터_x_test[-timesteps:]
전국관광지데이터_x_predict = 전국관광지데이터_x_predict.reshape(1, timesteps, 17)

predict_result = model.predict([관광펜션업_x_predict, 전국관광지데이터_x_predict])

print("투자가치는?  : ", np.round(predict_result[0], 2))