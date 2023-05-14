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
path = 'c:/study/_data/baseball/'
path_save = 'c:/study/_save/baseball/'

datasets_doosan = pd.read_csv(path + 'doosan_data.csv', index_col=0, encoding='cp949')
datasets_lg = pd.read_csv(path + 'lg_data.csv', index_col=0, encoding='cp949')



# 1.2 데이터 범위 설정,전처리
# 1.2.1 drop 설정
doosan_x = np.array(datasets_doosan.drop(['전일비', 'SCORE'], axis=1))
doosan_y = np.array(datasets_doosan['SCORE'])
lg_x = np.array(datasets_lg.drop(['전일비', 'SCORE'], axis=1))
lg_y = np.array(datasets_lg['SCORE'])

# 1.2.2 범위 선택
doosan_x = doosan_x[:180, :]
doosan_y = doosan_y[:180]
lg_x = lg_x[:180, :]
lg_y = lg_y[:180]

#1.2.3 np.flip으로 전체 순서 반전
doosan_x = np.flip(doosan_x, axis=1)
doosan_y = np.flip(doosan_y)
lg_x = np.flip(lg_x, axis=1)
lg_y = np.flip(lg_y)

print(doosan_x.shape, doosan_y.shape)
print(lg_x.shape, lg_y.shape)

# 1.2.4 np.char.replace   astype(str)    .astype(np.float64) 문자를 숫자로 변경
doosan_x = np.char.replace(doosan_x.astype(str), ',', '').astype(np.float64)
doosan_y = np.char.replace(doosan_y.astype(str), ',', '').astype(np.float64)
lg_x = np.char.replace(lg_x.astype(str), ',', '').astype(np.float64)
lg_y = np.char.replace(lg_y.astype(str), ',', '').astype(np.float64)

# 1.2.4 train, test 분리
doosan_x_train, doosan_x_test, doosan_y_train, doosan_y_test,\
lg_x_train, lg_x_test, lg_y_train, lg_y_test \
= train_test_split(doosan_x, doosan_y, lg_x, lg_y,
                    train_size=0.7, shuffle=False)

# 1.2.5 scaler (0,1로 분리)
scaler = MinMaxScaler()
doosan_x_train = scaler.fit_transform(doosan_x_train)
doosan_x_test= scaler.transform(doosan_x_test)
lg_x_train = scaler.transform(lg_x_train)
lg_x_test = scaler.transform(lg_x_test)

# 1.2.6 timesteps
timesteps = 9

# 1.2.7  split_x 
def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1.2.8 split_x 에 timestep 적용
doosan_x_train_split = split_x(doosan_x_train, timesteps)
doosan_x_test_split = split_x(doosan_x_test, timesteps)
lg_x_train_split = split_x(lg_x_train, timesteps)
lg_x_test_split = split_x(lg_x_test, timesteps)

# 1.2.9 timestep 의 범위 설정 (버려지는 범위 설정 위해 [timesteps:])
doosan_y_train_split = doosan_y_train[timesteps:]
doosan_y_test_split = doosan_y_test[timesteps:]
lg_y_train_split = lg_y_train[timesteps:]
lg_y_test_split = lg_y_test[timesteps:]

print(doosan_x_train_split.shape)      # 
print(lg_x_train_split.shape)      # 

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 8))
dense1 = LSTM(100, activation='relu', name='ss1')(input1)
dense2 = Dense(90, activation='relu', name='ss2')(dense1)
dense3 = Dense(80, activation='relu', name='ss3')(dense2)
output1 = Dense(70, activation='relu', name='ss4')(dense3)

# 2.2 모델2
input2 = Input(shape=(timesteps, 8))
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
hist = model.fit([doosan_x_train_split, lg_x_train_split], 
                 [doosan_y_train_split, lg_y_train_split], 
                 epochs=1000, batch_size=128, validation_split=0.02, callbacks=[es])

model.save(path_save + 'keras53_doosan2_kmg.h5')

# 4. 평가, 예측

loss = model.evaluate([doosan_x_test_split, lg_x_test_split], [doosan_y_test_split, lg_y_test_split])
print('loss : ', loss)

# for_r2 = model.predict([doosan_x_test_split, lg_x_test_split])
# print(f'결정계수 : {r2_score(doosan_y_test_split,for_r2[0])/2+r2_score(lg_y_test_split,for_r2[1])/2}')

doosan_x_predict = doosan_x_test[-timesteps:]
doosan_x_predict = doosan_x_predict.reshape(1, timesteps, 8)
lg_x_predict = lg_x_test[-timesteps:]
lg_x_predict = lg_x_predict.reshape(1, timesteps, 8)

predict_result = model.predict([doosan_x_predict, lg_x_predict])

print("SCORE : ", np.round(predict_result[0], 2))








