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
path = './_data/시험/'
path_save = './_save/samsung/'

datasets_samsung = pd.read_csv(path + '삼성전자 주가2.csv', index_col=0, encoding='cp949')
datasets_hyundai = pd.read_csv(path + '현대자동차.csv', index_col=0, encoding='cp949')

print(datasets_samsung.shape, datasets_hyundai.shape)
print(datasets_samsung.columns, datasets_hyundai.columns)
print(datasets_samsung.info(), datasets_hyundai.info())
print(datasets_samsung.describe(), datasets_hyundai.describe())
print(type(datasets_samsung), type(datasets_hyundai))

# 1.2 데이터 범위 설정,전처리
# 1.2.1 drop 설정
samsung_x = np.array(datasets_samsung.drop(['전일비', '종가'], axis=1))
samsung_y = np.array(datasets_samsung['종가'])
hyundai_x = np.array(datasets_hyundai.drop(['전일비', '종가'], axis=1))
hyundai_y = np.array(datasets_hyundai['종가'])

# 1.2.2 범위 선택
samsung_x = samsung_x[:180, :]
samsung_y = samsung_y[:180]
hyundai_x = hyundai_x[:180, :]
hyundai_y = hyundai_y[:180]

#1.2.3 np.flip으로 전체 순서 반전
samsung_x = np.flip(samsung_x, axis=1)
samsung_y = np.flip(samsung_y)
hyundai_x = np.flip(hyundai_x, axis=1)
hyundai_y = np.flip(hyundai_y)

print(samsung_x.shape, samsung_y.shape)
print(hyundai_x.shape, hyundai_y.shape)

# 1.2.4 np.char.replace   astype(str)    .astype(np.float64) 문자를 숫자로 변경
samsung_x = np.char.replace(samsung_x.astype(str), ',', '').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',', '').astype(np.float64)
hyundai_x = np.char.replace(hyundai_x.astype(str), ',', '').astype(np.float64)
hyundai_y = np.char.replace(hyundai_y.astype(str), ',', '').astype(np.float64)

# 1.2.4 train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test,\
hyundai_x_train, hyundai_x_test, hyundai_y_train, hyundai_y_test \
= train_test_split(samsung_x, samsung_y, hyundai_x, hyundai_y,
                    train_size=0.7, shuffle=False)

# 1.2.5 scaler (0,1로 분리)
scaler = MinMaxScaler()
samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test= scaler.transform(samsung_x_test)
hyundai_x_train = scaler.transform(hyundai_x_train)
hyundai_x_test = scaler.transform(hyundai_x_test)

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
samsung_x_train_split = split_x(samsung_x_train, timesteps)
samsung_x_test_split = split_x(samsung_x_test, timesteps)
hyundai_x_train_split = split_x(hyundai_x_train, timesteps)
hyundai_x_test_split = split_x(hyundai_x_test, timesteps)

# 1.2.9 timestep 의 범위 설정 (버려지는 범위 설정 위해 [timesteps:])
samsung_y_train_split = samsung_y_train[timesteps:]
samsung_y_test_split = samsung_y_test[timesteps:]
hyundai_y_train_split = hyundai_y_train[timesteps:]
hyundai_y_test_split = hyundai_y_test[timesteps:]

print(samsung_x_train_split.shape)      # (820, 20, 14)
print(hyundai_x_train_split.shape)      # (820, 20, 14)

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 14))
dense1 = LSTM(100, activation='relu', name='ss1')(input1)
dense2 = Dense(90, activation='relu', name='ss2')(dense1)
dense3 = Dense(80, activation='relu', name='ss3')(dense2)
output1 = Dense(70, activation='relu', name='ss4')(dense3)

# 2.2 모델2
input2 = Input(shape=(timesteps, 14))
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
hist = model.fit([samsung_x_train_split, hyundai_x_train_split], 
                 [samsung_y_train_split, hyundai_y_train_split], 
                 epochs=1000, batch_size=128, validation_split=0.02, callbacks=[es])

model.save(path_save + 'keras53_samsung2_kmg.h5')

# 4. 평가, 예측

loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], [samsung_y_test_split, hyundai_y_test_split])
print('loss : ', loss)

# for_r2 = model.predict([samsung_x_test_split, hyundai_x_test_split])
# print(f'결정계수 : {r2_score(samsung_y_test_split,for_r2[0])/2+r2_score(hyundai_y_test_split,for_r2[1])/2}')

samsung_x_predict = samsung_x_test[-timesteps:]
samsung_x_predict = samsung_x_predict.reshape(1, timesteps, 14)
hyundai_x_predict = hyundai_x_test[-timesteps:]
hyundai_x_predict = hyundai_x_predict.reshape(1, timesteps, 14)

predict_result = model.predict([samsung_x_predict, hyundai_x_predict])

print("종가 : ", np.round(predict_result[0], 2))

# 종가 :  [[62655.15]]






