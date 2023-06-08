import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler,MaxAbsScaler,StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

#######
# 1. data
# 1.1 path, path_save, read_csv
path = 'c:/study/_data/baseball/'
path_save = 'c:/study/_save/baseball/'

datasets_doosan = pd.read_csv(path + 'doosan_data.csv', index_col=0, encoding='cp949')
datasets_lg = pd.read_csv(path + 'lg_data.csv', index_col=0, encoding='cp949')
datasets_kt = pd.read_csv(path + 'kt_data.csv', index_col=0, encoding='cp949')
datasets_nc = pd.read_csv(path + 'nc_data.csv', index_col=0, encoding='cp949')
datasets_lotte = pd.read_csv(path + 'lotte_data.csv', index_col=0, encoding='cp949')
datasets_kiwoom = pd.read_csv(path + 'kiwoom_data.csv', index_col=0, encoding='cp949')
datasets_hanwha = pd.read_csv(path + 'hanwha_data.csv', index_col=0, encoding='cp949')
datasets_samsung = pd.read_csv(path + 'samsung_data.csv', index_col=0, encoding='cp949')
datasets_kia = pd.read_csv(path + 'kia_data.csv', index_col=0, encoding='cp949')
datasets_ssg = pd.read_csv(path + 'ssg_data.csv', index_col=0, encoding='cp949')



# 1.2 데이터 범위 설정,전처리
# 1.2.1 drop 설정
doosan_x = np.array(datasets_doosan.drop(['전일비', 'SCORE'], axis=1))
doosan_y = np.array(datasets_doosan['SCORE'])
lg_x = np.array(datasets_lg.drop(['전일비', 'SCORE'], axis=1))
lg_y = np.array(datasets_lg['SCORE'])
kt_x = np.array(datasets_kt.drop(['전일비', 'SCORE'], axis=1))
kt_y = np.array(datasets_kt['SCORE'])
nc_x = np.array(datasets_nc.drop(['전일비', 'SCORE'], axis=1))
nc_y = np.array(datasets_nc['SCORE'])
lotte_x = np.array(datasets_lotte.drop(['전일비', 'SCORE'], axis=1))
lotte_y = np.array(datasets_lotte['SCORE'])
kt_x = np.array(datasets_kt.drop(['전일비', 'SCORE'], axis=1))
kt_y = np.array(datasets_kt['SCORE'])
kiwoom_x = np.array(datasets_kiwoom.drop(['전일비', 'SCORE'], axis=1))
kiwoom_y = np.array(datasets_kiwoom['SCORE'])
hanwha_x = np.array(datasets_hanwha.drop(['전일비', 'SCORE'], axis=1))
hanwha_y = np.array(datasets_hanwha['SCORE'])
samsung_x = np.array(datasets_samsung.drop(['전일비', 'SCORE'], axis=1))
samsung_y = np.array(datasets_samsung['SCORE'])
kia_x = np.array(datasets_kia.drop(['전일비', 'SCORE'], axis=1))
kia_y = np.array(datasets_kia['SCORE'])
ssg_x = np.array(datasets_ssg.drop(['전일비', 'SCORE'], axis=1))
ssg_y = np.array(datasets_ssg['SCORE'])




# 1.2.2 범위 선택
doosan_x = doosan_x[:180, :]
doosan_y = doosan_y[:180]
lg_x = lg_x[:180, :]
lg_y = lg_y[:180]
kt_x = kt_x[:180, :]
kt_y = kt_y[:180]
nc_x = nc_x[:180, :]
nc_y = nc_y[:180]
lotte_x = lotte_x[:180, :]
lotte_y = lotte_y[:180]
kt_x = kt_x[:180, :]
kt_y = kt_y[:180]
kiwoom_x = kiwoom_x[:180, :]
kiwoom_y = kiwoom_y[:180]
hanwha_x = hanwha_x[:180, :]
hanwha_y = hanwha_y[:180]
samsung_x = samsung_x[:180, :]
samsung_y = samsung_y[:180]
kia_x = kia_x[:180, :]
kia_y = kia_y[:180]
ssg_x = ssg_x[:180, :]
ssg_y = ssg_y[:180]


#1.2.3 np.flip으로 전체 순서 반전
doosan_x = np.flip(doosan_x, axis=1)
doosan_y = np.flip(doosan_y)
lg_x = np.flip(lg_x, axis=1)
lg_y = np.flip(lg_y)
kt_x = np.flip(kt_x, axis=1)
kt_y = np.flip(kt_y)
nc_x = np.flip(nc_x, axis=1)
nc_y = np.flip(nc_y)
lotte_x = np.flip(lotte_x, axis=1)
lotte_y = np.flip(lotte_y)
kiwoom_x = np.flip(kiwoom_x, axis=1)
kiwoom_y = np.flip(kiwoom_y)
hanwha_x = np.flip(hanwha_x, axis=1)
hanwha_y = np.flip(hanwha_y)
samsung_x = np.flip(samsung_x, axis=1)
samsung_y = np.flip(samsung_y)
kia_x = np.flip(kia_x, axis=1)
kia_y = np.flip(kia_y)
ssg_x = np.flip(ssg_x, axis=1)
ssg_y = np.flip(ssg_y)


# 1.2.4 np.char.replace   astype(str)    .astype(np.float64) 문자를 숫자로 변경
doosan_x = np.char.replace(doosan_x.astype(str), ',', '').astype(np.float64)
doosan_y = np.char.replace(doosan_y.astype(str), ',', '').astype(np.float64)
lg_x = np.char.replace(lg_x.astype(str), ',', '').astype(np.float64)
lg_y = np.char.replace(lg_y.astype(str), ',', '').astype(np.float64)
kt_x = np.char.replace(kt_x.astype(str), ',', '').astype(np.float64)
kt_y = np.char.replace(kt_y.astype(str), ',', '').astype(np.float64)
nc_x = np.char.replace(nc_x.astype(str), ',', '').astype(np.float64)
nc_y = np.char.replace(nc_y.astype(str), ',', '').astype(np.float64)
lotte_x = np.char.replace(lotte_x.astype(str), ',', '').astype(np.float64)
lotte_y = np.char.replace(lotte_y.astype(str), ',', '').astype(np.float64)
kiwoom_x = np.char.replace(kiwoom_x.astype(str), ',', '').astype(np.float64)
kiwoom_y = np.char.replace(kiwoom_y.astype(str), ',', '').astype(np.float64)
hanwha_x = np.char.replace(hanwha_x.astype(str), ',', '').astype(np.float64)
hanwha_y = np.char.replace(hanwha_y.astype(str), ',', '').astype(np.float64)
samsung_x = np.char.replace(samsung_x.astype(str), ',', '').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',', '').astype(np.float64)
kia_x = np.char.replace(kia_x.astype(str), ',', '').astype(np.float64)
kia_y = np.char.replace(kia_y.astype(str), ',', '').astype(np.float64)
ssg_x = np.char.replace(ssg_x.astype(str), ',', '').astype(np.float64)
ssg_y = np.char.replace(ssg_y.astype(str), ',', '').astype(np.float64)


# 1.2.5 train, test 분리
doosan_x_train, doosan_x_test, doosan_y_train, doosan_y_test,\
lg_x_train, lg_x_test, lg_y_train, lg_y_test, \
kt_x_train, kt_x_test, kt_y_train, kt_y_test, \
nc_x_train, nc_x_test, nc_y_train, nc_y_test, \
lotte_x_train, lotte_x_test, lotte_y_train, lotte_y_test, \
kiwoom_x_train, kiwoom_x_test, kiwoom_y_train, kiwoom_y_test, \
hanwha_x_train, hanwha_x_test, hanwha_y_train, hanwha_y_test, \
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, \
kia_x_train, kia_x_test, kia_y_train, kia_y_test, \
ssg_x_train, ssg_x_test, ssg_y_train, ssg_y_test, \
= train_test_split(doosan_x, doosan_y, lg_x, lg_y,
                   kt_x, kt_y, nc_x, nc_y,
                   lotte_x, lotte_y, kiwoom_x, kiwoom_y,
                   hanwha_x, hanwha_y, samsung_x, samsung_y,
                   kia_x, kia_y, ssg_x, ssg_y,
                    train_size=0.3, shuffle=False)

# 1.2.5 scaler (0,1로 분리)
scaler = MaxAbsScaler()

doosan_x_train = scaler.fit_transform(doosan_x_train)
doosan_x_test = scaler.transform(doosan_x_test)

lg_x_train = scaler.fit_transform(lg_x_train)
lg_x_test = scaler.transform(lg_x_test)

kt_x_train = scaler.fit_transform(kt_x_train)
kt_x_test = scaler.transform(kt_x_test)

nc_x_train = scaler.fit_transform(nc_x_train)
nc_x_test = scaler.transform(nc_x_test)

lotte_x_train = scaler.fit_transform(lotte_x_train)
lotte_x_test = scaler.transform(lotte_x_test)

kiwoom_x_train = scaler.fit_transform(kiwoom_x_train)
kiwoom_x_test = scaler.transform(kiwoom_x_test)

hanwha_x_train = scaler.fit_transform(hanwha_x_train)
hanwha_x_test = scaler.transform(hanwha_x_test)

samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test = scaler.transform(samsung_x_test)

kia_x_train = scaler.fit_transform(kia_x_train)
kia_x_test = scaler.transform(kia_x_test)

ssg_x_train = scaler.fit_transform(ssg_x_train)
ssg_x_test = scaler.transform(ssg_x_test)




# 1.2.6 timesteps
timesteps = 51

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
kt_x_train_split = split_x(kt_x_train, timesteps)
kt_x_test_split = split_x(kt_x_test, timesteps)
nc_x_train_split = split_x(nc_x_train, timesteps)
nc_x_test_split = split_x(nc_x_test, timesteps)
lotte_x_train_split = split_x(lotte_x_train, timesteps)
lotte_x_test_split = split_x(lotte_x_test, timesteps)
kiwoom_x_train_split = split_x(kiwoom_x_train, timesteps)
kiwoom_x_test_split = split_x(kiwoom_x_test, timesteps)
hanwha_x_train_split = split_x(hanwha_x_train, timesteps)
hanwha_x_test_split = split_x(hanwha_x_test, timesteps)
samsung_x_train_split = split_x(samsung_x_train, timesteps)
samsung_x_test_split = split_x(samsung_x_test, timesteps)
kia_x_train_split = split_x(kia_x_train, timesteps)
kia_x_test_split = split_x(kia_x_test, timesteps)
ssg_x_train_split = split_x(ssg_x_train, timesteps)
ssg_x_test_split = split_x(ssg_x_test, timesteps)


# 1.2.9 timestep 의 범위 설정 (버려지는 범위 설정 위해 [timesteps:])

doosan_y_train_split = doosan_y_train[timesteps:]
doosan_y_test_split = doosan_y_test[timesteps:]
lg_y_train_split = lg_y_train[timesteps:]
lg_y_test_split = lg_y_test[timesteps:]
kt_y_train_split = kt_y_train[timesteps:]
kt_y_test_split = kt_y_test[timesteps:]
nc_y_train_split = nc_y_train[timesteps:]
nc_y_test_split = nc_y_test[timesteps:]
lotte_y_train_split = lotte_y_train[timesteps:]
lotte_y_test_split = lotte_y_test[timesteps:]
kiwoom_y_train_split = kiwoom_y_train[timesteps:]
kiwoom_y_test_split = kiwoom_y_test[timesteps:]
hanwha_y_train_split = hanwha_y_train[timesteps:]
hanwha_y_test_split = hanwha_y_test[timesteps:]
samsung_y_train_split = samsung_y_train[timesteps:]
samsung_y_test_split = samsung_y_test[timesteps:]
kia_y_train_split = kia_y_train[timesteps:]
kia_y_test_split = kia_y_test[timesteps:]
ssg_y_train_split = ssg_y_train[timesteps:]
ssg_y_test_split = ssg_y_test[timesteps:]

print(doosan_x_train_split.shape)      # (116, 9, 8)
print(lg_x_train_split.shape)      # (116, 9, 8)

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 8))
dense1 = LSTM(10, activation='relu', name='ss1')(input1)
dense2 = Dense(10, activation='relu', name='ss2')(dense1)
dense3 = Dense(10, activation='relu', name='ss3')(dense2)
output1 = Dense(10, activation='relu', name='ss4')(dense3)

# 2.2 모델2
input2 = Input(shape=(timesteps, 8))
dense11 = LSTM(10, name='ds1')(input2)
dense12 = Dense(10, name='ds2')(dense11)
dense13 = Dense(10, name='ds3')(dense12)
dense14 = Dense(10, name='ds4')(dense13)
output2 = Dense(10, name='output2')(dense14)

# 2.2 모델3
input3 = Input(shape=(timesteps, 8))
dense31 = LSTM(10, name='lg1')(input3)
dense32 = Dense(10, name='lg2')(dense31)
dense33 = Dense(10, name='lg3')(dense32)
dense34 = Dense(10, name='lg4')(dense33)
output3 = Dense(10, name='output3')(dense34)

# 2.2 모델4
input4 = Input(shape=(timesteps, 8))
dense41 = LSTM(10, name='kt1')(input4)
dense42 = Dense(10, name='kt2')(dense41)
dense43 = Dense(10, name='kt3')(dense42)
dense44 = Dense(10, name='kt4')(dense43)
output4 = Dense(10, name='output4')(dense44)

# 2.2 모델5
input5 = Input(shape=(timesteps, 8))
dense51 = LSTM(10, name='nc1')(input5)
dense52 = Dense(10, name='nc2')(dense51)
dense53 = Dense(10, name='nc3')(dense52)
dense54 = Dense(10, name='nc4')(dense53)
output5 = Dense(10, name='output5')(dense54)

# 2.2 모델6
input6 = Input(shape=(timesteps, 8))
dense61 = LSTM(10, name='lt1')(input6)
dense62 = Dense(10, name='lt2')(dense61)
dense63 = Dense(10, name='lt3')(dense62)
dense64 = Dense(10, name='lt4')(dense63)
output6 = Dense(10, name='output6')(dense64)

# 2.2 모델7
input7 = Input(shape=(timesteps, 8))
dense71 = LSTM(10, name='kw1')(input7)
dense72 = Dense(10, name='kw2')(dense71)
dense73 = Dense(10, name='kw3')(dense72)
dense74 = Dense(10, name='kw4')(dense73)
output7 = Dense(10, name='output7')(dense74)

# 2.2 모델8
input8 = Input(shape=(timesteps, 8))
dense81 = LSTM(10, name='hw1')(input8)
dense82 = Dense(10, name='hw2')(dense81)
dense83 = Dense(10, name='hw3')(dense82)
dense84 = Dense(10, name='hw4')(dense83)
output8 = Dense(10, name='output8')(dense84)

# 2.2 모델9
input9 = Input(shape=(timesteps, 8))
dense91 = LSTM(10, name='ki1')(input9)
dense92 = Dense(10, name='ki2')(dense91)
dense93 = Dense(10, name='ki3')(dense92)
dense94 = Dense(10, name='ki4')(dense93)
output9 = Dense(10, name='output9')(dense94)

# 2.2 모델10
input10 = Input(shape=(timesteps, 8))
dense101 = LSTM(10, name='sg1')(input10)
dense102 = Dense(10, name='sg2')(dense101)
dense103 = Dense(10, name='sg3')(dense102)
dense104 = Dense(10, name='sg4')(dense103)
output10 = Dense(10, name='output10')(dense104)

# 2.3 머지
merge1 = Concatenate(name='mg1')([output1, output2, output3, output4, output5, output6, output7, output8, output9, output10])
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(10, activation='relu', name='mg3')(merge2)
merge4 = Dense(10, activation='relu', name='mg4')(merge3)
merge5 = Dense(10, activation='relu', name='mg5')(merge4)
merge6 = Dense(10, activation='relu', name='mg6')(merge5)
merge7 = Dense(10, activation='relu', name='mg7')(merge6)
merge8 = Dense(10, activation='relu', name='mg8')(merge7)
merge9 = Dense(10, activation='relu', name='mg9')(merge8)
merge10 = Dense(10, activation='relu', name='mg10')(merge9)
hidden_output = Dense(10, name='last')(merge10)

# 2.5 분기1
bungi1 = Dense(10, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
bungi3 = Dense(10, name='bg3')(bungi2)
bungi4 = Dense(10, name='bg4')(bungi3)
bungi5 = Dense(10, name='bg5')(bungi4)
bungi6 = Dense(10, name='bg6')(bungi5)
bungi7 = Dense(10, name='bg7')(bungi6)
bungi8 = Dense(10, name='bg8')(bungi7)
bungi9 = Dense(10, name='bg9')(bungi8)
bungi10 = Dense(10, name='bg10')(bungi9)
last_output1 = Dense(1, name='last1')(bungi10)

# 2.6 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)
last_output3 = Dense(1, activation='linear', name='last3')(hidden_output)
last_output4 = Dense(1, activation='linear', name='last4')(hidden_output)
last_output5 = Dense(1, activation='linear', name='last5')(hidden_output)
last_output6 = Dense(1, activation='linear', name='last6')(hidden_output)
last_output7 = Dense(1, activation='linear', name='last7')(hidden_output)
last_output8 = Dense(1, activation='linear', name='last8')(hidden_output)
last_output9 = Dense(1, activation='linear', name='last9')(hidden_output)
last_output10 = Dense(1, activation='linear', name='last10')(hidden_output)

model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10], 
              outputs=[last_output1,last_output2,last_output3,last_output4,last_output5,last_output6,last_output7,last_output8,last_output9,last_output10])

model.summary()

#rmsprop
#adagrad
#sgd
#adam

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='rmsprop')
es = EarlyStopping(monitor='val_loss', mode='min', patience=200, restore_best_weights=True)
hist = model.fit([doosan_x_train_split, lg_x_train_split, kt_x_train_split, nc_x_train_split,lotte_x_train_split, hanwha_x_train_split, samsung_x_train_split, kiwoom_x_train_split, kia_x_train_split, ssg_x_train_split ], 
                 [doosan_y_train_split, lg_y_train_split, kt_y_train_split, nc_y_train_split,lotte_y_train_split, hanwha_y_train_split, samsung_y_train_split, kiwoom_y_train_split, kia_y_train_split, ssg_y_train_split ],                                 
                 epochs=2000, batch_size=99, validation_split=0.1, callbacks=[es])




model.save(path_save + 'keras53_doosan2_kmg.h5')

# 4. 평가, 예측

loss = model.evaluate([doosan_x_test_split, lg_x_test_split, kt_x_test_split, nc_x_test_split, lotte_x_test_split, kiwoom_x_test_split, hanwha_x_test_split, samsung_x_test_split, kia_x_test_split, ssg_x_test_split], 
                      [doosan_y_test_split, lg_y_test_split, kt_y_test_split, nc_y_test_split, lotte_y_test_split, kiwoom_y_test_split, hanwha_y_test_split, samsung_y_test_split, kia_y_test_split, ssg_y_test_split])
print('loss : ', loss)

# for_r2 = model.predict([doosan_x_test_split, lg_x_test_split])
# print(f'결정계수 : {r2_score(doosan_y_test_split,for_r2[0])/2+r2_score(lg_y_test_split,for_r2[1])/2}')

doosan_x_predict = doosan_x_test[-timesteps:]
doosan_x_predict = doosan_x_predict.reshape(1, timesteps, 8)
lg_x_predict = lg_x_test[-timesteps:]
lg_x_predict = lg_x_predict.reshape(1, timesteps, 8)
kt_x_predict = kt_x_test[-timesteps:]
kt_x_predict = kt_x_predict.reshape(1, timesteps, 8)
nc_x_predict = nc_x_test[-timesteps:]
nc_x_predict = nc_x_predict.reshape(1, timesteps, 8)
lotte_x_predict = lotte_x_test[-timesteps:]
lotte_x_predict = lotte_x_predict.reshape(1, timesteps, 8)
kiwoom_x_predict = kiwoom_x_test[-timesteps:]
kiwoom_x_predict = kiwoom_x_predict.reshape(1, timesteps, 8)
hanwha_x_predict = hanwha_x_test[-timesteps:]
hanwha_x_predict = hanwha_x_predict.reshape(1, timesteps, 8)
samsung_x_predict = samsung_x_test[-timesteps:]
samsung_x_predict = samsung_x_predict.reshape(1, timesteps, 8)
kia_x_predict = kia_x_test[-timesteps:]
kia_x_predict = kia_x_predict.reshape(1, timesteps, 8)
ssg_x_predict = ssg_x_test[-timesteps:]
ssg_x_predict = ssg_x_predict.reshape(1, timesteps, 8)

predict_result = model.predict([doosan_x_predict, lg_x_predict, kt_x_predict, nc_x_predict, 
                                lotte_x_predict, kiwoom_x_predict, hanwha_x_predict, 
                                samsung_x_predict, kia_x_predict, ssg_x_predict])

print("############ 최고 3600 배 고배당 가능  ###########")

print("############ 프로야구 오늘의 각팀 예상 득점 스코어 ###########")
print("DOOSAN SCORE  : ", np.round(predict_result[0], ))
print("LG SCORE      : ", np.round(predict_result[1], ))
print("KT SCORE      : ", np.round(predict_result[2], ))
print("NC SCORE      : ", np.round(predict_result[3], ))
print("LOTTE SCORE   : ", np.round(predict_result[4], ))
print("KIWOOM SCORE  : ", np.round(predict_result[5], ))
print("HANWHA SCORE  : ", np.round(predict_result[6], ))
print("SAMSUNG SCORE : ", np.round(predict_result[7], ))
print("KIA SCORE     : ", np.round(predict_result[8], ))
print("SSG SCORE     : ", np.round(predict_result[9], ))

print("############ 최고 3600 배 고배당 가능  ###########")












