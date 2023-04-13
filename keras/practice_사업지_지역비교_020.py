import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


# 1. data
# 1.1 path, path_save, read_csv
path = './_data/개인프로젝트/'
path_save = './_save/개인프로젝트/'

datasets_검색수 = pd.read_csv(path + '한국관광공사-검색수.csv', index_col=0, encoding='cp949')
datasets_방문자수 = pd.read_csv(path + '한국관광공사-방문자수.csv', index_col=0, encoding='cp949')

# 1.2 데이터 범위 설정,전처리
# 1.2.1 drop 설정
검색수_x = np.array(datasets_검색수.drop(['합계'], axis=1))
검색수_y = np.array(datasets_검색수['서울특별시'])
방문자수_x = np.array(datasets_방문자수.drop(['합계'], axis=1))
방문자수_y = np.array(datasets_방문자수['서울특별시'])

# 1.2.2 범위 선택
검색수_x = 검색수_x[:18, :]
검색수_y = 검색수_y[:18]
방문자수_x = 방문자수_x[:18, :]
방문자수_y = 방문자수_y[:18]

#1.2.3 np.flip으로 전체 순서 반전
검색수_x = np.flip(검색수_x, axis=1)
검색수_y = np.flip(검색수_y)
방문자수_x = np.flip(방문자수_x, axis=1)
방문자수_y = np.flip(방문자수_y)

print(검색수_x.shape, 검색수_y.shape) #(17, 17) (17,)
print(방문자수_x.shape, 방문자수_y.shape) #(17, 17) (17,)

# 1.2.4 np.char.replace   astype(str)    .astype(np.float64) 문자를 숫자로 변경
검색수_x = np.char.replace(검색수_x.astype(str), ',', '').astype(np.float64)
검색수_y = np.char.replace(검색수_y.astype(str), ',', '').astype(np.float64)
방문자수_x = np.char.replace(방문자수_x.astype(str), ',', '').astype(np.float64)
방문자수_y = np.char.replace(방문자수_y.astype(str), ',', '').astype(np.float64)

# 1.2.4 train, test 분리
검색수_x_train, 검색수_x_test, 검색수_y_train, 검색수_y_test,\
방문자수_x_train, 방문자수_x_test, 방문자수_y_train, 방문자수_y_test \
= train_test_split(검색수_x, 검색수_y, 방문자수_x, 방문자수_y,
                    train_size=0.7, shuffle=False)

# 1.2.5 scaler (0,1로 분리)
scaler = MinMaxScaler()
검색수_x_train = scaler.fit_transform(검색수_x_train)
검색수_x_test= scaler.transform(검색수_x_test)
방문자수_x_train = scaler.transform(방문자수_x_train)
방문자수_x_test = scaler.transform(방문자수_x_test)

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
검색수_x_train_split = split_x(검색수_x_train, timesteps)
검색수_x_test_split = split_x(검색수_x_test, timesteps)
방문자수_x_train_split = split_x(방문자수_x_train, timesteps)
방문자수_x_test_split = split_x(방문자수_x_test, timesteps)

# 1.2.9 timestep 의 범위 설정 (버려지는 범위 설정 위해 [timesteps:])
검색수_y_train_split = 검색수_y_train[timesteps:]
검색수_y_test_split = 검색수_y_test[timesteps:]
방문자수_y_train_split = 방문자수_y_train[timesteps:]
방문자수_y_test_split = 방문자수_y_test[timesteps:]

print(검색수_x_train_split.shape)      # (9, 2, 17)
print(방문자수_x_train_split.shape)      #(9, 2, 17)

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 17))
conv1_1 = Conv1D(32, 3, padding='same', activation='relu', name='conv1_1')(input1)
conv1_2 = Conv1D(32, 3, padding='same', activation='relu', name='conv1_2')(conv1_1)
maxpool1 = MaxPooling1D(pool_size=2, name='maxpool1')(conv1_2)
lstm1 = LSTM(100, activation='relu', name='ss1')(maxpool1)
dense1 = Dense(90, activation='relu', name='ss2')(lstm1)
dense2 = Dense(80, activation='relu', name='ss3')(dense1)
output1 = Dense(70, activation='relu', name='ss4')(dense2)

# 2.2 모델2
input2 = Input(shape=(timesteps, 17))
conv2_1 = Conv1D(32, 3, padding='same', activation='relu', name='conv2_1')(input2)
conv2_2 = Conv1D(32, 3, padding='same', activation='relu', name='conv2_2')(conv2_1)
maxpool2 = MaxPooling1D(pool_size=2, name='maxpool2')(conv2_2)
lstm2 = LSTM(100, name='hd1')(maxpool2)
dense12 = Dense(90, name='hd2')(lstm2)
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
hist = model.fit([검색수_x_train_split, 방문자수_x_train_split], 
                 [검색수_y_train_split, 방문자수_y_train_split], 
                 epochs=1000, batch_size=128, validation_split=0.02, callbacks=[es])

model.save(path_save + '방문수.h5')



# 4. 평가, 예측

loss = model.evaluate([검색수_x_test_split, 방문자수_x_test_split], [검색수_y_test_split, 방문자수_y_test_split])
print('loss : ', loss)



검색수_x_predict = 검색수_x_test[-timesteps:]
검색수_x_predict = 검색수_x_predict.reshape(1, timesteps, 17)
방문자수_x_predict = 방문자수_x_test[-timesteps:]
방문자수_x_predict = 방문자수_x_predict.reshape(1, timesteps, 17)

predict_result = model.predict([검색수_x_predict, 방문자수_x_predict])

print("2023 서울특별시 여행 예상 방문자수 : ", np.round(predict_result[1]))

import matplotlib.pyplot as plt

# 데이터 입력
regions = ['Seoul', 'Busan', 'Daegu', 'Gwangju', 'Daejeon', 'Ulsan', 'Sejong',
           'Gyeonggi', 'Gangwon', 'Chungcheongbuk', 'Chungcheongnam', 'Jeollabuk',
           'Jeollanam', 'Gyeongsangbuk', 'Gyeongsangnam', 'Jeju']
visitors = [609069200, 134441650, 105707260, 58001448, 78917680, 44275416, 24146794,
            568462300, 134789680, 105425120, 152457950, 87704000, 112620190, 155813570,
            141691000, 53112944]

# 막대 그래프 그리기
import matplotlib.pyplot as plt

# 데이터 입력
regions = ['Seoul', 'Busan', 'Daegu', 'Gwangju', 'Daejeon', 'Ulsan', 'Sejong',
           'Gyeonggi', 'Gangwon', 'Chungcheongbuk', 'Chungcheongnam', 'Jeollabuk',
           'Jeollanam', 'Gyeongsangbuk', 'Gyeongsangnam', 'Jeju']
visitors = [609069200, 134441650, 105707260, 58001448, 78917680, 44275416, 24146794,
            568462300, 134789680, 105425120, 152457950, 87704000, 112620190, 155813570,
            141691000, 53112944]

# 막대 그래프 그리기
plt.barh(regions, visitors)  # barh 함수를 사용하여 가로막대 그래프 그리기

# 그래프 타이틀과 축 레이블 설정
plt.title('Expected Visitors in Korean Regions in 2023')
plt.xlabel('Visitors')
plt.ylabel('Regions')

# 그래프 출력
plt.show()


# 2023 서울 여행 예상 방문자수     : 609,069,200
# 2023 부산광역시 여행 예상 방문자수: 134,441,650
# 2023 대구광역시 여행 예상 방문자수: 105,707,260
# 2023 광주광역시여행 예상 방문자수  : 58,001,448
# 2023 대전광역시 여행 예상 방문자수 : 78,917,680
# 2023 울산광역시 여행 예상 방문자수 : 44,275,416
# 2023 세종자치시 여행 예상 방문자수 : 24,146,794
# 2023 경기도 여행 예상 방문자수    : 568,462,300
# 2023 강원도 여행 예상 방문자수    : 134,789,680
# 2023 충청북도 여행 예상 방문자수  : 105,425,120
# 2023 충청남도 여행 예상 방문자수 :  152,457,950 
# 2023 전라북도 여행 예상 방문자수 :   87,704,000
# 2023 전라남도 여행 예상 방문자수 :  112,620,190
# 2023 경상북도 여행 예상 방문자수 :  155,813,570
# 2023 경상남도 여행 예상 방문자수 :  141,691,000
# 2023 제주도 여행 예상 방문자수   :   53,112,944




