from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Flatten,Conv2D,LSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# 1. 데이터
# 1.1 경로, 가져오기
path='c:/study/_data/dacon_cal/'
save_path= 'c:/study/_save/dacon_cal/'
submission = pd.read_csv(path+'sample_submission.csv')

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)
# (7500, 10) (7500, 9)

# 1.3 결측지
# print(train_csv.isnull().sum())
# print(train_csv.info())

# 1.4 라벨인코딩( object 에서 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
# print(len(train_csv.columns))
# print('==============')
# print(train_csv.info())
# print('===================')
train_csv=train_csv.dropna()
print(train_csv.shape)


# 1.5 x, y 분리
x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

print(x.shape,y.shape) #(7500, 9) (7500,)

# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=35468264, shuffle=True)

print(x_train.shape) #(6000, 9)
print(x_test.shape) #(1500, 9)
print(test_csv.shape) #(7500, 9)

# 1.7 Scaler
scaler = MinMaxScaler() # 여기서 어레이 형태로 해서 아래 리쉐잎때 변환안해줘도됨
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

x_train= x_train.reshape(6000,9,1)
x_test= x_test.reshape(1500,9,1)
test_csv = test_csv.reshape(7500,9,1) # test파일도 모델에서 돌려주니까 리쉐잎 해줘야됨.


model = Sequential()
model.add(LSTM(96,input_shape=(9,1),activation='linear',return_sequences=True))
model.add(LSTM(86,input_shape=(9,1),activation='relu'))
model.add(Dense(76,activation='selu'))
model.add(Dense(66))
model.add(Dense(56))
model.add(Dropout(0.1))
model.add(Dense(46))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

# model = Sequential()
# model.add(Conv2D(7,(2,1),input_shape=(8,1,1)))
# model.add(Conv2D(8,(2,1),activation='relu'))
# model.add(Flatten())
# model.add(Dense(9,activation='relu'))
# model.add(Dense(6))
# model.add(Dense(1))

# input1 = Input(shape=(78,))
# dense1 = Dense(32)(input1)
# drop1 = Dropout(0.2)(dense1)
# dense2 = Dense(64, activation='relu')(drop1)
# dense3 = Dense(64)(dense2)
# dense4 = Dense(32,activation='relu')(dense3)
# dense5 = Dense(35)(dense4)
# drop2 = Dropout(0.2)(dense5)
# output1 = Dense(1)(drop2)
# model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=300, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1, validation_split=0.1, callbacks=[es])

model.save('./_save/kcal/kcal_save_model01.h5')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# RMSE 함수 정의
def RMSE(y_test,y_pre):
    return np.sqrt(mean_squared_error(y_test,y_pre)) #정의
rmse=RMSE(y_test,y_predict) #사용
print('RMSE :',rmse)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

# Save submission file with timestamp
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)
