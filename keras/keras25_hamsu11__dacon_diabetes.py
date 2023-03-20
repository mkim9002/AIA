from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1. 데이터
path = './_data/dacon_diabetes/'   #점 하나 현재폴더의밑에 점하나는 스터디
path_save = './_save/dacon_diabetes/' 

train_csv = pd.read_csv(path + 'train.csv',
                       index_col=0) 

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 

print(test_csv)   #[116 rows x 8 columns]

print(train_csv)  #[652 rows x 9 columns]

#결측치 처리 1 .제거
# pirnt(train_csv.insul11())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() ####결측치 제거#####
print(train_csv.isnull().sum()) #(11)
print(train_csv.info())
print(train_csv.shape)

############################## train_csv 데이터에서 x와y를 분리
x = train_csv.drop(['Outcome'], axis=1) #2개 이상 리스트 
print(x)
y = train_csv['Outcome']
print(y)



scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

###############################train_csv 데이터에서 x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='relu',input_dim =8)) #print x shape의 (506. 13)input_dim은 x_shape두번째를를 본다
# model.add(Dense(9,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(7,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

input1 = Input(shape=(8,))
dense1 = Dense(10, activation = 'relu')(input1)
dense2 = Dense(9, activation = 'relu')(dense1)
dense3 = Dense(8, activation = 'relu')(dense2)
dense4 = Dense(7, activation = 'relu')(dense3)
output1 = Dense(1, activation = 'sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)





#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse']
              )

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=90, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              
              



hist = model.fit(x_train,y_train, epochs=350, batch_size=4,
          validation_split=0.2,
          verbose=1,
          callbacks=(es),
)
     
# print("===========================================================")
# print(hist)
# print("===========================================================")
# print(hist.history)
# print("===========================================================")
# print(hist.history['loss'])
print("===========================================================")
print(hist.history['val_loss'])


#4/ 평가 예측
results = model.evaluate(x_test,y_test)
print('results :', results )

y_predict = np.round(model.predict(x_test))

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc ;', acc)

#submission.csv 만들기
y_submit = np.round(model.predict(test_csv)) #위에서 'test_csv'명명 -> test_csv예측값을 y_submit이라 함 
# print(y_submit)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
# print(submission)

path_save = './_save/dacon_diabetes/' 
submission.to_csv(path_save + 'submit_0311_0147_val.csv')

