from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
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
print(np.unique(y, return_counts=True))
print(np.unique(y)) #[0 1]

# Reshape
print(x.shape) #(652, 8)
x = np.array(x)
x = x.reshape(652, 4, 2)

print(test_csv.shape) #(116, 8)

test_csv = np.array(test_csv)
test_csv = test_csv.reshape(116, 4, 2)



###############################train_csv 데이터에서 x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)



#2. 모델 구성
model=Sequential()
# model.add(LSTM(10, input_shape = (3,3)))  
model.add(Conv1D(10,2,input_shape = (4,2))) 
model.add(Conv1D(10,2))                    
model.add(Conv1D(10,2, padding='same'))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))





#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse']
              )

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=1000, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              
              



hist = model.fit(x_train,y_train, epochs=1000, batch_size=128,
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
submission.to_csv(path_save + 'submit_0314_10_val.csv')

