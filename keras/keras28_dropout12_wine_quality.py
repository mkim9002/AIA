from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/wine_quality/'   #점 하나 현재폴더의밑에 점하나는 스터디
path_save = './_save/wine_quality/' 

train_csv = pd.read_csv(path + 'train.csv', index_col=0) 

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

# 1.2 확인 사항 5가지
print(train_csv.shape, test_csv.shape) #(5497, 13) (1000, 12)
#print(train_csv.columns, test_csv.columns)
#print(train_csv.info(), test_csv.info())
#print(train_csv.describe(), test_csv.describe())
#print(type(train_csv), type(test_csv))

# 1.3 결측치 처리
# print(train_csv.isnull().sum()) # 결축지 없음

# 1.4 x, y 분리
x = train_csv.drop(['quality', 'type'], axis=1)
y = train_csv['quality']
test_csv = test_csv.drop(['type'], axis=1)

# print(x.shape) #(5497, 11)
# print(test_csv.shape) #(5497, 11)

#1.5 원핫인코딩
# print(np.unique(y)) #[3 4 5 6 7 8 9]
print(type(y))
y=pd.get_dummies(y)
print(type(y))
print(y)
y = np.array(y)
print(type(y))
print(y)

# print(type(y))
# print(y.shape  #5497,7)

# y = to_categorical(y)
# print(y)
# print(y.shape)
# y = y[:,3:]
# print(y.shape)
# print(y)


# 1.6 train, test 분리

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, 
                        train_size=0.7, random_state=123,stratify=y)


# 1.7 Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



#2. 모델 구성


input1 = Input(shape=(11,))
dense1 = Dense(500, activation = 'relu')(input1)
dense2 = Dense(64, activation = 'relu')(dense1)
drop1 = Dropout(0.25)(dense2)
dense3 = Dense(32, activation = 'relu')(drop1)
dense4 = Dense(16, activation = 'relu')(dense3)
dense5 = Dense(8, activation = 'relu')(dense4)
drop2 = Dropout(0.25)(dense5)
output1 = Dense(7, activation = 'softmax')(drop2)
model = Model(inputs=input1, outputs=output1)



#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc']
              )


es = EarlyStopping(monitor='val_loss', patience=500,
                   verbose=1,
                   restore_best_weights=True
                   )
              
hist = model.fit(x_train,y_train, epochs=10000, batch_size=50,
          validation_split=0.9,
          verbose=1,
          callbacks=(es),
)




#4/ 평가 예측
results = model.evaluate(x_test,y_test)
print('results :', results )

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict)
print('accuracy score :',acc)

#submission.csv 만들기
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(y_submit)
# print(y_submit.shape)
y_submit = np.argmax(y_submit, axis=1)
print(y_submit.shape)
y_submit += 3
submission['quality'] = y_submit
# print(submission)

path_save = './_save/wine_quality/' 
submission.to_csv(path_save + 'submit_013.csv')
