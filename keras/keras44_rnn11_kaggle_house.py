from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping


# 1. 데이터
path = './_data/kaggle_house/'  
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

print(train_csv) #[1460 rows x 80 columns]
print(train_csv.shape) #(1460, 80)

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)

# 1.3 결측지
print(train_csv.isnull().sum())

# 1.4 라벨인코딩( object 에서 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)


# 1.5 x, y 분리

x = train_csv.drop(['SalePrice'], axis=1)
print(x.shape) #(1121, 79)
y = train_csv['SalePrice']


# reshape
x = np.array(x)
x = x.reshape(-1, 79, 1)


test_csv = np.array(test_csv)
print(test_csv.shape) #(1459, 79)
test_csv = test_csv.reshape(-1, 79, 1)

# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)



# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(79,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=180, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)


# submission
y_submit = np.round(model.predict(test_csv)) 

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
import datetime
date= datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path_save = './_save/kaggle_house/' 
submission.to_csv(path_save + 'submit' + date +'.csv') 



