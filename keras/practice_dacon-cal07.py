import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessRegressor
from math import sqrt
import datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd

# Load data
path='c:/study/_data/dacon_cal/'
save_path= 'c:/study/_save/dacon_cal/'
submission = pd.read_csv(path+'sample_submission.csv')

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)

#1.4 x,y 분리
x = train_csv.drop(['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)', 'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender', 'Age'], axis=1)
y = train_csv['Calories_Burned']

print(x.shape) #(7500, 1)

x = np.array(x)
x = x.reshape(7500, 1, 1)

#1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=34553
    )     


#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(1,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              



hist = model.fit(x_train,y_train, epochs=1000, batch_size=128,
          validation_split=0.1,
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
loss = model.evaluate(x_test,y_test)
print('loss :', )

y_predict = model.predict(x_test)`1`
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)
#r2 스코어 : 0.6503924110719093

#RMSE함수의 정의 
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
#RMSE함수의 실행(사용)
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# Save submission file with timestamp
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)
#RMSE :  1.174009518693896
#RMSE :  0.1779004886084490   file: 0421_1753submission.csv
#RMSE :  0.18798388393213433  