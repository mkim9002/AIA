import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets)              #[420551 rows x 14 columns]
print(len(datasets))         #420551
print(datasets.shape)        #(420551, 14)
print(datasets.columns)      # Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    #    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    #    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    #    'wd (deg)'],
print(datasets.info())      
print(datasets.describe())
print(type(datasets))

print(datasets['T (degC)'].values)      # 판다스를 넘파이로
print(datasets['T (degC)'].to_numpy)    # 판다스를 넘파이로

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']
ts = 6 

x_train, x_test, _ , _ = train_test_split(x, y, train_size=0.7, shuffle=False)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x=scaler.transform(x)

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

# dataset은 x timesteps는 y 로 대처가능

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)
x_test, x_predict, y_test, y_predict = train_test_split(x_test, y_test, train_size=2/3, shuffle=False)




x_train_split = split_x(x_train, 6)
x_test_split = split_x(x_test, 6)
x_predict_split = split_x(x_predict, 6)

y_train_split = y_train[6:]
y_test_split = y_test[6:]
y_predict_split = y_predict[6:]

print(x_train.shape,y_train.shape) #(294385, 13) (294385,)
print(x_test.shape,y_test.shape)
print(x_predict.shape,y_predict.shape)




# 2. 모델구성
model = Sequential()
model.add(Conv1D(2, 4, input_shape=(6, 13)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train_split, y_train_split, epochs=1, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test_split, y_test_split)
print('loss : ', loss)

predict = model.predict(x_predict_split)

r2 = r2_score(y_predict_split, predict)
print('r2 : ', r2)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

rmse = RMSE(y_predict_split, predict)
print('rmse : ', rmse)