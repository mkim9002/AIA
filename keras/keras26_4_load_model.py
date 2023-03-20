from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))
print(x)

# 보스톤 1.2부터 임포트 안된다
#pip uninstall scikit-learn
#pip install scikit-learn==1.1

print(np.min(x), np.max(x))  #0.0 711.0
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x)) #0.0 1.0

x_train, x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state=333
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train) #아래 코드와 같다
# x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max)


#2. 모델
# model = Sequential()
# model.add(Dense(30, input_shape=(13,)))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

# input1 = Input(shape=(13), name='h1')
# dense1 = Dense(30, name='h2')(input1)
# dense2 = Dense(20, name='h3')(dense1)
# dense3 = Dense(10, name='h4')(dense2)
# output1 = Dense(1, name='h5')(dense3)
# model = Model(inputs=input1, outputs=output1)

model = load_model('./_save/keras26_3_save_model.h5')

# #데이터가 3차원이면(새 계열 데이터)
# # (1000,100,1)->>> input_shape = (100,1)
# #데이터가 4차원이면(이미지 데이터)
# #( 60000, 32 ,32 ,3)->>> input_shape (32, 32 ,3)




# #3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=10)

# model.save('./_save/keras26_3_save_model.h5')

#4/ 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss )

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2 스코어 :', r2)
# #r2 스코어 : 0.6503924110719093




