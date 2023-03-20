from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


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

#scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max)


#2. 모델
model = Sequential()
model.add(Dense(1,input_dim=13))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)



