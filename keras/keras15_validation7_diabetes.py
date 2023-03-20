from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1000)

# [실습]
#  R2 0.62 이상

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2) #학습시 데이터를 일부 나눠서 Validation(시스템 검증)으로 사용할 비율을 의미

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)
#r2 스코어 :  0.4857392047550714
#r2 스코어 :  0.4916748402930713
#r2 스코어 :  0.4901777414156322