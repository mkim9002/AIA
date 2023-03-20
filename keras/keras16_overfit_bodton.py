from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets =load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2)


#2. 모델 구성
model = Sequential()
model.add(Dense(5, activation='relu',input_dim =13)) #print x shape의 (506. 13)input_dim은 x_shape두번째를를 본다
model.add(Dense(4,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='linear'))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train,y_train, epochs=500, batch_size=8,
          validation_split=0.2,
          verbose=1)
print("===========================================================")
print(hist)
print("===========================================================")
print(hist.history)
print("===========================================================")
print(hist.history['loss'])
print("===========================================================")
print(hist.history['val_loss'])


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label='발_로스')
plt.title('보스톤')
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()

