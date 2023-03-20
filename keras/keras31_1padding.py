from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(7, (3,3),
                 padding='same' ,         #2,2,는 자르는 크기, 7은 output node #7장 으로늘여 하이퍼 파라미터 튜닝 (4.4.7)
                 input_shape=(8,8,1)))    #(출력 N,7,7,7)
                                          #(batch_size, rows, columns, channels) 컬러는3, 흑백은1
model.add(Conv2D(filters=4,
                 kernel_size=(3,3),
                 padding='valid',         #패딩의 디폴트는 valid
                 activation='relu' ))     #출력 : (N, 5 ,5 ,4)
model.add(Conv2D(10, (2,2)))              #출력 :(N, 4 ,4 ,10)
model.add(Flatten())                      #출력 : (N, 4*4*10) ->(N,160)
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))


model.summary()

