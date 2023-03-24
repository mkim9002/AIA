import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = load_breast_cancer()
print(datasets)

#print(datasets)
print(datasets.DESCR) #판다스 : .describe()
print(datasets.feature_names)   #판다스 :  .columns()

x = datasets ['data']
y = datasets.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x.shape, y.shape) #(569, 30) (569,) feature,열,columns 는 30
#print(y) #1010101 은 암에 걸린 사람과 아님

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(10, activation='relu',input_dim =30))
# model.add(Dense(9,activation='linear'))
# model.add(Dense(8,activation='linear'))
# model.add(Dense(7,activation='linear'))#이진분류는 sigmoid 쓴다
# model.add(Dense(1,activation='sigmoid')) #마지막 레이어에 0과1로 한정하는 것sigmoid 활성함수

input1 = Input(shape=(30,))
dense1 = Dense(10, activation = 'relu')(input1)
dense2 = Dense(9, activation = 'linear')(dense1)
dense3 = Dense(8, activation = 'linear')(dense2)
dense4 = Dense(7, activation = 'linear')(dense3)
output1 =Dense(1, activation = 'sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)



#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse']
              )

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'min',
                   verbose=1,
                   restore_best_weights=True
                   )
              



hist = model.fit(x_train,y_train, epochs=200, batch_size=16,
          validation_split=0.2,
          verbose=1,
          callbacks=(es),
)






# model.fit(x_train, y_train, epochs=100, batch_size=8,
#           validation_split=0.2,
#           verbose=1,
#           )


#4/ 평가 예측
results = model.evaluate(x_test,y_test)
print('results :', results )

y_predict = np.round(model.predict(x_test))
print("===============================")
print(y_test[:5])
print(np.round(y_predict[:5]))
print("===============================")



from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc ;', acc)


