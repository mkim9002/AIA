from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping



# data
import numpy as np
x1_datasets = np.array([range(100), range(301,401)])  #삼성 , 아모레
x2_datasets = np.array([range(101,201), range(411,511), range(150, 250)]) #온도, 습도, 강수량
x3_datasets = np.array([range(201,301), range(511,611), range(1300, 1400)]) 



print(x1_datasets.shape) #(2, 100)
print(x2_datasets.shape)   #(3, 100)
x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T

print(x1.shape) #(100, 2)
print(x2.shape)   #(100, 3)
print(x3.shape) 

y1 = np.array(range(2001,2101))  # 환율
y2 = np.array(range(1001,1101))  # 금리

 
 #실습 concaterate => Concaterate
 
 
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train , x2_test, x3_train, x3_test, \
y1_train, y1_test, y2_train, y2_test =train_test_split(
    x1,x2,x3,y1,y2, train_size=0.7 , random_state=333
)
# y_train, y_test, y_train, y_test =train_test_split(
#     x1,x2, train_size=0.7 , random_state=333
# )
print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(x3_train.shape, x3_test.shape)
print(y1_train.shape, y1_test.shape)
print(y2_train.shape, y2_test.shape)


#2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1 model1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='stock1')(input1)
dense2 = Dense(200, activation='relu', name='stock2')(dense1)
dense3 = Dense(300, activation='relu', name='stock3')(dense2)
output1 = Dense(11, activation='relu', name='outpu1')(dense3)

#2-2 model2
input2 = Input(shape=(3,))
dense11 = Dense(100, name='weather1')(input2)
dense12 = Dense(100, name='weather2')(dense11)
dense13 = Dense(100, name='weather3')(dense12)
dense14 = Dense(100, name='weather4')(dense13)
output2 = Dense(11, name='output2')(dense14)

#2-2 model3
input3 = Input(shape=(3,))
dense21 = Dense(100, name='steven1')(input3)
dense22 = Dense(100, name='steven2')(dense21)
dense23 = Dense(100, name='steven3')(dense22)
dense24 = Dense(100, name='steven4')(dense23)
output3 = Dense(11, name='output3')(dense24)

# from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2, output3], name='mg1')
# merge2 = Dense(200, activation='relu', name='mg2')(merge1)
# merge3 = Dense(300, activation='relu', name='mg3')(merge2)
# hidden_output = Dense(1, name='last')(merge3)

from tensorflow.keras.layers import Concatenate

merge1 = Concatenate(name='mg1')([output1, output2, output3])
merge2 = Dense(200, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, activation='relu', name='mg3')(merge2)
hidden_output = Dense(1, name='last')(merge3)

#Concatenate 객체를 생성할 때는 ()를 사용하여 생성자를 호출하고, 
#name 매개변수를 생성자에 전달하지 않습니다. 
# 대신 객체를 호출할 때 name 매개변수를 추가하여 전달합니다. 
# 따라서 merge1에서 name 매개변수는 호출에서 전달하고, Concatenate() 생성자에서는 제거합니다.

#2-1 분기1
bungi1 = Dense(10, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

#2-2 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])

model.summary()

#3. model.compile (evaluate는 test만 RMSE,R2 출력, 두개이상은?)
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=1000, verbose=1, mode='min', restore_best_weights=True)
model.fit([x1_train, x2_train, x3_train],[y1_train,y2_train], epochs=1000, batch_size=128, verbose=1, validation_split=0.01, callbacks=[es])

# 4. 평가, 예측
results = model.evaluate([x1_test, x2_test, x3_test],[y1_test, y2_test] )
print(results)

y_predict = model.predict([x1_test, x2_test, x3_test])
print(y_predict)
print(len(y_predict), len(y_predict[0]))   #2,30

from sklearn. metrics import r2_score
r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print('r2 score :', (r2_1 +r2_2)/2)


# loss = model.evaluate([x1_test,x2_test,x3_test], [y1_test, y2_test])
# print('loss : ', loss)

# y1_predict = model.predict([x1_test,x2_test,x3_test])
# y2_predict = model.predict([x1_test,x2_test,x3_test])


# r2_1 = r2_score(y1_predict, y1_test )
# print('r2 : ', r2_1)

# r2_2 = r2_score(y2_predict, y2_test )
# print('r2 : ', r2_2)


# def RMSE(x, y):
#     return np.sqrt(mean_squared_error(x, y))

# rmse = RMSE(y1_predict, y1_test, y2_predict, y2_test )
# print('rmse : ', rmse)

#

