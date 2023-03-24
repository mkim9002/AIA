#다중분류
# output layer, loss만 다름
#1. 마지막 아웃풋 레이어, activation = 'softmax'사용 (바뀌지않음!무조건)
#   y의 라벨값의 개수만큼 노드를 정해준다!(이중분류와의 차이점)
#2. loss = 'categorical_crossentropy'사용

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터 
datasets = load_iris()
print(datasets.DESCR) #(150,4)  #pandas : describe()
'''
#class : label값 (3개를 맞춰라)
- Iris-Setosa
- Iris-Versicolour
- Iris-Virginica
'''
print(datasets.feature_names)  #pandas : colums()
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(150, 4) (150,) 
print(x)
print(y)  #0,1,2로만 순서대로 나옴-> 섞어줘야함(shuffle)
print('y의 라벨값 :', np.unique(y))  #y의 라벨값 : [0 1 2]

#1.2 reshape
x = x.reshape(150, 4, 1)



#데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y    #통계적으로 (y값,같은 비율로)
)
print(y_train)                                  #[0 1 0 0 1 0 1 2 2 1 0 2 2 1 2]
print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2]), array([5, 5, 5], dtype=int64)) #return_counts=True : 개수 반환

# one hot encoding
y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, input_shape=(4,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=16,
          validation_split=0.2,
          verbose=1,
          )

# *ValueError: Shapes (16, 1) and (16, 3) are incompatible 
# y의 라벨값의 개수 = y의 클래스의 개수
# (150,)인데 노드3으로 잡으면 (150,3)으로 출력하기때문에 오류가 발생 (..y값1개인데 이 안에 클래스3개 들어가있음..)
# => one-hot Encoding 사용!
#softmax의 전체의 합=1, one-hot encoding의 전체의 합 =1 


#[과제] accuracy_score를 사용해서 스코어를 빼세요.
#y_predict값(소수점나오니까..)=> 0,1,2 로 바꿔줘야함 (np에 있음)


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results)  # 1. loss 2. accuracy 가 나온다
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)


print(y_test.shape) 



#acc = accuracy_score(y_test, y_predict)
# print('acc: ', acc)

# print(y_test.shape)
# print(y_predict.shape)
# print(y_test[:5])
# print(y_predict[:5])

print(y_test)
print(y_test.shape)
y_test_acc = np.argmax(y_test, axis =1)
y_pred = np.argmax(y_pred,axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score :', acc)

