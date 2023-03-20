from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score #분류=> 결과지표 'accuracy_score' 떠올라야함
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터 
datasets = fetch_covtype()
print(datasets.DESCR) # Classes 7

print(datasets.feature_names) 

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
print(x)
print(y)  #[5 5 2 ... 3 3 3]
print('y의 라벨값 :', np.unique(y)) #y의 라벨값 : [1 2 3 4 5 6 7]

##########데이터 분리전에 one-hot encoding하기############
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y)
# print(y.shape) #(581012, 8)

#######################################################
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()

###########################
#데이터분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, 
    train_size=0.8,
    stratify=y    #통계적으로 (y값,같은 비율로)
)
print(y_train)   #                               
print(np.unique(y_train, return_counts=True)) 
#(array([0., 1.], dtype=float32), array([3253663,  464809], dtype=int64))


#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=54))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#EarlyStopping추가
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=2000, validation_split=0.2, verbose=1 )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results:', results)  # 1. 
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis =1)
y_pred = np.argmax(y_pred,axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score :', acc)

