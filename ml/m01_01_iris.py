import numpy as np
from sklearn.datasets import load_iris

#1. data
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x,y = load_iris(return_X_y=True)

print(x.shape, y.shape) #(150, 4) (150,)

#.model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

# model = LinearSVC(C=1) #0.9666666666666667
# model = LogisticRegression() #0.9733333333333334
# model = DecisionTreeClassifier()  # 1.0
model = RandomForestRegressor()  #0.992262

# #3. compile,epochs
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer = 'adam',
#               metrics=['acc'])
# model.fit(x,y, epochs=100, validation_split=0.2)
model.fit(x,y)


# #. evaluate, prediction
# results = model.evaluate(x,y)
# print(results)
results = model.score(x,y)

print(results)  #

