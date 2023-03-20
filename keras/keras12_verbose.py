from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# Split the dataset into train and test sets with a train size of 0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# 2. 모델 Create a neural network model with 3 hidden layers
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3.컴파일, 훈련 Compile the model with mean squared error loss and adam optimizer
model.compile(loss='mse', optimizer='adam')

# Train the model for 500 epochs with a batch size of 32
model.fit(x_train, y_train, epochs=10, batch_size=2, verbose='auto')


'''
# 평가, 예측 Evaluate the model on the test set
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

# Make predictions on the test set and calculate R2 score
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)



# 실습
# 1. train 0.7
# 2. R2 0.8 이상
'''

#