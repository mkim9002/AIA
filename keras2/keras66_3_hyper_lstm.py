import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Input


#####
# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2]).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]).astype('float32') / 255.

# 2. 모델 구성
def build_model(drop=0.3, optimizer='adam', activation='relu', lstm_units=64, lr=0.001):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]), name='input')
    x = LSTM(lstm_units, activation=activation, name='lstm')(inputs)
    x = Dropout(drop)(x)
    x = Dense(512, activation=activation, name='hidden1')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden4')(x)
    outputs = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batch_sizes = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    lstm_units = [32, 64, 128]
    return {'batch_size': batch_sizes,
            'optimizer': optimizers,
            'drop': dropouts,
            'activation': activations,
            'lstm_units': lstm_units}

hyperparameters = create_hyperparameter()
print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=1, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=3)
end = time.time()
print("걸린 시간:", end - start)
print("model.best_params_:", model.best_params_)
print("model.best_estimator_:", model.best_estimator_)
print("model.best_score_:", model.best_score_)
print("model.score:", model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score:', accuracy_score(y_test, y_predict))
