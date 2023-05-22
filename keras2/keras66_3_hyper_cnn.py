import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.

# 2. 모델 구성
def build_model(drop=0.3, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(32, (3, 3), activation=activation, padding='same', name='conv1')(inputs)
    x = MaxPool2D(pool_size=(2, 2), name='pool1')(x)
    x = Conv2D(64, (3, 3), activation=activation, padding='same', name='conv2')(x)
    x = MaxPool2D(pool_size=(2, 2), name='pool2')(x)
    x = Flatten(name='flatten')(x)
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
    return {'batch_size': batch_sizes,
            'optimizer': optimizers,
            'drop': dropouts,
            'activation': activations}

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
