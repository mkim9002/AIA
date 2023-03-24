#save_model과 비교

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))
print(x)

# 보스톤 1.2부터 임포트 안된다
#pip uninstall scikit-learn
#pip install scikit-learn==1.1

print(np.min(x), np.max(x))  #0.0 711.0
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x)) #0.0 1.0

x_train, x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state=333
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train) #아래 코드와 같다
# x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max)


print(x.shape, y.shape) #(506, 13) (506,)

#2. 모델


input1 = Input(shape=(13), name='h1')
dense1 = Dense(30, name='h2')(input1)
dense2 = Dense(20, name='h3')(dense1)
dense3 = Dense(10, name='h4')(dense2)
output1 = Dense(1, name='h5')(dense3)
model = Model(inputs=input1, outputs=output1)



#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                   verbose=1, 
                #    restore_best_weights=True
                   )
                   
mcp = ModelCheckpoint(monitor='val_loss', mode= 'auto',
            verbose=1,
            save_best_only=True,
            filepath='./_save/MCP/keras27_MCP.hdf5'
            )

model.fit(x_train, y_train, epochs=10000,
          callbacks=[es, mcp],
          validation_split=0.2)
        


model.save('./_save/MCP/keras27_3_save_model.h5')
#4/ 평가 예측
from sklearn.metrics import r2_score

print("================1. 기본출력==================")
loss = model.evaluate(x_test,y_test, verbose=0)
print('loss :',loss )
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 스코어 :', r2)

print("================2. load_model 출력==========")
model2 = load_model('./_save/MCP/keras27_3_save_model.h5')
loss = model2.evaluate(x_test,y_test, verbose=0)
print('loss :',loss )
y_predict = model2.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 스코어 :', r2)

print("================3. MCP 출력==================")
model3 = load_model('./_save/MCP/keras27_MCP.hdf5')
loss = model3.evaluate(x_test,y_test, verbose=0)
print('loss :',loss )
y_predict = model3.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 스코어 :', r2)

# ================1. 기본출력==================
# loss : 24.09328842163086
# r2 스코어 : 0.7543480716449408
# ================2. load_model 출력============
# loss : 24.09328842163086
# r2 스코어 : 0.7543480716449408
# ================3. MCP 출력==================
# loss : 24.012470245361328
# r2 스코어 : 0.7551721012124069

