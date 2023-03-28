#2. 모델 구성
from keras.models import load_model
model = load_model("savetest01.h5")

from keras.layers import Dense
model.add(Dense(1, name='dense_x')) # 수정

model.summary()

