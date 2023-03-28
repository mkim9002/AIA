#2. 모델 구성
from keras.models import load_model
model = load_model("savetest01.h5")

from keras.layers import Dense # 추가
model.add(Dense(1)) # 추가

model.summary()

'''
에러 발생.
'''