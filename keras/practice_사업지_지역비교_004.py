import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# 데이터 로드
data1 = np.loadtxt('data1.txt')
data2 = np.loadtxt('data2.txt')
data3 = np.loadtxt('data3.txt')
data4 = np.loadtxt('data4.txt')

# 데이터 정규화
scaler = StandardScaler()
data1 = scaler.fit_transform(data1)
data2 = scaler.fit_transform(data2)
data3 = scaler.fit_transform(data3)
data4 = scaler.fit_transform(data4)

# 모델 학습
model1 = LinearRegression()
model1.fit(data1, labels)

model2 = RandomForestRegressor()
model2.fit(data2, labels)

model3 = KNeighborsRegressor()
model3.fit(data3, labels)

model4 = RandomForestRegressor()
model4.fit(data4, labels)

# 앙상블 모델 예측
pred1 = model1.predict(data1)
pred2 = model2.predict(data2)
pred3 = model3.predict(data3)
pred4 = model4.predict(data4)

ensemble_pred = (pred1 + pred2 + pred3 + pred4) / 4