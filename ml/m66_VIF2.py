

from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

#1. dataset
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()
# df.plot.box()
# plt.show()


y = df['target']
x = df.drop(['target'], axis=1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


##다중공선성 -> y 넣지 않는다
vif = pd.DataFrame()
vif['variables'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
print(vif)
#     variables       vif
# 0      MedInc  2.501295
# 1    HouseAge  1.241254
# 2    AveRooms  8.342786
# 3   AveBedrms  6.994995
# 4  Population  1.138125
# 5    AveOccup  1.008324
# 6    Latitude  9.297624
# 7   Longitude  8.962263

# x = x.drop(['Latitude'], axis=1)
x = x.drop(['Latitude', 'Longitude'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2,
    # stratify=y,
)
scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)


#2. model
model = RandomForestRegressor(random_state=337)

#3. 훈련
model.fit(x_train,y_train)

#4, 평가 예측
results = model.score(x_test,y_test)
print("결과 :", results)




