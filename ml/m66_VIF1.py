import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = {
    'size': [30, 35, 40, 45, 50, 45],
    'rooms': [2, 2, 3, 3, 4, 3],
    'window': [2, 2, 3, 3, 4, 3],
    'year': [2010, 2015, 2010, 2015, 2010, 2014],
    'price': [1.5, 1.8, 2.0, 2.2, 2.5, 2.3]
}

df = pd.DataFrame(data)

print(df)

x = df[['size', 'rooms', 'year', 'window']]
y = df['price']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x - x_scaled)

vif = pd.DataFrame()
vif['variables'] = x.columns

vif['vif'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]

print(vif)

print("================================ 사이즈 제거전 =========================")
lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y, y_pred)
print("r2 :", r2)

print("================================ rooms 제거 =========================")

x_scaled = df[['size', 'year', 'window']]

vif2 = pd.DataFrame()
vif2['variables'] = x_scaled.columns

vif2['vif'] = [variance_inflation_factor(x_scaled.values, i) for i in range(x_scaled.shape[1])]

print(vif2)

lr = LinearRegression()
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y, y_pred)
print("r2 :", r2)
