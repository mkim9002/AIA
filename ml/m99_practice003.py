import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

#1.data
x,y = fetch_california_housing(return_X_y=True)

n_split = 5
kf = KFold(n_splits=n_split, shuffle=True, random_state=123)

#2. model
model = RandomForestRegressor()

#3. coom[ile]
scores = cross_val_score(model,x,y, cv=kf)
print(scores)

print('ACC :', scores, '\n mean of cross_val_score :',round(np.mean(scores)))