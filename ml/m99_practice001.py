import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_estimators=1000, learning_rate=0.3, max_depth=2, gamma=0, min_child_weight=1,
                     subsample=0.4, colsample_bytree=0.8, colsample_bylevel=0.7, colsample_bynode=0.9,
                     reg_alpha=0, reg_lambda=0.01, random_state=1234)

# 3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='rmse',
          verbose=0)
