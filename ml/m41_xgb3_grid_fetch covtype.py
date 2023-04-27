import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler 
from sklearn.model_selection import KFold, StratifiedKFold


# 1. 데이터
datasets = fetch_covtype()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(581012, 54) (581012,)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 55):
    if n == 0:
        x_pca = x
    else:
        pca = PCA(n_components=n)
        x_pca = pca.fit_transform(x)
    print(x_pca.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, train_size=0.8, shuffle=True, random_state=123
    )

    scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits =5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = {'n_estimators' : [200],
              'learning_rate' : [0.1],
              'max_depth': [3],
              'gamma': [0],
              'min_child_weight': [1],
              'subsaample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1],
              'reg_alpha': [1],
              'reg_lambda': [1]
              }

# 'n_estimators' : [100, 200, 300, 400, 500, 1000]  /디폴트100/ 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] / 디폴트0.3 / 0~1/ eta
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] /디폴트6/ 0~inf/ 정수 
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] / 디폴트 0/ 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] /디폴트 1/ 0~inf
# 'subsaample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] /디폴트 0/ 0~inf/ L1 절대값 가중치 규제/ alpha
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10] /디폴트 1/ 0~inf/ L2 제곱 가중치 규제/ lambda





# 2. 모델
model = RandomForestRegressor(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
results = model.score(x_test, y_test)
print(f"n_components={n}, 결과 : {results}")


############