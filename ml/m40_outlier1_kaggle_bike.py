#이상치 를 제거 하던지 최고의 성능을 내던지
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA

#1. 데이터
path = 'C:/study/_data/kaggle_bike/'
path_save = 'C:/study/_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
# print(train_csv.isnull().sum()) 
#결측치 없음

###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

# 이상치 제거
outliers = EllipticEnvelope(contamination=.2)
outliers.fit(x)
results = outliers.predict(x)
x = x[results == 1]
y = y[results == 1]

print(x.shape)   #(8699, 8)

for i in range(8, 0, -1):
    pca=PCA(n_components=i)
    x_pca = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, train_size=0.8, random_state=123, shuffle=True)
    model = RandomForestRegressor(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i}, 결과: {result}")
