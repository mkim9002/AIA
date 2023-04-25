# 오늘 배운것 모두 이용 하여 성능비교 결측지 처리
# imputer = SimpleImputer(strategy='mean')                 #디폴트 평균!!
# imputer = SimpleImputer(strategy='median')               #중위값
# imputer = SimpleImputer(strategy='most_frequent')        #최빈값 (갯수가 같을 경우 가장 작은값)
# imputer = SimpleImputer(strategy='constant')             #끊임없음 0 들어감
# imputer = SimpleImputer(strategy='constant',fill_value=7777)             #0 들어감
# imputer = KNNImputer()                                   #평균값
# imputer = IterativeImputer()
# imputer = IterativeImputer(DecisionTreeRegressor())
# imputer = IterativeImputer(estimator=XGBRegressor())

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer    #결축지에 대한 책임을 돌릴것 같아
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


#1. 데이터
path = 'C:/study/_data/ddarung/'
path_save = 'C:/study/_save/dacon_ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


###결측치제거### 
train_csv = train_csv.dropna() 

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

imp_mean = SimpleImputer(strategy='mean')
imp_median = SimpleImputer(strategy='median')
imp_frequent = SimpleImputer(strategy='most_frequent')
imp_constant_0 = SimpleImputer(strategy='constant', fill_value=0)
imp_constant_7777 = SimpleImputer(strategy='constant', fill_value=7777)
imp_knn = KNNImputer()
imp_iterative_rf = IterativeImputer(estimator=RandomForestRegressor(random_state=123))
imp_iterative_dt = IterativeImputer(estimator=DecisionTreeRegressor(random_state=123))
imp_iterative_xgb = IterativeImputer(estimator=XGBRegressor(random_state=123))

imputers = [imp_mean, imp_median, imp_frequent, imp_constant_0, imp_constant_7777, imp_knn, imp_iterative_rf, imp_iterative_dt, imp_iterative_xgb]

for imputer in imputers:
    print(f"Imputer: {imputer}")
    for i in range(9, 0, -1):
        pca = PCA(n_components=i)
        x_imputed_pca = pca.fit_transform(imputer.fit_transform(x))
        x_train, x_test, y_train, y_test = train_test_split(x_imputed_pca, y, train_size=0.8, random_state=123, shuffle=True,)
        model = RandomForestRegressor(random_state=123)
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(f"    n_components={i}, 결과: {result}")   


