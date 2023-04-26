from sklearn.covariance import EllipticEnvelope
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
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

# Outlier detection
outliers = EllipticEnvelope(contamination=0.1)
outliers.fit(x)
mask = outliers.predict(x) == 1  # 1 for inliers, -1 for outliers
x = x[mask]
y = y[mask]

# Imputation and dimensionality reduction
imputers = [SimpleImputer(strategy='mean'), 
            SimpleImputer(strategy='median'),
            SimpleImputer(strategy='most_frequent'), 
            SimpleImputer(strategy='constant', fill_value=0),
            SimpleImputer(strategy='constant', fill_value=7777),
            KNNImputer(),
            IterativeImputer(estimator=RandomForestRegressor(random_state=123)),
            IterativeImputer(estimator=DecisionTreeRegressor(random_state=123)),
            IterativeImputer(estimator=XGBRegressor(random_state=123))]

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
