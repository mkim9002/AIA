import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
from sklearn.covariance import EllipticEnvelope

#1. 데이터 
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'
train_csv= pd.read_csv(path+'train.csv', index_col=0)
test_csv= pd.read_csv(path+'test.csv', index_col=0)

# 이상치 제거
outliers = EllipticEnvelope(contamination=.2)
outliers.fit(train_csv)
results = outliers.predict(train_csv)
train_csv = train_csv[results == 1]

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

# 결측치 처리
imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=123))
x_imputed = imputer.fit_transform(x)

for i in range(8, 0, -1):
    pca = PCA(n_components=i)
    x_pca = pca.fit_transform(x_imputed)
    x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestClassifier(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_components={i},  결과: {result} ")


'''
n_coponets=8,  결과: 0.7404580152671756 
n_coponets=7,  결과: 0.7480916030534351 
n_coponets=6,  결과: 0.7480916030534351 
n_coponets=5,  결과: 0.6946564885496184 
n_coponets=4,  결과: 0.6946564885496184 
n_coponets=3,  결과: 0.7175572519083969 
n_coponets=2,  결과: 0.6641221374045801 
n_coponets=1,  결과: 0.6335877862595419 
'''



# #1. 데이터 
# datasets = load_diabetes()
# print(datasets.feature_names) #sklearn컬럼명 확인 /###pd : .columns
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# x = datasets['data']
# y = datasets.target
# print(x.shape, y.shape)    #(442, 10) (442,)

# #데이터x 컬럼 축소
# pca = PCA(n_components=7)
# x = pca.fit_transform(x)
# print(x.shape)             #(442, 5)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=123, shuffle=True,
# )

# #2. 모델구성 
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=123)

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가,예측
# results = model.score(x_test, y_test)
# print("결과:", results)

'''
#pca_before
결과: 0.5260875642282989
#n_components=7
결과: 0.5141328515687419
'''