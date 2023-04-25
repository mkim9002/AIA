import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# [실습]
# for문으로 pca 10~1개까지 

#1. 데이터
path = 'C:/study/_data/ddarung/'
path_save = 'C:/study/_save/dacon_ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


###결측치제거### 
train_csv = train_csv.dropna() 

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
# print(train_csv.shape)  #(1328, 9)

for i in range(9, 0, -1):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestRegressor(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")

'''
n_coponets=9,  결과: 0.6936276869466653 
n_coponets=8,  결과: 0.6784454950446617 
n_coponets=7,  결과: 0.6754821123352452 
n_coponets=6,  결과: 0.6935666626368776 
n_coponets=5,  결과: 0.62252151710199 
n_coponets=4,  결과: 0.29494787387754695 
n_coponets=3,  결과: 0.2353812467908073 
n_coponets=2,  결과: 0.09707754599708784 
n_coponets=1,  결과: -0.26917844142607183 
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