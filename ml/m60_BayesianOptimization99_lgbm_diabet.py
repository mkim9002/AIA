import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization

# 1. 데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets.target

# 2. 함수 정의
def y_function(max_depth, min_samples_leaf, min_samples_split, min_weight_fraction_leaf, max_features):
    # 모델 정의
    model = RandomForestRegressor(max_depth=int(max_depth),
                                  min_samples_leaf=int(min_samples_leaf),
                                  min_samples_split=int(min_samples_split),
                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                  max_features=max_features,
                                  random_state=337)

    # 데이터 전처리 (PCA)
    pca = PCA(n_components=5)
    x_pca = pca.fit_transform(x)

    # 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=337)

    # 모델 학습
    model.fit(x_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    return r2

# 3. Bayesian Optimization
param_bounds = {
    'max_depth': (3, 16),
    'min_samples_leaf': (1, 50),
    'min_samples_split': (2, 200),
    'min_weight_fraction_leaf': (0, 0.5),
    'max_features': (0.1, 1)
}

optimizer = BayesianOptimization(
    f=y_function,
    pbounds=param_bounds,
    random_state=337
)

optimizer.maximize(init_points=5, n_iter=1)

# 최적화 결과 출력
print("===================================")
print("최적화 :", optimizer.max)
print("r2  :",r2_score)
