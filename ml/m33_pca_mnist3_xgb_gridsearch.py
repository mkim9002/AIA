# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch 를 쓸것

# m33_2 결과를 뛰어 넘어라!!

from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# PCA를 사용하여 0.95, 0.99, 0.999, 1.0 이상의 variance ratio를 가지는 feature 개수를 구함
pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d95 = np.argmax(cumsum >= 0.95) + 1
d99 = np.argmax(cumsum >= 0.99) + 1
d999 = np.argmax(cumsum >= 0.999) + 1
d100 = np.argmax(cumsum == 1.0) + 1

# 각각의 feature 개수에 대해 XGBoost 모델의 정확도 계산
for d in [d95, d99, d999, d100]:
    pca = PCA(n_components=d)
    x_pca = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    model = XGBClassifier()

    # RandomizedSearchCV를 이용하여 최적의 하이퍼파라미터를 찾음
    parameters = {
        'n_estimators': [1, 2, 3],
        'learning_rate': [0.1, 0.3, 0.001, 0.01],
        'max_depth': [4, 5, 6],
        'colsample_bytree': [0.6, 0.9, 1],
        'colsample_bylevel': [0.6, 0.7, 0.9]
    }

    search = RandomizedSearchCV(model, parameters, n_iter=10, n_jobs=-1, cv=3, random_state=42)
    search.fit(x_train, y_train)

    print(f"PCA {d}: Best params: {search.best_params_}")
    print(f"PCA {d}: Best score: {search.best_score_:.4f}")
    print(f"PCA {d}: Test score: {search.score(x_test, y_test):.4f}")



# parameters = [
#     {"_estimators": [100, 200, 300],
#      "learning_rate": [0.1, 0.3, 0.001, 0.01],
#     "max_depth": [4, 5, 6]},

#     {"_estimators": [90, 100, 110],
#     "learning_rate": [0.1, 0.001, 0.01],
#     "max _depth": [4,5,6],
#     "colsample_bytree": [0.6, 0.9, 1]},

#     {"_estimators": [90, 110],
#     "learning rate": [0.1, 0.001, 0.5],
#     "max _depth": [4,5,6],
#     "colsample _bytree": [0.6, 0.9, 1]},

#     {"colsample_bylevel": [0.6, 0.7, 0.9]}
# ]

# n_jobs = -1
# tree_method ='gpu_hist'
# predictor ='gpu_predictor'
# gpu_id =0,