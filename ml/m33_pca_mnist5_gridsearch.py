from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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

# 각각의 feature 개수에 대해 RandomForestClassifier 모델의 정확도 계산
for d in [d95, d99, d999, d100]:
    pca = PCA(n_components=d)
    x_pca = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    # GridSearchCV를 사용하여 RandomForestClassifier의 하이퍼파라미터 튜닝
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(rf, param_grid, cv=5)
    clf.fit(x_train, y_train)

    # 최적의 하이퍼파라미터로 모델을 학습하고 정확도를 계산
    best_rf = clf.best_estimator_
    acc = best_rf.score(x_test, y_test)
    print(f"PCA {d}: {acc:.4f}, Best parameters: {clf.best_params_}")
