from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

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

# 각각의 feature 개수에 대해 xgboost 모델의 정확도 계산
for d in [d95, d99, d999, d100]:
    pca = PCA(n_components=d)
    x_pca = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    # xgboost 모델 훈련
    parameters = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.3, 0.001, 0.01],
        'max_depth': [4, 5, 6],
        'colsample_bytree': [0.6, 0.9, 1.0],
        'subsample': [0.6, 0.8, 1.0]
    }
    xgb_model = xgb.XGBClassifier(n_jobs=-1, tree_method='gpu_hist', predictor='gpu_predictor')
    clf = GridSearchCV(xgb_model, parameters, scoring='accuracy', n_jobs=-1, cv=3)
    clf.fit(x_train, y_train)

    acc = clf.score(x_test, y_test)
    print(f"PCA {d}: {acc:.4f}")
