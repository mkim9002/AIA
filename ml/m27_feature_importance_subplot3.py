import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

#1. 데이터 
datasets = load_breast_cancer()
# datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델 구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

#3. 컴파일, 훈련 및 평가
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
ax = axes.ravel()

for i, model in enumerate(models):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} accuracy_score: {acc:.4f}")
    print(f"{model.__class__.__name__} feature importances:", model.feature_importances_)

    # 그림그리기
    n_features = datasets.data.shape[1]
    ax[i].barh(np.arange(n_features), model.feature_importances_, align='center')
    ax[i].set_yticks(np.arange(n_features))
    ax[i].set_yticklabels(datasets.feature_names)
    ax[i].set_xlabel('Feature Importances')
    ax[i].set_ylabel('Features')
    ax[i].set_ylim(-1, n_features)
    ax[i].set_title(model.__class__.__name__)

plt.show()
