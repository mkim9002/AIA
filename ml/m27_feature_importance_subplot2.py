# for 문으로 ,xgboost 잔소리 지우기
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
datasets = load_iris()
# datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델 구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

#3. 컴파일, 훈련 및 평가
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} accuracy_score: {acc:.4f}")
    print(f"{model.__class__.__name__} feature importances:", model.feature_importances_)

    # 그림그리기
    def plot_feature_importances(model):
        n_features = datasets.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), datasets.feature_names)
        plt.xlabel('Feature Importances')
        plt.ylabel('Features')
        plt.ylim(-1, n_features)
        plt.title(model.__class__.__name__)

    plot_feature_importances(model)
    plt.show()

