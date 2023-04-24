#실습
#피쳐 임포턴스 가 전체 중요도에서  하위 20-25% 컬럼제거
#재구성 후
#모델을 돌려서 결과 도출
#기존 모델들과 성능비교


import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#1. 데이터 
datasets = fetch_covtype()

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

#4. 모델 재학습 및 평가
x_train_imp = x_train[:,model.feature_importances_ > np.percentile(model.feature_importances_, 25)] 
x_test_imp = x_test[:,model.feature_importances_ > np.percentile(model.feature_importances_, 25)] 

for i, model in enumerate(models):
    model.fit(x_train_imp, y_train)
    y_pred = model.predict(x_test_imp)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} accuracy_score: {acc:.4f}")



#결과비교
# 예)
#1. DecesionTree
#기존 acc : 0.9298
# 컬럼삭제후 acc :0.9298

#2. RandomForest
#기존 acc : 0.9561
# 컬럼삭제후 acc :0.9561

#3. GradientDecentBoosting
#기존 acc : 0.9825
# 컬럼삭제후 acc :0.9737

#4. XGBoost
#기존 acc : 0.9737
# 컬럼삭제후 acc :0.9737

