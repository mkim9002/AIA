import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

#1. 데이터 
datasets = load_iris()
# datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델구성
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3. 컴파일, 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)



#4. 평가, 예측
# result = model.score(x_test, y_test)
# # print("model.score:", result)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score:", acc)
# print(model, ":", model.feature_importances_)


# 그림그리기
import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)
    
plt.subplot(2, 2, 1)
plot_feature_importances(model1)

plt.subplot(2, 2, 2)
plot_feature_importances(model2)

plt.subplot(2, 2, 3)
plot_feature_importances(model3)

plt.subplot(2, 2, 4)
plot_feature_importances(model4)

plt.show()

