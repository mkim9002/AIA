import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#1. data
x,y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train,y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True,random_state=337
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = [
    {'rf__n_estimators' : [100,200], 'rf__max_depth' : [6,8,10,12], 'rf__min_samples_leaf' : [3,5,7,10]},
    {'rf__max_depth' : [6,8,10,12], 'rf__min_samples_leaf' : [3,5,7,10]},
    {'rf__min_samples_leaf' : [3,5,7,10], 'rf__min_samples_split' : [2,3,5,10]},
    {'rf__min_samples_split' : [2,3,5,10]},
  ]



#2. 모델
# model = RandomForestClassifier()
# model =make_pipeline(StandardScaler(), SVC())
pipe =Pipeline([("sdf",StandardScaler()), ("rf",RandomForestClassifier())])         #Pipeline List 형태로 써야 한다 List안엔 튜플

model =GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=1)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가 예측
result = model.score(x_test,y_test)
print("model.score :", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)