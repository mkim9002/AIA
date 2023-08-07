import autokeras as ak
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import time
######
# 1. 데이터 
datasets = load_breast_cancer()
print(datasets)

#print(datasets)
print(datasets.DESCR) #판다스 : .describe()
print(datasets.feature_names)   #판다스 :  .columns()

x = datasets ['data']
y = datasets.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x.shape, y.shape) #(569, 30) (569,) feature,열,columns 는 30
#print(y) #1010101 은 암에 걸린 사람과 아님

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
model = ak.StructuredDataClassifier(
    overwrite=False,  # True일 경우 모델탐색을 처음부터 다시 함(속도 느림) -> 성능이 너무 안좋을때 True사용하기 / 보통이상 성능이면 True일때 더 성능향상이 됨  
    max_trials=2,  # 디폴트 False
)

# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.15)
end = time.time()

# 최적의 모델 출력
best_model = model.export_model()
print(best_model.summary())

# 최적의 모델 저장
path = './_save/autokeras/'
best_model.save(path + "keras62_autokeras1.h5")

# 4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과:', results)
