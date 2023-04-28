#Dancon_wine 
#결측치/ 원핫인코딩, 데이터분리, 스케일링/ 함수형,dropout
#다중분류 - softmax, categorical
#와인의 퀄러티를 맞춰라.퀄러티는 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 전처리
path = 'c:/study/_data/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

print(train_csv['quality'].value_counts().sort_index())

# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5


x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality'] - 3  # Shift class labels to start from 0

# 1-3 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=640874)

parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.09,
              'max_depth': 37,
              'gamma': 1,
              'min_child_weight': 1,
              'subsample': 0.7,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.7,
              'colsample_bynode': 1,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'random_state' : 337,
            #   'eval_metric' : 'error'
              }

# 2. 모델 구성
model = XGBClassifier(parameters, n_estimators=100, random_state=640874)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 및 예측
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print('Train Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)

# 5. 예측 및 제출 파일 생성
test_pred = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = test_pred + 3  # Shift class labels back to the original range
submission.to_csv(path + 'submit_wine_xgboost03.csv')
