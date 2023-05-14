import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

#1. 데이터
path = 'D:/study/_data/book/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

x = train_csv.drop(['Book-Rating'], axis = 1)
y = train_csv['Book-Rating']

# 카테고리 데이터를 Label Encoding으로 변환
categorical_cols = ['User-ID', 'Book-ID', 'Location', 'Book-Title', 'Book-Author', 'Publisher']
le = LabelEncoder()
for col in categorical_cols:
    le.fit(x[col])
    x[col] = le.transform(x[col])
    test_csv[col] = le.transform(test_csv[col])
    
# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.1, random_state= 331)

# XGBoost 모델 학습
model = XGBRegressor(enable_categorical=True)
model.fit(x_train, y_train)
# 전체 데이터셋을 대상으로 원-핫 인코딩을 수행
all_data = pd.concat([x_train, x_test, test_csv], axis=0)
all_data = pd.get_dummies(all_data, columns=categorical_cols)

# 훈련 데이터와 테스트 데이터를 다시 분리
x_train = all_data[:len(x_train)]
x_test = all_data[len(x_train):(len(x_train) + len(x_test))]
test_csv = all_data[(len(x_train) + len(x_test)):]

# XGBoost 모델 학습 및 예측
model = XGBRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", rmse)

# Submission
save_path = 'D:/study/_save/book/'
y_sub=model.predict(test_csv)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]]=y_sub
sample_submission_csv.to_csv(save_path + 'book_' + date + '.csv', index=False, float_format='%.0f')

# 예측 및 평가
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print("RMSE : ", np.sqrt(mse))
