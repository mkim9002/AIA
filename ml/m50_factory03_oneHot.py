import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score

path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'
submission = pd.read_csv(path + 'answer_sample.csv')

train_files = glob.glob(path + "TRAIN/*.csv")
test_input_files = glob.glob(path + 'test_input/*.csv')

################## train folder ##########################
li = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8-sig')
    li.append(df)

train_dataset = pd.concat(li, axis=0, ignore_index=True)

################## test folder #################################
li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8-sig')
    li.append(df)

test_input_dataset = pd.concat(li, axis=0, ignore_index=True)

################# 측정소 라벨 인코더 #########################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1)

################### 일시-> 월, 일, 시간 으로 분리 !! #######################
train_dataset['month'] = train_dataset['일시'].str[:2]
train_dataset['hour'] = train_dataset['일시'].str[6:8]
train_dataset = train_dataset.drop(['일시'], axis=1)

test_input_dataset['month'] = test_input_dataset['일시'].str[:2]
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8]
test_input_dataset = test_input_dataset.drop(['일시'], axis=1)

train_dataset['month'] = train_dataset['month'].astype('int8')
train_dataset['hour'] = train_dataset['hour'].astype('int8')

test_input_dataset['month'] = test_input_dataset['month'].astype('int8')
test_input_dataset['hour'] = test_input_dataset['hour'].astype('int8')

train_dataset = train_dataset.dropna()

##### 시즌 -파생피쳐도 생각!!!

# Perform one-hot encoding on categorical features
train_dataset = pd.get_dummies(train_dataset, columns=['month', 'hour', 'locate'])
test_input_dataset = pd.get_dummies(test_input_dataset, columns=['month', 'hour', 'locate'])

y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=1, shuffle=True)

parameter = {'n_estimators': 20000,
              'learning_rate': 0.7,
              'max_depth': 3,
              'n_jobs': -1
              }

# 2. 모델
model = XGBRegressor()

# 3. 컴파일, 훈련
model.set_params(**parameter,
                 eval_metric='mae',
                 early_stopping_rounds=200
                 )

start_time = time.time()
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=True
          )
end_time = time.time()

print("걸린시간:", round(end_time - start_time, 2), "초")

# 4. 평가 예측
y_predict = model.predict(x_test)

result = model.score(x_test, y_test)
print("model.score:", result)

r2 = r2_score(y_test, y_predict)
print("r2 스코어:", r2)

mae = mean_absolute_error(y_test, y_predict)
print("mae 스코어:", mae)

# 제출 파일 만들기
x_submit = test_input_dataset[test_input_dataset.isna().any(axis=1)]
x_submit = x_submit.drop(['PM2.5'], axis=1)

y_submit = model.predict(x_submit)

answer_sample_csv = pd.read_csv(path + 'answer_sample.csv', index_col=None, header=0)
answer_sample_csv['PM2.5'] = y_submit
answer_sample_csv.to_csv(path + '004.csv', index=None)
