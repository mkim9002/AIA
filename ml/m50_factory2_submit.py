import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score
path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'
submission = pd.read_csv(path + 'answer_sample.csv')
# _data
#     TRAIN
#     TRAIN_AWS
#     TEST_INPUT
#     TEST_AWS
#     Meta
#     answer_sample.csv

#변수명과 random state바꿔라


train_files = glob.glob(path + "TRAIN/*.csv")
# print(train_files)
test_input_files = glob.glob(path + 'test_input/*.csv') 
# print(test_input_files)


################## train folder ##########################
li = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
# print(li) #35064...7개
# print(len(li))  #17

train_dataset = pd.concat(li, axis=0,
                          ignore_index=True)
# print(train_dataset)
#[596088 rows x 4 columns]

################## test folder #################################

li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
# 
# 

test_input_dataset = pd.concat(li, axis=0,
                          ignore_index=True)
#
#

################# 측정소 라벨 인코더 #########################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])
print(train_dataset)  #[596088 rows x 5 columns]
print(test_input_dataset)  #[131376 rows x 5 columns]
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1)
print(train_dataset)  #[596088 rows x 4 columns]
print(test_input_dataset)  #[131376 rows x 4 columns]

################### 일시-> 월, 일, 시간 으로 분리 !!     #######################
# 12-31 21:00 -> 12 와 21 추출
# print(train_dataset.info())  


train_dataset['month'] = train_dataset['일시'].str[:2]
print(train_dataset['month'])
train_dataset['hour'] = train_dataset['일시'].str[6:8]
# print(train_dataset['hour'])
train_dataset = train_dataset.drop(['일시'], axis=1)
print(train_dataset)   #[596088 rows x 5 columns]

### str -> int
############## test ##############################################

test_input_dataset['month'] = test_input_dataset['일시'].str[:2]
print(train_dataset['month'])
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8]
# print(test_input_dataset['hour'])
test_input_dataset = test_input_dataset.drop(['일시'], axis=1)
print(test_input_dataset)   #[596088 rows x 5 columns]



# print(train_dataset.info())
print(test_input_dataset.info())

## str -> int

# train_dataset['month'] = pd.to_numeric(train_dataset['month'])
# train_dataset['month'] = pd.to_numeric(train_dataset['month']).astype('int8')
train_dataset['month'] = train_dataset['month'].astype('int8')
train_dataset['hour'] = train_dataset['hour'].astype('int8')

test_input_dataset['month'] = test_input_dataset['month'].astype('int8')
test_input_dataset['hour'] = test_input_dataset['hour'].astype('int8')

print(train_dataset.info())
print(test_input_dataset.info())


########################## 결축지 제거 PM2.5에 15542 개 있다 ########################

train_dataset = train_dataset.dropna()
print(train_dataset.info())

################ 제출용 x_submit 준비 ########################################
x_submit = test_input_dataset[test_input_dataset.isna().any(axis=1)]
# 결측지 있는 데이터 행만 추출
print(x_submit)  #[78336 rows x 5 columns]
print(x_submit.info())
    
#  0   연도      78336 non-null  int64
#  1   PM2.5   0 non-null      float64
#  2   locate  78336 non-null  int32
#  3   month   78336 non-null  int8
#  4   hour    78336 non-null  int8
x_submit = x_submit.drop(['PM2.5'], axis=1)
print(x_submit)
 
##### 시즌 -파생피쳐도 생각!!!

y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis=1)
print(x, '\n', y)


x_train, x_test, y_train, y_test= train_test_split(
    x,y, train_size=0.8, random_state=282, shuffle=True
) 

parameter = {'n_estimators' : 10000,
              'learning_rate' : 0.07,   # 이게 성능이 가장 좋다
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 0.7,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 0.01,
              'random_state' : 1234,
              'verbose' :0,
              'n_jobs': '-1'
              }

#2. 모델
model = XGBRegressor()

#3. 컴파일, 훈련
model.set_params(**parameter,
                 eval_metrics='mae',
                 early_stopping_rounds=200,
                 )
start_time = time.time()
model.fit(x_train, y_train, verbose =1,
          eval_set = [(x_train,y_train), (x_test,y_test)]
    
)

end_time=time.time()
print("걸린시간 :", round(end_time-start_time,2),"초")



#4 평가 예측
y_predict = model.predict(x_test)
y_submit = model.predict(x_submit)

result = model.score(x_test, y_test)
print("model.score: ", result)

r2= r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

mae= mean_absolute_error(y_test, y_predict)
print("mae 스코어 : ", mae)

####### 제출 파일 만들기 #######
answer_sample_csv = pd.read_csv(path + 'answer_sample.csv',
                                index_col=None, header=0, encoding='')

answer_sample_csv['PM2.5'] = y_submit
print(answer_sample_csv)

answer_sample_csv.to_csv(path + 'm50_factory2_submit.csv',
                         index=None)









# # Update the submission dataframe with the predicted values
# submission = submission.reindex(range(len(y_predict)))
# submission['PM2.5'] = y_predict

# # Save the results
# submission.to_csv(save_path + 'submit41.csv', index=False)
# print(f'Results saved to {save_path}submit.csv')





