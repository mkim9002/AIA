import math
import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score, f1_score
from catboost import CatBoostClassifier,CatBoostRegressor
from hyperopt import hp, fmin, tpe, Trials
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from catboost import CatBoostClassifier, CatBoostRegressor
import time
import warnings
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

path = "d:/study/_data/crime/"
path_save = "d:/study/_save/crime/"



#1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(871393, 9)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (159621, 8)

print(train_csv.info()) 
'''
#   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   월        84406 non-null  int64
 1   요일       84406 non-null  object
 2   시간       84406 non-null  int64
 3   소관경찰서    84406 non-null  int64
 4   소관지역     84406 non-null  float64
 5   사건발생거리   84406 non-null  float64
 6   강수량(mm)  84406 non-null  float64
 9   풍향       84406 non-null  float64
 10  안개       84406 non-null  float64
 11  짙은안개     84406 non-null  float64
 12  번개       84406 non-null  float64
 13  진눈깨비     84406 non-null  float64
 14  서리       84406 non-null  float64
 15  연기/연무    84406 non-null  float64
 16  눈날림      84406 non-null  float64
 17  범죄발생지    84406 non-null  object
 18  TARGET   84406 non-null  int64
dtypes: float64(13), int64(4), object(2)

'''


import matplotlib.pyplot as plt


'''
print(train_csv.columns)
Index(['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)',
       '적설량(cm)', '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림',
       '범죄발생지', 'TARGET'],
      dtype='object')
''' 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의
le_list = ['요일', '범죄발생지']
csv_all = pd.concat([train_csv,test_csv],axis=0)
print(csv_all.columns)
for i,v in enumerate(le_list) :
    csv_all[v] = le.fit_transform(csv_all[v]) # 0과 1로 변화
    train_csv[v] = le.transform(train_csv[v]) # 0과 1로 변화
    test_csv[v] = le.transform(test_csv[v]) # 0과 1로 변화
    print(f'{i}번째',np.unique(csv_all[v]))
    
data_list = ['월', 
             '요일','시간', '소관경찰서', '소관지역','사건발생거리', 
             '강수량(mm)', '강설량(mm)','적설량(cm)', '풍향','안개', 
            '짙은안개', '번개', '진눈깨비', '서리','연기/연무', 
            '눈날림','범죄발생지', 'TARGET']
# '사건발생거리','강수량(mm)', '강설량(mm)','적설량(cm)',
# '짙은안개', '번개', '진눈깨비', '서리','연기/연무', 
# '눈날림','범죄발생지',
for i,v in enumerate(data_list)  : 
    # plt.subplot(4,5,i+1)
    plt.title(i)
    plt.boxplot(np.array(train_csv[v]))
    plt.show()

test_csv = test_csv.drop(['시간','요일','풍향',],axis=1)
x = train_csv.drop(['TARGET','시간','요일','풍향',],axis=1)
print("x : ", x) 

y = train_csv['TARGET']
# y = to_categorical(train_csv['TARGET'])
print(y.shape)


# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# vif = pd.DataFrame()
# vif['variables'] = x.columns
# vif['vif'] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
# print(vif)
'''

   variables       vif
0            월  1.282735
1          요일  1.032687
2          시간  1.000298
3    소관경찰서  1.850206
4      소관지역  1.844475
5  사건발생거리  1.014597
6    강수량(mm)  1.270713
7    강설량(mm)  1.740622
8    적설량(cm)  1.679097
9          풍향  1.056074
10         안개  1.595804
11     짙은안개  1.217778
12         번개  1.330880
13     진눈깨비  1.573900
14         서리  1.640009
15    연기/연무  1.262941
16       눈날림  1.439312
17   범죄발생지  1.001912

'''

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.1,random_state=337,stratify=y
)
# 114 f1 : 0.531572088615093
# 8715 f1 : 0.5325198436204241
# 337 f1 : 0.5217391304347826



####################################### 상관관계 찾기 #####################################
# import matplotlib.pyplot as plt
# import seaborn as sns

# print(test_csv.corr())
# plt.figure(figsize=(10,8))
# sns.set(font_scale=1.2)
# sns.heatmap(train_csv.corr(),square=True, annot=True,cbar=True)
# plt.show()


# ValueError: Object arrays cannot be loaded when allow_pickle=False
# 저장한걸 스케일러 사용할 거면 이런식으로 사용해야함 



'''
n_splits = 5
kfold = StratifiedKFold(
    n_splits = n_splits,#디폴트 5  옛날에는 3이였음 근데 바뀐거면 지금것이 좋다는 의미 
      shuffle=True,  # 처음에 섞고 나서 나중에 잘라서 테스트, 테스트 할때 마다 섞는건 아님 
      random_state=123,
      )

# 2. 모델 구성
search_space = { 
    "n_estimators" : hp.quniform('n_estimators',100,1000,1), # 디폴트 100 / 1 ~ inf / 정수
    "max_depth" : hp.quniform('max_depth',3,16,1),  
    'learning_rate' : hp.uniform('learning_rate', 0.001,1.0), # hp.uniform : 정규 분포 , 중앙 수치가 0.5로 예측 가운데로 갈수록 분포가 많다. 
    "gamma" :  hp.quniform('gamma',0,10,1), # 디폴트 0 / 0 ~ inf 
    "min_child_weight" : hp.uniform('min_child_weight',10,200), # 정수 형태 
    "subsample" : hp.uniform('subsample',0.5,1),
    "colsample_bytree" : hp.uniform('colsample_bytree',0.5,1),
    "colsample_bylevel" : hp.uniform('colsample_bylevel',0.5,1),
    "colsample_bynode" : hp.uniform('colsample_bynode',0.5,1),
    "reg_alpha":hp.uniform('reg_alpha',0.001,10), 
    "reg_lambda":hp.uniform('reg_lambda',0.01,50)
}
def xgb_hamsu(search_space) : 
    params = {
        'n_estimators' : int(round(search_space['n_estimators'])),
        'max_depth' : int(round(search_space['max_depth'])),
        "learning_rate" : search_space['learning_rate'], 
        'gamma' : int(round(search_space['gamma'])),
        "min_child_weight" : search_space['min_child_weight'], 
        "subsample" : search_space['subsample'], 
        "colsample_bytree" : search_space['colsample_bytree'], 
        "colsample_bylevel" : search_space['colsample_bylevel'], 
        "colsample_bynode" : search_space['colsample_bynode'], 
        "reg_alpha" : search_space['reg_alpha'], 
        "reg_lambda" : search_space['reg_lambda'], 
        
    }
    model = XGBClassifier(
        **params,
    )
    
    # 3. 훈련
    model.fit(x_train,y_train)

    # 4. 평가, 예측 
    y_pred = model.predict(x_test)
    result_mse = mean_squared_error(y_test,y_pred)
    return result_mse

start = time.time()
trials_val = Trials()
best = fmin(
    fn=xgb_hamsu,
    space=search_space,
    algo= tpe.suggest,
    max_evals = 50,
    trials = trials_val,
    rstate=np.random.default_rng(seed = 10)
)
print('best : ',best)

'''



print('모델 시작')
start = time.time()
# model = GridSearchCV(
# model= BaggingClassifier(
model =   XGBClassifier(
            colsample_bylevel= 0.5903167248154069,
            colsample_bynode= 0.6162948448917287,
            colsample_bytree= 0.6637033614330781, 
            gamma= 3, 
            learning_rate= 0.4999718214812156,
            max_depth= 3,
            min_child_weight= 177.95618176346483, 
            n_estimators= 702, 
            reg_alpha= 4.489071402553407, 
            reg_lambda= 31.90187287150227, 
            subsample= 0.767545370379666,
)
#       n_estimators=10,
#       random_state=8715,
#       bootstrap= False,
# )

# 배깅 :  0.2564463677
# 3. 훈련
print('훈련')
num_boost_round = int(10.0)
model.fit(
    x_train,y_train,
    eval_set=[(x_train, y_train),(x_test, y_test)],
    eval_metric="mlogloss",
    early_stopping_rounds=100,
    verbose=10, 
          )

# 4. 평가, 예측 
print('훈련')
result = model.score(x_test,y_test)
print(" result: ", result)
y_pred= model.predict(x_test)  
# y_pred=np.argmax(y_pred,axis=1)
print(y_pred)
acc = accuracy_score(y_test,y_pred)
print('acc :',acc)
f1 = f1_score(y_test,y_pred,average='micro')
print('f1 :',f1)

end = time.time()
print('걸린시간:',round(end-start,2))

# 5. 제출
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
print(submission.shape)
y_submit = model.predict(test_csv)
print(y_submit.shape)
submission['TARGET'] =y_submit
submission.to_csv(path_save+'00002_last.csv')

'''
0.5300490938

'''