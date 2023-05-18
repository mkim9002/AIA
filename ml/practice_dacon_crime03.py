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

