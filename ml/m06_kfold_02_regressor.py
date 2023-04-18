from sklearn.utils import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings(action='ignore')

path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes, ddarung_train, kaggle_train]

algorithms_classifier = all_estimators(type_filter='classifier')
algorithms_regressor = all_estimators(type_filter='regressor')

max_score=0
max_name=''

scaler_list = [RobustScaler(), StandardScaler(), MinMaxScaler(), MaxAbsScaler()]

n_split = 10
kf = KFold(n_splits=n_split, shuffle=True, random_state=123)

for i in range(len(data_list)):
    if i<4:
        x, y = data_list[i](return_X_y=True)
        for j in scaler_list:
            scaler = j
            x = scaler.fit_transform(x)
            for name, algorithm in algorithms_classifier:
                try:
                    model = algorithm()
                    results = cross_val_score(model, x, y, cv=kf)
                    if max_score<round(np.mean(results), 5):
                        max_score=round(np.mean(results), 5)
                        max_name=name
                    # print(type(j).__name__, data_list[i].__name__, name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, data_list[i].__name__, name, 'set default value first')
                    continue
            print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_score :', max_name, max_score)
    elif 4<=i<6:
        x, y = data_list[i](return_X_y=True)
        for j in scaler_list:
            scaler = j
            x = scaler.fit_transform(x)
            for name, algorithm in algorithms_regressor:
                try:
                    model = algorithm()
                    if name=="GaussianProcessRegressor":
                        del model
                        model = GaussianProcessRegressor(alpha=1000)
                    results = cross_val_score(model, x, y, cv=kf)
                    if max_score<round(np.mean(results), 5):
                        max_score=round(np.mean(results), 5)
                        max_name=name
                    # print(type(j).__name__, data_list[i].__name__, name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, data_list[i].__name__, name, 'set default value first')
                    continue
            print('\n', type(j).__name__, ' - ', data_list[i].__name__, 'max_score :', max_name, max_score)
    elif i==6:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
        for j in scaler_list:
            scaler = j
            x = scaler.fit_transform(x)
            for name, algorithm in algorithms_regressor:
                try:
                    model = algorithm()
                    results = model.score(model, x, y, cv=kf)
                    if max_score<round(np.mean(results), 5):
                        max_score=round(np.mean(results), 5)
                        max_name=name
                    # print(type(j).__name__, 'ddarung', name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, 'ddarung', name, 'set deault value first')
                    continue
            print('\n', type(j).__name__, ' - ', 'ddarung max_score :', max_name,  max_score)
    else:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
        for j in data_list:
            scaler = j
            x = scaler.fit_transform
            for name, algorithm in algorithms_regressor:
                try:
                    model = algorithm()
                    if name=="QuantileRegressor":
                        del model
                        model = QuantileRegressor(alpha=1000)
                    results = cross_val_score(model, x, y, cv=kf)
                    if max_score<round(np.mean(results), 5):
                        max_score=round(np.mean(results), 5)
                        max_name=name
                    # print(type(j).__name__, 'kaggle', name, 'acc :', results, 'mean of cross_val_score : ', round(np.mean(results), 5))
                except:
                    # print(type(j).__name__, 'kaggle', name, 'set deault value first')
                    continue
            print('\n', type(j).__name__, ' - ', 'kaggle max_score :', max_name, max_score)