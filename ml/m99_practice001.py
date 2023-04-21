from sklearn.utils import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings(action='ignore')

path_ddarung = './_data/ddarung/'
path_kaggle = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv' index_col=0).dropn()

#######################




