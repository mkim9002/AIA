#선형 판별 분석(Linear Discriminant Analysis, LDA)
#컬럼의 갯수가 클래스의 갯수보다 작을때
#
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine, fetch_covtype
from tensorflow.keras.datasets import cifar100

# Load datasets
datasets = [load_iris, load_breast_cancer, load_digits, load_diabetes, load_wine, fetch_covtype]

for dataset in datasets:
    x, y = dataset(return_X_y=True)
    
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x, y)
    print(dataset.__name__)
    print(x_lda.shape)

    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(cumsum)
    print('-' * 30)

# load_iris
# (150, 2)
# [0.9912126 1.       ]
# ------------------------------
# load_breast_cancer
# (569, 1)
# [1.]
# ------------------------------
# load_digits
# (1797, 9)
# [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
#  0.94984789 0.9791736  1.        ]
# ------------------------------
# load_diabetes
# (442, 10)
# [0.2520478  0.36528222 0.47649866 0.57413357 0.66752075 0.7556825
#  0.82963874 0.8962967  0.9501501  1.        ]
# ------------------------------
# load_wine
# (178, 2)
# [0.68747889 1.        ]
# ------------------------------
# fetch_covtype
# (581012, 6)
# [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]





