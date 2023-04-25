#선형 판별 분석(Linear Discriminant Analysis, LDA)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine, fetch_covtype


# 1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
x, y = load_digits(return_X_y=True)

# pca = PCA(n_components=3)
# x = pca.fit_transform(x)

# print(x.shape) #(150, 3)

lda = LinearDiscriminantAnalysis(n_components=3)
#n_components는 클래스의 갯수 빼가 하나 이하로 가능!!
x = lda.fit_transform(x,y)
print(x.shape)

 
 
 
 
 
 
 
 