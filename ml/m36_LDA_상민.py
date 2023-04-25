#선형 판별 분석(Linear Discriminant Analysis, LDA)
#컬럼의 갯수가 클래스의 갯수보다 작을때
#회귀에서 가능?
#y 에 라운드 씌워 가능?
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, fetch_california_housing
from tensorflow.keras.datasets import cifar100



# 1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
x, y = load_diabetes(return_X_y=True)
# x, y = fetch_california_housing(return_X_y=True)
# y = np.round(y)
print(y)
print(len(np.unique(y, return_counts=True)))

# print(x.shape) #(150, 3)

# lda = LinearDiscriminantAnalysis(n_components=101)
lda = LinearDiscriminantAnalysis()

#n_components는 클래스의 갯수 빼가 하나 이하로 가능!!
x_lda = lda.fit_transform(x, y)
print(x_lda.shape)


 #회귀는 원래 안되지만 하지만 deibetes는 정수형 이라서 LDA에서 y의 클래스로 잘못 인식이라 돌아감
 #회귀데이터는 원칙적으로 에러인데 위처럼 돌려도 되지만 성능은 보장못함
 

 
 
 
 
 
 
 