#선형 판별 분석(Linear Discriminant Analysis, LDA)
#컬럼의 갯수가 클래스의 갯수보다 작을때
#
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100

# 1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
x, y = load_digits(return_X_y=True)
(x_train, y_train),(x_test, y_test) = cifar100.load_data()
print(x_train.shape) #(1797, 3)

x_train = x_train.reshape(50000, 32*32*3)

pca = PCA(n_components=99)
x_train = pca.fit_transform(x_train)

# print(x.shape) #(150, 3)

# lda = LinearDiscriminantAnalysis(n_components=101)
lda = LinearDiscriminantAnalysis()

#n_components는 클래스의 갯수 빼가 하나 이하로 가능!!
x = lda.fit_transform(x_train,y_train)
print(x.shape)

 # 2. 모델
 
 
import matplotlib.pyplot as plt
#scatter plot 그리기
plt.scatter(x[:,0], x[:,1],c=y)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('iris')
plt.show()
 
 
 
 
 
 
 
 