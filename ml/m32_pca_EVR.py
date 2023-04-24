import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target
# print(x.shape, y.shape) #(569, 30) (569,)

pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x.shape)                #(569, 30)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) #0.9999999999999998


pca_cumsum = np.cumsum(pca_EVR)
print(pca_cumsum)

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()


