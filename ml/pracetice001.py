import numpy as np
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
x,y = load_iris(return_X_y=True)

print(x.shape,y.shape)
