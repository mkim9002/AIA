import numpy as np
import pandas as pd

a =[[1,2,3],[4,5,6]]

b = np.array(a)
print(b)

c = [[1,2,3],[4,5]]
print(c)  #list 는 행렬이 다라도 출력됩니다
d = np.array(c)
print(d)  #[list([1, 2, 3]) list([4, 5])]



###############################
e = [[1,2,3,],["바보","맹구",5,6]]
print(e)

# 2. 리스트에는 다른 자료형을 넣어도 상관없다.
f = np.array(e)
print(f)

# print(e.shape)   #'list' object has no attribute 'shape'



