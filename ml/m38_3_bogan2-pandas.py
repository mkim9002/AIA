import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2,4,np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0. 결측지 확인
print(data.isnull())           #True 가 결축지
print(data.isnull().sum())
print(data.info())

#1, 결측지 삭제
print('===============결측지 삭제================')
print(data['x1'].dropna())
print(data.dropna())       #디폴트 행위주로 삭제
print('===============결측지 삭제================')
print(data.dropna(axis=0)) #위와 같다 디폴트 행위주로 삭제
print('===============결측지 삭제================')
print(data.dropna(axis=1)) #디폴트 열위주로 삭제


#2-1 특정값.평균
print('===============결측지 처리 mean()================')
means = data.mean()
print('평균 :', means)
data2 = data.fillna(means)      #평균값
print(data2)

#2-2 특정값.중위값
print('===============결측지 처리 median()================')
median = data.median()
print('중위값 :', median)
data3 = data.fillna(median)
print(data3)

 #2-3 특정값.ffill, bfill
print('===============결측지 처리 ffill, bfill================')
data4 = data.fillna(method='ffill')
print(data4)

data5 = data.fillna(method='bfill')
print(data4)

#2-4 특정값. 임의값으로 채우기
data6 = data.fillna(value=7777777)
print(data6)
 
 
 
