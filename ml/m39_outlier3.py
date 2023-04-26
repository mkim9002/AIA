# import numpy as np
# import matplotlib.pyplot as plt

# aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
#                 [100,200,-30,400,500,600,-70000,800,900,1000,210,429,350]])
# aaa = np.transpose(aaa)
# print(aaa)
# print(aaa.shape) #(13, 2)
#[실습] outlier1 을 이용하여 이상치 찾기
# 해결책 1 : 컬럼을 for로 2 번 돌리기
# 해결책 2 : dataframe 통째로 함수로 받아들여서 return 하게 수정

import numpy as np
import pandas as pd

def outliers(data_out):
    Q1 = data_out.quantile(0.25)
    Q3 = data_out.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = np.where((data_out < lower_bound) | (data_out > upper_bound))
    outlier_values = data_out.iloc[outlier_indices]
    return outlier_values, outlier_indices

# Example usage
aaa = pd.DataFrame({'col1': [-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50], 
                    'col2': [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 429, 350]})
outlier_values, outlier_indices = outliers(aaa)
print("Outlier values:")
print(outlier_values)
print("Outlier indices:")
print(outlier_indices)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()


