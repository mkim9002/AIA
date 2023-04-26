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
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5) # -5
    upper_bound = quartile_3 + (iqr * 1.5) # 19
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

# Example usage
aaa = pd.DataFrame({'col1': [-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50], 
                    'col2': [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 429, 350]})
outliers_loc = outliers(aaa)
print("이상치:", outliers_loc)

