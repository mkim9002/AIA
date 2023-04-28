#MICE(Multiple Imputation by chained Equation)
import numpy as np
import pandas as pd
import sklearn as sk
from impyute.imputation.cs import mice

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2,4,np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# import_df = mice(data)   #마이스는 numpy로 해야해

# import_df = mice(data.values)   #.values 는 numpy로 바꿀때
import_df = mice(data.to_numpy())   #to_numpy() 로도 가능
print(import_df)