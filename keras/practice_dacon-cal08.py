import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessRegressor
from math import sqrt
import datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd

# Load data
path='c:/study/_data/dacon_cal/'
save_path= 'c:/study/_save/dacon_cal/'
submission = pd.read_csv(path+'sample_submission.csv')

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)

#1.4 x,y 분리
x = train_csv.drop(['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)', 'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender', 'Age'], axis=1)
y = train_csv['Calories_Burned']

print(x.shape) #(7500, 1)


#1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=34553
    )     


#2. 모델 구성

model = DecisionTreeRegressor()
model.fit(x_train, y_train)

# Predict on training data and calculate RMSE
y_pred_train = model.predict(x_train)
rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
print('RMSE on training data:', rmse_train)


# Save submission file with timestamp
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)

#RMSE on training data: 0.0  model = DecisionTreeRegressor() #0421_1859submission.csv


