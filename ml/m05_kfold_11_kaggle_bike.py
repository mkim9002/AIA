import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 데이터
path = './_data/kaggle_bike/'
data = pd.read_csv(path + 'train.csv', index_col=0).dropna()
x = data.drop(['casual', 'registered', 'count'], axis=1)
y = data['count']
x=x[:100]
y[:100]
n_split = 5
kf = KFold(n_splits=n_split, shuffle=True, random_state=123)
# kf = KFold()

# 2. 모델
model = RandomForestClassifier()

# 3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kf, n_jobs=-1)
print(scores)

print('ACC : ', scores, '\n mean of cross_val_score : ', round(np.mean(scores), 4))