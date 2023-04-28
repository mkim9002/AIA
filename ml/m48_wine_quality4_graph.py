#그래프 그린다
#1. value_counts ->쓰지마
#2. np.unique의 return_counts 쓰지마
#3. grooupby 써 ,count() 써  pandas

#plt.bar 로 그린다 (quality 컬럼)

#힌트
#데이터개수 (y축) =데이터갯수, ......

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 전처리
path = 'c:/study/_data/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv['type'] = pd.get_dummies(train_csv['type'])['white']  # 원핫인코딩
test_csv['type'] = pd.get_dummies(test_csv['type'])['white']  # 원핫인코딩

# 2. Tukey's fences를 이용한 이상치 탐지
q1 = train_csv.quantile(0.25)
q3 = train_csv.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = ((train_csv < lower_fence) | (train_csv > upper_fence)).any(axis=1)

# 이상치 제거
train_csv = train_csv[~outliers]

# 3. 데이터 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality'] - 3  # Shift class labels to start from 0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=640874)


# 4. 모델 구성
model = RandomForestClassifier(n_estimators=1000, random_state=640874)

# 5. 훈련
model.fit(x_train, y_train)

# 6. 평가 및 예측
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print('Train Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)

# 7. 예측 및 제출 파일 생성
test_pred = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = test_pred + 3  # Shift class labels back to the original range
submission.to_csv(path + 'submit_wine_randomforest.csv')


# count_data = train_csv.groupby('quality')['quality'.count]()

# 8. 그래프 그리기
train_quality_counts = train_csv['quality'].value_counts().sort_index()
quality_labels = train_quality_counts.index
data_counts = train_quality_counts.values

plt.bar(quality_labels, data_counts)
plt.xlabel('Quality')
plt.ylabel('Data Count')
plt.title('Distribution of Wine Quality')
plt.show()
