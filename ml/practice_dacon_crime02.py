import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import classification_report
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# 1. 데이터
path = 'D:/study/_data/crime/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['TARGET'], axis=1)

y = train_csv['TARGET']
y = pd.cut(y, bins=[-1, 2, 4, 6, 8, 10], labels=[0, 1, 2, 3, 4]) # 카테고리화

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=5555
)

categorical_cols = ['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)', '적설량(cm)', '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림', '범죄발생지']
x_train[categorical_cols] = x_train[categorical_cols].astype('category')
x_test[categorical_cols] = x_test[categorical_cols].astype('category')
test_csv[categorical_cols] = test_csv[categorical_cols].astype('category')

model = LGBMClassifier()

f1score_callback = Callback()
f1score_callback.on_epoch_end = lambda epoch, logs: print("Validation F1-score: {:.4f}".format(f1_score(y_test, model.predict(x_test), average='micro')))

early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, eval_set=[(x_test, y_test)], callbacks=[f1score_callback, early_stop_callback])

y_predict = model.predict(x_test)
report = classification_report(y_test, y_predict) # classification_report로 평가 지표 계산
print(report)

# time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")


# Submission
save_path = 'D:/study/save/crime/'
y_sub = model.predict(test_csv)
y_sub = pd.cut(y_sub, bins=[-1, 2, 4, 6, 8, 10], labels=[0, 1, 2, 3, 4]) # 카테고리화
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]] = y_sub
sample_submission_csv.to_csv(save_path + 'book' + date + '.csv', index=False, float_format='%.0f')

