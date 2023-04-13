import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, concatenate, Dropout, Bidirectional
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
df2 = pd.read_csv(path + '전국관광지데이터.csv', encoding='CP949')
print(df1)

# 연도별 폐업 건수 계산
폐업추세 = df1.groupby('폐업일자').size().reset_index(name='건수')
print(폐업추세)

# 연도별 폐업 비율 계산
total = 폐업추세['건수'].sum()
a = 폐업추세['비율(%)'] = round(폐업추세['건수'] / total * 100, 2)
# print(폐업추세)

# 2. 전국 관광지 데이터 가공
# 2021-2022 관광지 출액, 검색건수, 방문자수 증가율 계산

# 전체 관광지 데이터에서 2021, 2022 데이터 추출
df2_21 = df2[['2021관광지출액', '2021검색건수', '2021방문자수']].dropna()
df2_22 = df2[['2022관광지출액', '2022검색건수', '2022방문자수']].dropna()

# 2021, 2022 관광지출액 합계 계산 및 증가율 계산
total_rev_21 = df2_21['2021관광지출액'].sum()
total_rev_22 = df2_22['2022관광지출액'].sum()
b = rev_increase = round((total_rev_22 - total_rev_21) / total_rev_21 * 100, 2)


# 2021, 2022 검색건수 합계 계산 및 증가율 계산
total_search_21 = df2_21['2021검색건수'].sum()
total_search_22 = df2_22['2022검색건수'].sum()
c = search_increase = round((total_search_22 - total_search_21) / total_search_21 * 100, 2)

# 2021, 2022 방문자수 합계 계산 및 증가율 계산
total_visit_21 = df2_21['2021방문자수'].sum()
total_visit_22 = df2_22['2022방문자수'].sum()
d = visit_increase = round((total_visit_22 - total_visit_21) / total_visit_21 * 100, 2)

# 결과 출력
print('2021-2022 관광지 출액 증가율: {}%'.format(b))
print('2021-2022 검색건수 증가율: {}%'.format(c))
print('2021-2022 방문자수 증가율: {}%'.format(d))

rev = ('2021-2022 관광지 출액 증가율: {}%'.format(b))
print(rev)
search = (('2021-2022 검색건수 증가율: {}%'.format(c)))
print(search)
visit = ('2021-2022 방문자수 증가율: {}%'.format(d))
print(visit)







