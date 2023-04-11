import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
df2 = pd.read_csv(path + '전국관광지데이터.csv', encoding='CP949')

# 2-1. 데이터 전처리
# 지역명에서 첫번째 단어 추출
df1['지역명'] = df1['지역명'].apply(lambda x: x.split()[0])
status_mapping = {'영업': 1, '휴업': 0.5, '폐업': 0, '취소': 0}
df1['영업상태구분코드'] = df1['영업상태구분코드'].map(status_mapping).fillna(df1['영업상태구분코드'])

# 3. 각 지역명 영업 중인 비율 계산
region_counts = df1['지역명'].value_counts()
region_operating_counts = df1[df1['영업상태구분코드'] == 1]['지역명'].value_counts()
region_operating_ratios = region_operating_counts / region_counts * 100
print(region_operating_ratios)

# 4. 각 지역명별 검색건수, 방문자수, 관광지출액을 min-max scaler로 0과 1 사이로 변환하여 합산 점수 계산
# 각 지역명별 검색건수 합산 점수 계산
scaler = MinMaxScaler()
search_sum = df2.groupby('지역명')['검색건수'].sum().reset_index(name='검색건수_합계')
search_sum['검색건수_합계_정규화'] = scaler.fit_transform(search_sum[['검색건수_합계']])

# 각 지역명별 방문자수 합산 점수 계산
visitor_sum = df2.groupby('지역명')['방문자수'].sum().reset_index(name='방문자수_합계')
visitor_sum['방문자수_합계_정규화'] = scaler.fit_transform(visitor_sum[['방문자수_합계']])

# 각 지역명별 관광지출액 합산 점수 계산
expense_sum = df2.groupby('지역명')['관광지출액'].sum().reset_index(name='관광지출액_합계')
expense_sum['관광지출액_합계_정규화'] = scaler.fit_transform(expense_sum[['관광지출액_합계']])

# 합산 점수 계산
region_scores = pd.merge(search_sum, visitor_sum, on='지역명')
region_scores = pd.merge(region_scores, expense_sum, on='지역명')
region_scores['합산점수'] = region_scores['검색건수_합계_정규화'] + region_scores['방문자수_합계_정규화'] + region_scores['관광지출액_합계_정규화']
region_scores = region_scores[['지역명', '합산점수']]
print(region_scores)

# 결측치 처리
df1 = df1.dropna(subset=['폐업일자', '소재지전화','상세영업상태명'], how='any')

# 데이터프레임 합치기
merged_data = pd.merge(df1, df2, how='inner', on='지역명')

# 3. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 4. 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습
model.fit(X, y, epochs=50, batch_size=64, verbose=1)

# 6. 내년 각 지역별 도시별 영업율, 폐업율, 휴업율, 취소율 예측

# 예측할 데이터 불러오기
df_predict = pd.read_csv(path + 'predict_data.csv', encoding='CP949')

# 지역명에서 첫번째 단어 추출
df_predict['지역명'] = df_predict['지역명'].apply(lambda x: x.split()[0])

# 스케일링
scaled_data_predict = scaler.transform(df_predict[['검색건수', '방문자수', '관광지출액']].values)
scaled_data_predict[np.isnan(scaled_data_predict)] = 0

# 시계열 데이터로 변환
X_predict = []
for i in range(time_steps, len(scaled_data_predict)):
    X_predict.append(scaled_data_predict[i-time_steps:i])
X_predict = np.array(X_predict)

# 예측
y_predict = model.predict(X_predict)

print(y_predict)
