import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten

# 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
df2 = pd.read_csv(path + '전국관광지데이터.csv', encoding='CP949')

# 결측치 처리
df1 = df1.dropna(subset=['폐업일자', '소재지전화'], how='any')

# 데이터프레임 합치기
merged_data = pd.merge(df1, df2, how='inner', on='지역명')

# ',' 제거하고 숫자형으로 변환
merged_data[['방문자수', '검색건수', '관광지출액']] = merged_data[['방문자수', '검색건수', '관광지출액']].apply(lambda x: x.str.replace(',', '')).astype(float)

# '지역명' 열에서 첫번째 단어만 추출하여 새로운 '지역' 열을 생성
merged_data['지역'] = merged_data['지역명'].str.split().str.get(0)

# '지역'과 '영업상태명' 열만 추출
subset = merged_data[['지역', '방문자수', '검색건수', '관광지출액']]

print(subset)

# '지역'과 '영업상태명' 열의 조합별로 개수를 세어 데이터프레임으로 저장
counts = subset.groupby(['지역']).agg({'방문자수': np.sum, '검색건수': np.sum, '관광지출액': np.sum})

# 비율을 계산
percentages = counts.apply(lambda x: x/x.sum(), axis=1)

# 결과 출력
print(percentages)

# 전국관광지데이터에서 '방문자수', '검색건수', '관광지출액'을 지역별로 계산
subset2 = df2[['지역명', '방문자수', '검색건수', '관광지출액']]
subset2['지역'] = subset2['지역명'].str.split().str.get(0)
counts2 = subset2.groupby(['지역']).agg({'방문자수': np.sum, '검색건수': np.sum, '관광지출액': np.sum})
percentages2 = counts2.div(counts2.sum(axis=1), axis=0)

# 결과 출력
print(percentages2)


####################################################################################
# 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
df2 = pd.read_csv(path + '전국관광지데이터.csv', encoding='CP949')

# 방문자수, 검색건수, 관광지출액 데이터 추출
df2 = df2[['지역명', '방문자수', '검색건수', '관광지출액']]

# 지역명 첫번째 단어 추출
df2['지역명'] = df2['지역명'].str.split().str[0]

# 영업상태명을 1에서 0으로 변환하는 함수
def status_to_score(status):
    if status == '영업':
        return 1
    elif status == '휴업':
        return 0.5
    else:
        return 0

# 영업상태명 열의 값을 점수로 변환
df1['score'] = df1['영업상태명'].apply(status_to_score)

# 평균값 대신 중앙값을 사용해 계산하는 함수
def calc_median(df):
    return np.median(df)

# groupby를 사용해 지역별로 데이터를 그룹화하고, 방문자수, 검색건수, 관광지출액에 대해 중앙값 계산
df2 = df2.groupby('지역명').agg({'방문자수': calc_median, '검색건수': calc_median, '관광지출액': calc_median})

# groupby를 사용해 지역별로 데이터를 그룹화하고, score에 대해 평균값 계산
df2_grouped = df2.groupby('지역명').mean().reset_index()
df2_grouped.rename(columns={'score': '평균_score'}, inplace=True)

# 데이터 병합
df = pd.concat([df1, df2], axis=1)

# 지역별로 각 변수마다의 점수 계산
df['방문자수_score'] = (df['방문자수'].rank(method='dense', ascending=False) / len(df)).apply(lambda x: round(1 - x, 2))
df['검색건수_score'] = (df['검색건수'].rank(method='dense', ascending=False) / len(df)).apply(lambda x: round(1 - x, 2))
df['관광지출액_score'] = (df['관광지출액'].rank(method='dense', ascending=False) / len(df)).apply(lambda x: round(1 - x, 2))
df['score'] = df['score'].apply(lambda x: round(x, 2))

# 각 변수마다의 점수를 합산하여 최종 점수 계산
df['total_score'] = df['방문자수_score'] + df['검색건수_score'] + df['관광지출액_score'] + df['score']

# 점수 기준으로 정렬
df = df.sort_values(by='total_score', ascending=False)

# 결과 출력
print(df)

#############################################################
# 데이터 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[['방문자수', '검색건수', '관광지출액']])

# 입력 데이터와 타깃 데이터로 분리
X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# train set과 test set으로 데이터 분리
split_index = int(len(X) * 0.8)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
