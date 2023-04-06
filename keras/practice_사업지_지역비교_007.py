import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

df1 = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
df2 = pd.read_csv(path +'전국관광지데이터.csv', encoding='CP949')

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

# '지역'과 '영업상태명' 열의 조합별로 개수를 세어 데이터프레임으로 저장
counts = subset.groupby(['지역']).agg({'방문자수': 'sum', '검색건수': 'sum', '관광지출액': 'sum'})

# 비율을 계산
percentages = counts.apply(lambda x: x/x.sum(), axis=1)

# 결과 출력
print(percentages)

# 전국관광지데이터에서 '방문자수', '검색건수', '관광지출액'을 지역별로 계산
subset2 = df2[['지역명', '방문자수', '검색건수', '관광지출액']]
subset2['지역'] = subset2['지역명'].str.split().str.get(0)
counts2 = subset2.groupby(['지역']).agg({'방문자수': 'sum', '검색건수': 'sum', '관광지출액': 'sum'})
percentages2 = counts2.apply(lambda x: x/x.sum(), axis=1)

# 결과 출력
print(percentages2)