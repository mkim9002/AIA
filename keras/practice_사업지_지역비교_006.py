import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 불러오기
path = './_data/개인프로젝트/관광 숙박 사업/'
path_save = './_save/개인프로젝트/'

pension_data = pd.read_csv(path + '관광펜션업.csv', encoding='CP949')
tourist_data = pd.read_csv(path +'전국관광지데이터.csv', encoding='CP949')

# 결측치 처리
pension_data = pension_data.dropna(subset=['폐업일자', '소재지전화'], how='any')

# '소재지전체주소' 열에서 시도명만 추출하여 새로운 '시도명' 열을 생성합니다.
pension_data['시도명'] = pension_data['소재지전체주소'].str.split().str.get(0)

# '시도명'과 '영업상태명' 열만 추출합니다.
pension_subset = pension_data[['시도명', '영업상태명']]

# '시도명'과 '영업상태명' 열의 조합별로 개수를 세어 데이터프레임으로 저장합니다.
pension_counts = pd.crosstab(pension_subset['시도명'], pension_subset['영업상태명'])

# 비율을 계산합니다.
pension_percentages = pension_counts.apply(lambda x: x/x.sum(), axis=1)

# '시도명'을 기준으로 두 데이터를 병합합니다.
merged_data = pd.merge(pension_data, tourist_data, how='inner', on='시도명')

# 결측치 처리
merged_data = merged_data.dropna(subset=['방문자수', '검색건수'], how='any')


# 데이터 정규화
scaler = MinMaxScaler()
merged_data[['방문자수', '검색건수', '관광지출액']] = merged_data[['방문자수', '검색건수', '관광지출액']].apply(lambda x: x.replace(',', '')).astype(float)
scaled_data = scaler.fit_transform(merged_data[['방문자수', '검색건수', '관광지출액']])

# 입력과 출력을 나눕니다.
x_train = []
y_train = []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i])
    y_train.append(scaled_data[i, 2])
    
x_train, y_train = np.array(x_train), np.array(y_train)

# 모델 구성 (LSTM)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 3)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, batch_size=16)


# 모델을 이용해 최적의 투자지역 도출
X_test = np.array([X[-1]])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_value = model.predict(X_test)
predicted_value = scaler.inverse_transform(predicted_value)

# 투자 지역 추출
result = []
for i in range(len(predicted_value)):
    max_value = max(predicted_value[i])
    max_index = np.where(predicted_value[i] == max_value)
    result.append(max_index[0][0])
    
invest_region = list(le.inverse_transform(result))
print('최적의 투자지역:', invest_region)