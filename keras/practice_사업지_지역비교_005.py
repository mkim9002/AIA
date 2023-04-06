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
# print(df1)
# print(df2)

# 결측치 처리
df1 = df1.dropna(subset=['폐업일자', '소재지전화'], how='any')

# 데이터가 저장된 CSV 파일을 불러옵니다.
data = pd.read_csv(path +'관광펜션업.csv', encoding='cp949')

# '소재지전체주소' 열에서 첫번째 단어만 추출하여 새로운 '지역' 열을 생성합니다.
data['지역'] = data['소재지전체주소'].str.split().str.get(0)

# '지역'과 '영업상태명' 열만 추출합니다.
subset = data[['지역', '영업상태명']]

# '지역'과 '영업상태명' 열의 조합별로 개수를 세어 데이터프레임으로 저장합니다.
counts = pd.crosstab(subset['지역'], subset['영업상태명'])

# 비율을 계산합니다.
percentages = counts.apply(lambda x: x/x.sum(), axis=1)

# 결과를 출력합니다.
print(percentages)



# 데이터 정규화
scaler = MinMaxScaler()
df1[['관광지출액']] = scaler.fit_transform(df1[['관광지출액']])
df2[['검색량']] = scaler.fit_transform(df2[['검색량']])


# 데이터 합치기
df = pd.concat([df1['관광지출액'], df2['검색량']], axis=1)

# 학습 데이터 생성
train = df.iloc[:-1,:]
train_label = df.iloc[1:,:]

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(4,1)))
model.add(Dense(4))

# 모델 컴파일 및 학습
model.compile(loss='mse', optimizer='adam')
model.fit(train.values.reshape(-1,4,1), train_label.values, epochs=100, batch_size=1, verbose=2)

# 마지막 데이터로 추론하여 최적의 투자 지역 도출
prediction = model.predict(np.array(df.iloc[-1,:]).reshape(1,4,1))
investment_area = df.index[np.argmax(prediction)]

print("최적의 투자 지역은 {} 입니다.".format(investment_area))