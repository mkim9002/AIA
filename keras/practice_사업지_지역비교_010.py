import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,  Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional
from sklearn.metrics import r2_score, mean_squared_error


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


X = merged_data[['검색건수', '방문자수', '관광지출액']] # feature 데이터
y = merged_data['영업상태구분코드'] # target 데이터

# train set과 test set 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1.2.5 scaler (0,1로 분리)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#timesteps
def split_x(data, size):
    x_list = []
    for i in range(len(data) - size + 1):
        x = data[i:i+size, :]
        x_list.append(x)
    return np.array(x_list)

timesteps = 3

# X_train 데이터 준비
X_train_scaled = scaler.fit_transform(X_train)
X_train_timestep = split_x(X_train_scaled, timesteps)
X_train_timestep = X_train_timestep[:-timesteps+1] # 마지막 timesteps개 데이터 버림
print(X_train_timestep.shape) #(77, 3, 3)

# X_test 데이터 준비
X_test_scaled = scaler.transform(X_test)
X_test_timestep = split_x(X_test_scaled, timesteps)
X_test_timestep = X_test_timestep[:-timesteps+1] # 마지막 timesteps개 데이터 버림
print(X_test_timestep.shape) #(17, 3, 3)

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(timesteps, 3))
conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='ss0')(input1)
lstm1 = LSTM(90, activation='relu', name='ss1')(conv1)
dense1 = Dense(80, activation='relu', name='ss2')(lstm1)
dense2 = Dense(70, activation='relu', name='ss3')(dense1)
output1 = Dense(50, activation='relu', name='ss4')(dense2)

# 2.2 모델2
input2 = Input(shape=(timesteps, 14))
dense11 = LSTM(90, name='hd1')(input2)
dense12 = Dense(80, name='hd2')(dense11)
dense13 = Dense(70, name='hd3')(dense12)
dense14 = Dense(60, name='hd4')(dense13)
output2 = Dense(50, name='output2')(dense14)

# 2.3 머지
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(50, activation='relu', name='mg2')(merge1)
merge3 = Dense(50, activation='relu', name='mg3')(merge2)
hidden_output = Dense(50, name='last')(merge3)

# 2.5 분기1
bungi1 = Dense(30, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

# 2.6 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)
model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

#3. 컴파일 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#
