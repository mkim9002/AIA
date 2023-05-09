import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# 데이터 불러오기
data_dir = 'c:/study/_data/baseball'
save_dir = 'c:/study/_save/baseball'
data_file = 'baseball_data.csv'
data_path = os.path.join(data_dir, data_file)
data = pd.read_csv(data_path, encoding='cp949')

# 필요한 열만 선택
data = data[["team", "win", "score", "stadium", "date"]]

# 누락된 값 처리
data = data.dropna()

# # 범주형 데이터 수치화
# encoder = LabelEncoder()
# data["team"] = encoder.fit_transform(data["team"])
# data["stadium"] = encoder.fit_transform(data["stadium"])

# 범주형 데이터 수치화
encoder = LabelEncoder()
encoder.fit(data["team"])
encoder.fit(data["stadium"])  # add this line to fit the stadium column as well
data["team"] = encoder.transform(data["team"])
data["stadium"] = encoder.transform(data["stadium"])

# 학습 데이터와 테스트 데이터로 분할
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 입력 데이터와 타겟 데이터 분리
train_x = train_data.drop("win", axis=1)
train_y = train_data["win"]
test_x = test_data.drop("win", axis=1)
test_y = test_data["win"]

# 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=4, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 학습
model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=1)

# 모델 평가
test_loss, test_acc = model.evaluate(test_x, test_y)
print("Test accuracy:", test_acc)

# 모델 예측
new_data = pd.DataFrame({"team": ["LG", "SK"], "score": [4, 2], "stadium": ["1", "2"]})
new_data["team"] = encoder.transform(new_data["team"])
new_data["stadium"] = encoder.transform(new_data["stadium"])

# transform() method for new data
new_data["team"] = encoder.transform(new_data["team"])
new_data["stadium"] = encoder.transform(new_data["stadium"])



# 모델 예측
prediction = model.predict(new_data)
print(prediction)


