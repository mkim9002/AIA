from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 데이터 로드 및 전처리
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=543432)

# 모델 구성
model = Sequential([
    Dense(128, input_dim=8, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(1)
])

# 모델 컴파일
optimizer = RMSprop(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

# 모델 훈련
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2)

# 모델 평가
loss, mae = model.evaluate(x_test, y_test)
print('loss:', loss)
print('mae:', mae)

# 예측 및 R2 스코어 계산
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('R2 score:', r2)


