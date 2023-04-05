from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import 

# 데이터셋 생성
X_train = [[10, 5, 7, 3], [8, 9, 6, 4], [3, 5, 8, 2], [2, 3, 4, 5], [9, 5, 4, 6]]
y_train = [1, 2, 4, 5, 3]

# 모델 구성
model = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

# RankNetLoss loss 함수와 Adam optimizer 사용
model.compile(loss=RankNetLoss(), optimizer=Adam(learning_rate=0.001))

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 새로운 데이터 예측
X_test = [[8, 3, 6, 2], [5, 8, 7, 6], [7, 9, 10, 8]]
y_pred = model.predict(X_test)