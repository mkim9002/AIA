import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

# 순위 데이터
rankings = [
    (1, 2), # 부산이 서울보다 높은 순위
    (1, 3), # 경기가 서울보다 높은 순위
    (4, 5), # 대전이 대구보다 높은 순위
    (2, 4), # 대구가 부산보다 높은 순위
    (3, 5), # 대전이 경기보다 높은 순위
]

# 지역 데이터
locations = ['서울', '부산', '경기', '대구', '대전']

# 지역을 one-hot encoding으로 변환
locations_onehot = np.eye(len(locations))

# RankNet 모델 구성
input_1 = Input(shape=(len(locations),))
input_2 = Input(shape=(len(locations),))

dense_1 = Dense(16, activation='relu')(input_1)
dense_2 = Dense(16, activation='relu')(input_2)

merged = Dense(1, activation='sigmoid')(dense_1 - dense_2)

model = Model(inputs=[input_1, input_2], outputs=merged)
model.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy)

# 학습
for i in range(1000):
    np.random.shuffle(rankings)
    x1, x2, y = [], [], []
    for r in rankings:
        x1.append(locations_onehot[r[0]-1])
        x2.append(locations_onehot[r[1]-1])
        y.append(1.0)
        x1.append(locations_onehot[r[1]-1])
        x2.append(locations_onehot[r[0]-1])
        y.append(0.0)
    model.train_on_batch([np.array(x1), np.array(x2)], np.array(y))

# 예측
for i in range(len(locations)):
    for j in range(i+1, len(locations)):
        p = model.predict([locations_onehot[i][np.newaxis,:], locations_onehot[j][np.newaxis,:]])[0][0]
        if p > 0.5:
            print(locations[i], ">", locations[j])
        else:
            print(locations[j], ">", locations[i])
            
            from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import RankNetLoss


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