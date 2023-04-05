import tensorflow as tf

x_train=1
y_train=2
# 모델 구성
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])

# 손실 함수 지정
loss = tf.keras.losses.RankNetLoss()

# 모델 컴파일
model.compile(optimizer='adam', loss=loss)

# 모델 학습
model.fit(x_train, y_train, epochs=10, batch_size=32)