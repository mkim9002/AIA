import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, make_scorer, accuracy_score


# 훈련 데이터 및 테스트 데이터 로드
path = 'd:/study_data/_data/gas/'
save_path = 'd:/study_data/_save/gas/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')


# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
def type_to_HP(type):
    HP=[60,60,60,60,60,60,60,60]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])


# Select subset of features for Autoencoder model
features = ['air_inflow', 'air_end_temp', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Split data into train and validation sets
x_train, x_val = train_test_split(data[features], train_size=0.9, random_state=555)

# Normalize data
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# Define Autoencoder model
input_layer = Input(shape=(len(features),))
encoder = Dense(4, activation='relu')(input_layer)
decoder = Dense(len(features), activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile Autoencoder model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train Autoencoder model
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=60)
autoencoder.fit(x_train, x_train, epochs=3000, batch_size=20, validation_data=(x_val, x_val), callbacks=[es])

# Predict anomalies in test data
test_data = scaler.transform(test_data[features])
predictions = autoencoder.predict(test_data)
mse = ((test_data - predictions) ** 2).mean(axis=1)

# Use LOF to detect anomalies
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
binary_predictions = [1 if x == -1 else 0 for x in clf.fit_predict(test_data)]

# Combine predictions from Autoencoder and LOF
final_predictions = [1 if x+y >= 1 else 0 for x, y in zip(binary_predictions, mse > (mse.mean() + mse.std() * 2))]

submission['label'] = pd.DataFrame({'Prediction': final_predictions})

# Save submission file
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submit12.csv', index=False)