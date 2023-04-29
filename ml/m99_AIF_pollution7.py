import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load the data
path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'

train_data = pd.read_csv(path + 'train_all.csv')
test_data = pd.read_csv(path + 'test_all.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Combine train and test data for preprocessing
all_data = pd.concat([train_data, test_data], axis=0)

# Perform data preprocessing
# Encode categorical columns
categorical_cols = ['연도', '일시', '측정소']
label_encoder = LabelEncoder()
for col in categorical_cols:
    all_data[col] = label_encoder.fit_transform(all_data[col])

# Split the data back into train and test
train_data = all_data[:train_data.shape[0]]
test_data = all_data[train_data.shape[0]:]

# Remove rows with missing values in target variable (PM2.5)
train_data = train_data.dropna(subset=['PM2.5'])

# Split the data into features and target
X_train = train_data.drop(['PM2.5'], axis=1).values.astype(float)
y_train = train_data['PM2.5'].values.astype(float)

X_test = test_data.drop(['PM2.5'], axis=1).values.astype(float)

# Scale the data
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model architecture
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
hidden_layer = Dense(190, activation='relu')(input_layer)
output_layer = Dense(1)(hidden_layer)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=250, batch_size=128, validation_split=0.3, callbacks=[EarlyStopping(patience=5)])

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Update the submission dataframe with the predicted values
submission = submission.reindex(range(len(y_pred)))
submission['PM2.5'] = y_pred

# Save the results
submission.to_csv(save_path + 'submit20.csv', index=False)
print(f'Results saved to {save_path}submit.csv')
