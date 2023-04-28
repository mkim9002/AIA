import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from keras import regularizers

# Load the data
path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'

train_data = pd.read_csv(path + 'train_all.csv')
test_data = pd.read_csv(path + 'test_all.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Perform any necessary data cleaning or manipulation here
# ...

# Split the data into features and target
X_train = train_data.drop(['연도', '일시', '측정소', 'PM2.5'], axis=1)
y_train = train_data['PM2.5']

X_test = test_data.drop(['연도', '일시', '측정소', 'PM2.5'], axis=1)
y_test = test_data['PM2.5']

# # Scale the data
# scaler = MaxAbsScaler()  # Choose the appropriate scaler
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Define the model architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
hidden_layer = Dense(64, activation='relu')(input_layer)  # Adjust the number of units in the hidden layer as needed
output_layer = Dense(1)(hidden_layer)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])

# Predict on the test set
y_pred = model.predict(X_test)

# Update the submission dataframe with the predicted values
submission['PM2.5'] = y_pred

# Save the results
submission.to_csv(save_path + 'submit.csv', index=False)
print(f'Results saved to {save_path}submit.csv')
