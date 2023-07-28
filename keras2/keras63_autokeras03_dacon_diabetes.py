import autokeras as ak
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
##
# Load and preprocess the data
train_csv = pd.read_csv('d:/study/_data/dacon_diabetes/train.csv', index_col=0)
train_csv = train_csv.dropna()
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=777)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model
model = ak.StructuredDataClassifier(overwrite=False, max_trials=2)
model.fit(x_train, y_train, epochs=10, validation_split=0.15)

# Export and save the best model
best_model = model.export_model()
best_model.save('path_to_save_model', save_format='tf')

# Evaluate the model
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('Model results:', results)
