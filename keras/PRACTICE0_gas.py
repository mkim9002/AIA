import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

# 훈련 데이터 및 테스트 데이터 로드
path = 'd:/study_data/_data/gas/'
save_path = 'd:/study_data/_save/gas/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)

train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Select subset of features for LOF model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size=0.9999999999, random_state=1)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Tune Isolation Forest model
if_model = IsolationForest(random_state=1)
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.1, 0.5, 1.0],
    'contamination': ['auto', 0.01, 0.05, 0.1]
}
if_search = RandomizedSearchCV(if_model, param_distributions, n_iter=10, random_state=1)
if_search.fit(X_train)

# Print best hyperparameters
print(if_search.best_params_)

# Predict anomalies in test data using tuned Isolation Forest
test_data_if = scaler.transform(test_data[features])
y_pred_test_if = if_search.predict(test_data_if)
if_predictions = [1 if x == -1 else 0 for x in y_pred_test_if]

submission['label'] = pd.DataFrame({'Prediction': if_predictions})
print(submission.value_counts())

# Save submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)