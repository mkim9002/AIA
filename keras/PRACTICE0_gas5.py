import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

# Load data
path = 'd:/study_data/_data/gas/'
save_path = 'd:/study_data/_save/gas/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[29,21,9,51,29,31,29,31]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Select subset of features for LOF model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and validation data
X = train_data[features]
X_train, X_val = train_test_split(X, train_size= 0.89, random_state=1)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define parameter grid for GridSearchCV
param_grid = {'n_neighbors': [21, 25, 29, 33],
              'contamination': [0.05, 0.07, 0.09]}

# Define custom scoring function
def lof_auc_score(estimator, X):
    y_pred = estimator.fit_predict(X)
    return roc_auc_score(y_true=[1 if x == 1 else 0 for x in y_pred],
                         y_score=-estimator.negative_outlier_factor_)

# Apply GridSearchCV to tune hyperparameters
lof = LocalOutlierFactor()
grid_search = GridSearchCV(lof, param_grid=param_grid, scoring=lof_auc_score, cv=5)
grid_search.fit(X_train)

# Print best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Use best hyperparameters to predict anomalies in test data
lof_tuned = LocalOutlierFactor(n_neighbors=grid_search.best_params_['n_neighbors'],
                               contamination=grid_search.best_params_['contamination'])
test_data_lof = scaler.transform(test_data[features])
y_pred_test_lof = lof_tuned.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)

# Plot correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
print(test_data.corr())
plt.figure(figsize=(10, 8 ))
sns.set(font_scale=1.2)
sns.heatmap(train_data.corr(),square=True, annot=True, cbar=True)
plt.show()