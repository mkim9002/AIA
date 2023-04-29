import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier

# Load data
train = pd.read_csv('d:/study/_data/dacon_air/train.csv')
test = pd.read_csv('d:/study/_data/dacon_air/test.csv')
sample_submission = pd.read_csv('d:/study/_data/dacon_air/sample_submission.csv', index_col=0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
NaN = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']

for col in NaN:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

label_encoders = {}  # Dictionary to store label encoders

for i in qual_col:
    le = LabelEncoder()
    le.fit(train[i])
    train[i] = le.transform(train[i])
    test[i] = np.where(test[i].isin(le.classes_), test[i], le.transform(test[i]))
    label_encoders[i] = le  # Store label encoder in the dictionary
print('Done.')

# Remove unlabeled data
train.dropna(inplace=True)

column_dict = {column: i for i, column in enumerate(sample_submission.columns)}

def to_number(x, dic):
    return dic[x]

train['Delay_num'] = train['Delay'].map(lambda x: to_number(x, column_dict))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model and hyperparameter tuning using GridSearchCV
model = XGBClassifier(random_state=42)

param_grid = {
    'n_estimators': [1000],
    'learning_rate': [0.1],  # This seems to be the best performing value
    'max_depth': [1],
    'gamma': [11],
    'min_child_weight': [11],
    'subsample': [1],
    'colsample_bytree': [1],
    'colsample_bylevel': [1],
    'colsample_bynode': [1],
    'reg_alpha': [1],
    'reg_lambda': [1],
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(train_x, train_y)
best_model = grid.best_estimator_

# Model evaluation
val_x_encoded = val_x.copy()

for col, le in label_encoders.items():
    val_x_encoded[col] = np.where(val_x_encoded[col].isin(le.classes_), val_x_encoded[col], le.transform(val_x_encoded[col]))

val_y_pred = best_model.predict(val_x_encoded)
acc = accuracy_score(val_y, val_y_pred)
f1 = f1_score(val_y, val_y_pred, average='weighted')
pre = precision_score(val_y, val_y_pred, average='weighted')
recall = recall_score(val_y, val_y_pred, average='weighted')

print('Accuracy_score:', acc)
print('F1 Score:', f1)

test_x_encoded = test_x.copy()

for col, le in label_encoders.items():
    test_x_encoded[col] = np.where(test_x_encoded[col].isin(le.classes_), test_x_encoded[col], le.transform(test_x_encoded[col]))

y_pred = best_model.predict_proba(test_x_encoded)
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('d:/study/_data/dacon_air/submit16.csv')