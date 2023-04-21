import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime

# Load data
path='c:/study/_data/dacon_cal/'
save_path= 'c:/study/_save/dacon_cal/'
train_df = pd.read_csv(path+'train.csv')
test_df = pd.read_csv(path+'test.csv')
submission = pd.read_csv(path+'sample_submission.csv')


# Separate features and target variable
X_train = train_df[['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)', 'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender', 'Age']]
y_train = train_df['Calories_Burned']

X_test = test_df[['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)', 'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender', 'Age']]

# Convert categorical features to numerical features
X_train = pd.get_dummies(X_train, columns=['Weight_Status', 'Gender'])
X_test = pd.get_dummies(X_test, columns=['Weight_Status', 'Gender'])

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict on training data and calculate RMSE
y_pred_train = model.predict(X_train)
rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
print('RMSE on training data:', rmse_train)

# # Predict on test data and save the result to samplesubmission file
# y_pred_test = model.predict(X_test)
# submission_df = pd.DataFrame({'ID': test_df['ID'], 'Calories_Burned': y_pred_test})
# submission_df.to_csv('samplesubmission.csv', index=False)

# Save submission file with timestamp
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + date + 'submission.csv', index=False)

#model = GradientBoostingRegressor() :RMSE on training data: 3.3384837382279358
#model = RandomForestRegressor() : RMSE on training data: 1.3484722021606528
