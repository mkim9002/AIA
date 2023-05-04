from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer
from sklearn.datasets import load_wine, fetch_covtype, load_digits, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = [fetch_california_housing(), load_iris(), load_breast_cancer(), load_wine(),
            fetch_covtype(), load_digits(), load_diabetes()]

scalers = [QuantileTransformer(), StandardScaler(), MinMaxScaler(), MaxAbsScaler(),
           RobustScaler(), PowerTransformer(method='yeo-johnson')]

for dataset in datasets:
    print('Dataset:', dataset.__class__.__name__)
    print('-------------------')

    x, y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=337, train_size=0.8)

    for scaler in scalers:
        print('Scaler:', scaler.__class__.__name__)

        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = RandomForestRegressor()
        model.fit(x_train_scaled, y_train)

        y_pred = model.predict(x_test_scaled)
        r2 = r2_score(y_test, y_pred)

        print('R2 score:', r2)
        print('---')
