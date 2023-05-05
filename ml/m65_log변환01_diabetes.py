from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Data
datasets = load_diabetes()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

df.info()
print(df.describe())

df['sex'].plot.box()
plt.show()

df['target'].hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

# 2. Log Transformation
y = np.log1p(y)

# 3. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=337
)

# 4. Model
model = RandomForestRegressor(random_state=337)

# 5. Training
model.fit(x_train, y_train)

# 6. Evaluation
score = model.score(x_test, y_test)
r2 = r2_score(np.expm1(y_test), np.expm1(model.predict(x_test)))

print("로그 -> 지수 r2 :", r2)
print('score :', score)
