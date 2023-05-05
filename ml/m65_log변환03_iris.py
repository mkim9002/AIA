from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Data
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

df.info()
print(df.describe())

df['sepal length (cm)'].plot.box()
plt.show()

df['target'].hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

# 2. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=337, stratify=y
)

# 3. Model
model = RandomForestClassifier(random_state=337)

# 4. Training
model.fit(x_train, y_train)

# 5. Evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
