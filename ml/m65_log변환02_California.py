from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score



#1. data
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot()
# df.plot.box()
# plt.show()

df.info()
print(df.describe())

# df['population'].boxplot()
df['Population'].plot.box()
plt.show()

# df['population'].hist(bins=50)
# plt.show()

df['target'].hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

################ x population 로그변환 ######################
# x['Population'] = np.log1p(x['Population'])        #지수변환 np.explm




# #################################################

x_train,x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=337
)
# ################ y 로그변환 ######################

y_train_log =np.log1p(y_train)
y_test_log = np.log1p(y_test)

#2. model
model = RandomForestRegressor(random_state=337)


#3. compile
model.fit(x_train, y_train_log)

#4 평가 
score = model.score(x_test, y_test_log)

# r2=score("로그 -> 지수 r2 :", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))
r2 = r2_score(np.expm1(y_test), np.expm1(model.predict(x_test)))
print("로그 -> 지수 r2 :", r2)


print('score :', score)


#로그 변화전  score 0.80211
# x  : 0.80226
# y : 0.8244322268075517
#x[pop], y 로그변환 : 0.8244753





