import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(150, 4) (150,)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 5):
    if n == 0:
        x_pca = x
    else:
        pca = PCA(n_components=n)
        x_pca = pca.fit_transform(x)
    print(x_pca.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, train_size=0.8, shuffle=True, random_state=123
    )

    # 2. 모델
    model = RandomForestRegressor(random_state=1234)

    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가 예측
    results = model.score(x_test, y_test)
    print(f"n_components={n}, 결과 : {results}")


# n_components=0, 결과 : 0.9676703910614525
# (150, 1)
# n_components=1, 결과 : 0.8699483240223465
# (150, 2)
# n_components=2, 결과 : 0.9068072625698325
# (150, 3)
# n_components=3, 결과 : 0.9258505586592178
# (150, 4)
# n_components=4, 결과 : 0.9142402234636872