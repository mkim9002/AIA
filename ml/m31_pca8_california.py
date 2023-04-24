import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets =fetch_california_housing()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(20640, 8) (20640,)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 9):
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


# n_components=0, 결과 : 0.8144038389277144
# (20640, 1)
# n_components=1, 결과 : -0.4453617583734859
# (20640, 2)
# n_components=2, 결과 : 0.0508425839248976
# (20640, 3)
# n_components=3, 결과 : 0.07977526553439707
# (20640, 4)
# n_components=4, 결과 : 0.3286665436119468
# (20640, 5)
# n_components=5, 결과 : 0.5927869104090877
# (20640, 6)
# n_components=6, 결과 : 0.7012788833098638
# (20640, 7)
# n_components=7, 결과 : 0.7776893227910943
# (20640, 8)
# n_components=8, 결과 : 0.781338697297561