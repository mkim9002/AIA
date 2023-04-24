import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_wine()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(178, 13) (178,)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 14):
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


# n_components=0, 결과 : 0.9509186813186813
# (178, 1)
# n_components=1, 결과 : 0.2902285714285715
# (178, 2)
# n_components=2, 결과 : 0.44080439560439566
# (178, 3)
# n_components=3, 결과 : 0.6429758241758241
# (178, 4)
# n_components=4, 결과 : 0.7580835164835165
# (178, 5)
# n_components=5, 결과 : 0.7367120879120879
# (178, 6)
# n_components=6, 결과 : 0.7677758241758241
# (178, 7)
# n_components=7, 결과 : 0.7719296703296703
# (178, 8)
# n_components=8, 결과 : 0.7662461538461538
# (178, 9)
# n_components=9, 결과 : 0.7604263736263737
# (178, 10)
# n_components=10, 결과 : 0.7390769230769232
# (178, 11)
# n_components=11, 결과 : 0.7473846153846153
# (178, 12)
# n_components=12, 결과 : 0.7375560439560439
# (178, 13)
# n_components=13, 결과 : 0.7448879120879122