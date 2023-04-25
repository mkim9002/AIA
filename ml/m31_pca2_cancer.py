import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(569, 30)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 31):
    if n == 0:
        # PCA를 사용하지 않는 경우
        x_pca = x
    else:
        # PCA를 사용하는 경우
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
    if n == 1:
        print(f"PCA를 사용하지 않는 경우, 결과 : {results}")
    else:
        print(f"n_components={n}, 결과 : {results}")


# PCA를 사용하지 않는 경우, 결과 : 0.9146771132642832
# (569, 2)
# n_components=2, 결과 : 0.796784430337454
# (569, 3)
# n_components=3, 결과 : 0.7941372535917139
# (569, 4)
# n_components=4, 결과 : 0.9058214500501169
# (569, 5)
# n_components=5, 결과 : 0.9007213498162379
# (569, 6)
# n_components=6, 결과 : 0.8930731039091213
# (569, 7)
# n_components=7, 결과 : 0.8873102572669562
# (569, 8)
# n_components=8, 결과 : 0.8990682926829268
# (569, 9)
# n_components=9, 결과 : 0.8979103909121283
# (569, 10)
# n_components=10, 결과 : 0.8989273638489809
# (569, 11)
# n_components=11, 결과 : 0.8974571333110591
# (569, 12)
# n_components=12, 결과 : 0.8951565653190778
# (569, 13)
# n_components=13, 결과 : 0.8946347477447377
# (569, 14)
# n_components=14, 결과 : 0.8913210157033077
# (569, 15)
# n_components=15, 결과 : 0.8877939859672569
# (569, 16)
# n_components=16, 결과 : 0.8903345138656865
# (569, 17)
# n_components=17, 결과 : 0.8827738723688606
# (569, 18)
# n_components=18, 결과 : 0.8866322753090544
# (569, 19)
# n_components=19, 결과 : 0.882354894754427
# (569, 20)
# n_components=20, 결과 : 0.8841031740728366
# (569, 21)
# n_components=21, 결과 : 0.8833642499164718
# (569, 22)
# n_components=22, 결과 : 0.8829376545272302
# (569, 23)
# n_components=23, 결과 : 0.8831052455730036
# (569, 24)
# n_components=24, 결과 : 0.8745581022385567
# (569, 25)
# n_components=25, 결과 : 0.8748437687938523
# (569, 26)
# n_components=26, 결과 : 0.8718461744069496
# (569, 27)
# n_components=27, 결과 : 0.8711262946876044
# (569, 28)
# n_components=28, 결과 : 0.86797253591714
# (569, 29)
# n_components=29, 결과 : 0.8713395923822251
# (569, 30)
# n_components=30, 결과 : 0.8677859004343468