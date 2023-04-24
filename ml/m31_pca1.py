import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(442, 10)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 11):
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


# n_components=0, 결과 : 0.5139768412680075
# (442, 1)
# n_components=1, 결과 : 0.050755953397053966
# (442, 2)
# n_components=2, 결과 : 0.12235427079587391
# (442, 3)
# n_components=3, 결과 : 0.19289910290085954
# (442, 4)
# n_components=4, 결과 : 0.4791220458288944
# (442, 5)
# n_components=5, 결과 : 0.4705347543022226
# (442, 6)
# n_components=6, 결과 : 0.4741577299072418
# (442, 7)
# n_components=7, 결과 : 0.5028415915849482
# (442, 8)
# n_components=8, 결과 : 0.49777302535691514
# (442, 9)
# n_components=9, 결과 : 0.48061460577634196
# (442, 10)
# n_components=10, 결과 : 0.49825261578787894