import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = load_digits()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(1797, 64) (1797,)
df = pd.DataFrame(x, columns=datasets.feature_names)

for n in range(0, 65):
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

# n_components=0, 결과 : 0.8501461803992196
# (1797, 1)
# n_components=1, 결과 : -0.5002529508876943
# (1797, 2)
# n_components=2, 결과 : 0.3993442275604524
# (1797, 3)
# n_components=3, 결과 : 0.6268137828746988
# (1797, 4)
# n_components=4, 결과 : 0.6661035922698662
# (1797, 5)
# n_components=5, 결과 : 0.7411265372425422
# (1797, 6)
# n_components=6, 결과 : 0.7458387760327003
# (1797, 7)
# n_components=7, 결과 : 0.8118344501240389
# (1797, 8)
# n_components=8, 결과 : 0.801968835800867
# (1797, 9)
# n_components=9, 결과 : 0.807663426649363
# (1797, 10)
# n_components=10, 결과 : 0.8186170511428345
# (1797, 11)
# n_components=11, 결과 : 0.817792126846236
# (1797, 12)
# n_components=12, 결과 : 0.8190785637983244
# (1797, 13)
# n_components=13, 결과 : 0.8169517705326165
# (1797, 14)
# n_components=14, 결과 : 0.8182959804362988
# (1797, 15)
# n_components=15, 결과 : 0.8157094048785656
# (1797, 16)
# n_components=16, 결과 : 0.8214323878132973
# (1797, 17)
# n_components=17, 결과 : 0.8216915891975881
# (1797, 18)
# n_components=18, 결과 : 0.8157136071898368
# (1797, 19)
# n_components=19, 결과 : 0.8147117620573668
# (1797, 20)
# n_components=20, 결과 : 0.8140170387830954
# (1797, 21)
# n_components=21, 결과 : 0.8128393675344969
# (1797, 22)
# n_components=22, 결과 : 0.8138531486435187
# (1797, 23)
# n_components=23, 결과 : 0.8154328645460887
# (1797, 24)
# n_components=24, 결과 : 0.8087587820360023
# (1797, 25)
# n_components=25, 결과 : 0.8158966019546045
# (1797, 26)
# n_components=26, 결과 : 0.8029762957862119
# (1797, 27)
# n_components=27, 결과 : 0.8060509397815858
# (1797, 28)
# n_components=28, 결과 : 0.8052012960069215
# (1797, 29)
# n_components=29, 결과 : 0.8019811249128197
# (1797, 30)
# n_components=30, 결과 : 0.8020921859964156
# (1797, 31)
# n_components=31, 결과 : 0.8031585842801776
# (1797, 32)
# n_components=32, 결과 : 0.7980064800346073
# (1797, 33)
# n_components=33, 결과 : 0.7988928498909694
# (1797, 34)
# n_components=34, 결과 : 0.8004343212296174
# (1797, 35)
# n_components=35, 결과 : 0.8037268850809122
# (1797, 36)
# n_components=36, 결과 : 0.7980433120569255
# (1797, 37)
# n_components=37, 결과 : 0.7946283161621245
# (1797, 38)
# n_components=38, 결과 : 0.7927685992001483
# (1797, 39)
# n_components=39, 결과 : 0.793148926026962
# (1797, 40)
# n_components=40, 결과 : 0.7921393119156712
# (1797, 41)
# n_components=41, 결과 : 0.7939641037864943
# (1797, 42)
# n_components=42, 결과 : 0.7904582461530312
# (1797, 43)
# n_components=43, 결과 : 0.791276566817632
# (1797, 44)
# n_components=44, 결과 : 0.7908857165558705
# (1797, 45)
# n_components=45, 결과 : 0.7885960925567886
# (1797, 46)
# n_components=46, 결과 : 0.7931428520980657
# (1797, 47)
# n_components=47, 결과 : 0.7917490619840912
# (1797, 48)
# n_components=48, 결과 : 0.7897535291469132
# (1797, 49)
# n_components=49, 결과 : 0.7855142799127756
# (1797, 50)
# n_components=50, 결과 : 0.7847081071059672
# (1797, 51)
# n_components=51, 결과 : 0.7887555685038536
# (1797, 52)
# n_components=52, 결과 : 0.787156041705291
# (1797, 53)
# n_components=53, 결과 : 0.7928257718215607
# (1797, 54)
# n_components=54, 결과 : 0.789609026140848
# (1797, 55)
# n_components=55, 결과 : 0.7891528105163723
# (1797, 56)
# n_components=56, 결과 : 0.7899710605538929
# (1797, 57)
# n_components=57, 결과 : 0.7913865684950252
# (1797, 58)
# n_components=58, 결과 : 0.7875592517060854
# (1797, 59)
# n_components=59, 결과 : 0.7884147222148652
# (1797, 60)
# n_components=60, 결과 : 0.7921548498733126
# (1797, 61)
# n_components=61, 결과 : 0.7892512293526146
# (1797, 62)
# n_components=62, 결과 : 0.7862635979200325
# (1797, 63)
# n_components=63, 결과 : 0.7874464249454847
# (1797, 64)
# n_components=64, 결과 : 0.7871292387283594
