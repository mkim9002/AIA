from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# PCA를 사용하여 0.95, 0.99, 0.999, 1.0 이상의 variance ratio를 가지는 feature 개수를 구함
pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d95 = np.argmax(cumsum >= 0.95) + 1
d99 = np.argmax(cumsum >= 0.99) + 1
d999 = np.argmax(cumsum >= 0.999) + 1
d100 = np.argmax(cumsum == 1.0) + 1

# 각각의 feature 개수에 대해 RandomForestClassifier 모델의 정확도 계산
for d in [d95, d99, d999, d100]:
    pca = PCA(n_components=d)
    x_pca = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"PCA {d}: {acc:.4f}")



##############################################
##########실습################################
#############################################3

# y를 빼주고
#                acc
#1. 나의 최고의 CNN : 0.8911699779249448
#2. 나의 최고의 DNN : 0.9606999754905701
#3. PCA(154) 0.95  : 0.8896
#4. PCA(331) 0.99  : 0.8614
#5. PCA(486) 0.999  : 0.8240
#6. PCA(1) 1.0 : 0.2433







