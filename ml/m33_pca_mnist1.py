from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)

pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d95 = np.argmax(cumsum >= 0.95) + 1
d99 = np.argmax(cumsum >= 0.99) + 1
d999 = np.argmax(cumsum >= 0.999) + 1
d100 = np.argmax(cumsum == 1.0) + 1

print(f"0.95 이상의 n_components 개수: {d95}")
print(f"0.99 이상의 n_components 개수: {d99}")
print(f"0.999 이상의 n_components 개수: {d999}")
print(f"1.0 이상의 n_components 개수: {d100}")


####실습##########
#pca 를 통해 0.95 이상인 n_components는 몇개?
#0.95 몇개?
#0.99 몇개??
#0.999 몇개?
#1.0 몇개???

####실습##########

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
cumsum =np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.95) +1)   #154
print(np.argmax(cumsum >= 0.99) +1)   #331
print(np.argmax(cumsum >= 0.999) +1)   #486
print(np.argmax(cumsum >= 1.0) +1)   #713





