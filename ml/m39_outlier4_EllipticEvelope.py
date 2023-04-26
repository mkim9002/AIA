import numpy as np
aaa = np.array([-10, 2, 3, 4 ,5, 6,700,
                8, 9, 10, 11, 12, 50])
aaa = aaa.reshape(-1, 1)   #요구한 형식이 이차원이다

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)