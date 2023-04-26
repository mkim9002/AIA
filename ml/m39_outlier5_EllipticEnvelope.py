import numpy as np
aaa = np.array([[-10, 2, 3, 4 ,5, 6,700, 8, 9, 10, 11, 12, 50],
               [100,200,-30,400,500,600,-70000,
                800,900,1000,210,420,350]])
aaa = np.transpose(aaa)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

outliers.fit(aaa)
results = outliers.predict(aaa)

outliers_index = np.where(results == -1)[0]
outliers_values = aaa[outliers_index]

print("이상치 위치 : ", outliers_index)
print("이상치 값 : ", outliers_values)

#형태와 위치가 두개가 나올수 있게 수정