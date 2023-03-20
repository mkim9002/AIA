#과제
#3가지 원핫인코딩 방식을 비교할것

#1. pandas의 get+dummies
#2. keras 의 to_categorical
#3. sklearn oneHot Encoder

#미세한 차이를 정리

# pandas의 get_dummies:
# 데이터 크기가 작고 메모리 용량이 충분한 경우
# 범주형 변수가 적은 경우
# 인코딩 결과를 바로 데이터프레임으로 반환하여 사용하고자 하는 경우


# keras의 to_categorical:
# 다중 클래스 분류를 수행하는 경우
# keras 모델과 함께 사용하는 경우
# 데이터 크기가 크지 않은 경우


# sklearn의 OneHotEncoder:
# 데이터 크기가 큰 경우
# 범주형 변수가 많은 경우
# 데이터를 희소행렬로 인코딩하여 메모리 문제를 해결해야 하는 경우

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F'],
    'color': ['red', 'green', 'blue', 'red'],
    'age': [20, 30, 25, 35]
})

# 1.pandas의 get_dummies를 사용한 원핫인코딩 예시:

dummies = pd.get_dummies(df[['gender', 'color']])
df_encoded = pd.concat([df['age'], dummies], axis=1)
print(df_encoded)

# 결과
#    age  gender_F  gender_M  color_blue  color_green  color_red
# 0   20         0         1           0            0          1
# 1   30         1         0           0            1          0
# 2   25         0         1           1            0          0
# 3   35         1         0           0            0          1


#2. keras의 to_categorical을 사용한 원핫인코딩 예시:

labels = np.array(['M', 'F', 'M', 'F'])
labels_encoded = to_categorical(labels)
print(labels_encoded)

# 결과
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [1. 0.]]


#3. sklearn의 OneHotEncoder를 사용한 원핫인코딩 예시:

encoder = OneHotEncoder()
encoder.fit(df[['gender', 'color']])
encoded = encoder.transform(df[['gender', 'color']]).toarray()
df_encoded = pd.concat([df['age'], pd.DataFrame(encoded)], axis=1)
print(df_encoded)

# 결과
#    age    0    1    2    3    4
# 0   20  0.0  1.0  0.0  0.0  1.0
# 1   30  1.0  0.0  1.0  0.0  0.0
# 2   25  0.0  1.0  0.0  1.0  0.0
# 3   35  1.0  0.0  0.0  0.0  1.0

# 위 예시에서는 pandas의 get_dummies와 sklearn의 OneHotEncoder가 모두 사용 
# 가능한 데이터 크기와 형태를 가지고 있습니다. 
# 하지만 다중 클래스 분류를 수행할 경우에는 keras의 to_categorical을 사용하는 
# 것이 적합합니다.