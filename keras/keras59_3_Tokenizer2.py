from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'

token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)  #{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
#가장 많은 음절이 맨 앞에 나온다. 그 외는 순서대로
print(token.word_counts)  #단어 갯수
#OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

# print(x)  #[[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]


x = token.texts_to_sequences([text1, text2])
x = x[0] + x[1]
print(x)

###### one hot           ##########  
###### 1. to_categorical ##########

# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape)   


###### 2. get_dummies #####  1차원으로 받아 들여야 한다
# import pandas as pd
# import numpy as np
# # x = pd.get_dummies(np.array(x).reshape(11,))  #두개중 하나 쓴다
# x = pd.get_dummies(np.array(x).ravel())
# print(x)
# print(x.shape)  #(18, 13)


# ########### 3. 사이킷런 one hot  ######
from sklearn.preprocessing import OneHotEncoder
import numpy as np
ohe = OneHotEncoder()
x = ohe.fit_transform(np.array(x).reshape(-1,1)).toarray()
print(x)
print(x.shape) #(18, 13)






















