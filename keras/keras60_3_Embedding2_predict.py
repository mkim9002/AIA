from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Reshape, Embedding
from tensorflow.python.keras.callbacks import EarlyStopping

# 1 data
docs = ['너므 제밋어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천 하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요'
        ]

#긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'참': 1, '잘': 2, '환희가': 3, '너므': 4, '제밋어요': 5, '최고에요': 6, '만든': 7, 
# '영화에요': 8, '추천': 9, '하고': 10, '싶은': 11, '영화입니다': 12, '한': 13, 
# '번': 14, '더': 15, '보고': 16, '싶네요': 17, '글쎄요': 18, '별로에요': 19, '생각보다': 20, 
# '지루해요': 21, '연기가': 22, '어색해요': 23, '재미없어요': 24, '너무': 25, '재미없다': 26, 
# '재밋네요': 27, '생기긴': 28, '했어요': 29, '안해요': 30}

x = token.texts_to_sequences(docs)
print(x)
#[[4, 5], [1, 6], [1, 2, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17], [18], [19], [20, 21],
# [22, 23], [24], [25, 26], [1, 27], [3, 2, 28, 29], [3, 30]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # 0 0 0 4 5, 0 0 0 1 6...(padding)
print(pad_x)
print(pad_x.shape)  #(14, 5) -> (14, 5, 1) LSTM .RNN


# Reshape pad_x to have an additional dimension
pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)

word_size = len(token.word_index)
print("단어사전의 갯수 :", word_size)  #단어사전의 갯수 : 30

#2. 모델
model =  Sequential()
# model.add(Embedding(28, 10))
# model.add(Embedding(input_dim=28, output_dim =10))

# model.add(Embedding(28, 32) input_length=5)   #통상적으로 텍스트 에서는 embedding 이 좋다
model.add(Embedding(input_dim=100, output_dim= 100, input_length=5))  
model.add(LSTM(64))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

model.summary

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(pad_x, labels, epochs=30, batch_size=8)

#4. 평가 예측
acc = model.evaluate(pad_x,labels)[1]
print('acc :', acc)


#5.
x_predict = '나는 성호가 정말 재미없다 너무 정말'

# Convert the new input sequence to a sequence of integers using the tokenizer
x_predict_seq = token.texts_to_sequences([x_predict])

# Pad the sequence with zeros to match the shape of the training data
pad_x_predict = pad_sequences(x_predict_seq, padding='pre', maxlen=5)

# Reshape the padded sequence to have an additional dimension
pad_x_predict = pad_x_predict.reshape(pad_x_predict.shape[0], pad_x_predict.shape[1], 1)

# Use the trained model to predict the sentiment of the new input sequence
y_predict = model.predict(pad_x_predict)

# Print the predicted sentiment (0 for negative, 1 for positive)
if y_predict > 0.5:
    print("성호가 재미 없다는 말에 긍정적 이다")
else:
    print("성호가 재미 없다는 말에 부정적 이다")

############### [실습]###########
#x_predict = '나는 성호가 정말 재미없다 너무 정말'

#긍정인지 부정인지 맞추봐 !!!





