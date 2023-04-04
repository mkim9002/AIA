from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical 


# 1. 데이터
path = 'd:/study_data/_data/gas/'  
save_path = 'd:/study_data/_save/gas/'


train_csv = pd.read_csv(path + 'train_data.csv', 
                        index_col=0) 

test_csv = pd.read_csv(path + 'test_data.csv',
                       index_col=0) 

print(test_csv)   # [7389 rows x 7 columns]

print(train_csv)  #[2463 rows x 7 columns]

#1-2. 결측치 처리 .제거

print(train_csv.isnull().sum())
train_csv = train_csv.dropna() 
print(train_csv.isnull().sum()) 
print(train_csv.info())
print(train_csv.shape)


# 1-2 train_csv 데이터에서 x와 y를 분리
x = train_csv.drop(['type'], axis=1) #2개 이상 리스트 
print(x)
y = train_csv['type']
print(y) #Name: type, Length: 2463, dtype: int64
print(np.unique(y, return_counts=True))
print(np.unique(y)) #(array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int64), array([432, 369, 366, 306, 306, 249, 249, 186], dtype=int64))

# 1-3 Reshape
print(x.shape) #(2463, 7)
print(y.shape) #(2463,)
print(train_csv.shape) #(2463, 7)
print(test_csv.shape) #(7389, 7)

x = np.array(x)
x = x.reshape(2463, 3, 2, 1)

print(test_csv.shape) #(7389, 7)

test_csv = np.array(test_csv)
test_csv = test_csv.reshape(7389, 1, 7, 1)

# 1-4 train_csv 데이터에서 x와 y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=777
)
print(x_train.shape, x_test.shape) #(1724, 3, 2, 1) (739, 3, 2, 1)
print(y_train.shape, y_test.shape) #(1724,) (739,)

# 1-5 Scaler
x_train = x_train.reshape(1724, 6)  # Reshape
x_test = x_test.reshape(739, 6)  # Reshape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 1-5 Scaler
x_train = x_train.reshape(1724, 6)  # Reshape
x_test = x_test.reshape(739, 6)  # Reshape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
#2. 모델 구성
model = Sequential()
model.add(Conv2D(6,(2,2),padding='same', input_shape=(1,7,1)))
model.add(Conv2D(filters=4, padding='same', kernel_size=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax')) # change the number of neurons and the activation function

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', # change the loss function
              metrics=['accuracy','mse'])

# convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# ...

# convert predicted values to class labels
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) + 1 # add 1 to convert from zero-indexed to 1-indexed labels
y_test_acc = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc, y_predict)


print('accuracy score :',acc)


f1_score_macro = f1_score(y_test, y_predict, average='macro')
print("macro F1-Score:", f1_score_macro)


#submission.csv 만들기
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(y_submit)
# print(y_submit.shape)
y_submit = np.argmax(y_submit, axis=1)
print(y_submit.shape)
y_submit += 3
submission['quality'] = y_submit
# print(submission)

path_save = 'd:/study_data/_save/gas/' 
submission.to_csv(path_save + 'gas_01.csv')