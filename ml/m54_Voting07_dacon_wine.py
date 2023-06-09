#실습
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier



#1. 데이터
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]
print(train_csv.shape) #(5497,13)
 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)   #[1 0 1 ... 1 1 1]
print(type(aaa))  #<class 'numpy.ndarray'>
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white'])) #[0 1]


#1-1 결측치 제거 
# print(train_csv.isnull().sum()) #결측치없음 

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']



x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()




model = VotingClassifier(
    estimators=[('LR',lr), ('KNN',knn),('DT',dt)],
    voting='soft'   #default
)

#3.훈련
model.fit(x_train,y_train)

#4.평가
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test,y_test))
print('Voting acc :', accuracy_score(y_test,y_pred))


#hard voting 결과
#BaggingClassifier model.score :  0.9912280701754386   acc :  0.9912280701754386
#RandomForestClassifier model.score : 0.956140350877193  acc : 0.956140350877193
#DecisionTreeClassifier : model.score : 0.8947368421052632 acc : 0.8947368421052632
#VotingClassifier : model.score : 0.9824561403508771   acc : 0.9824561403508771     
# soft :   model.score : 0.9824561403508771  acc : 0.9824561403508771

classifiers = [lr,knn,dt]
for model2 in classifiers:
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_predict)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name,score2))


# model.score : 0.9824561403508771
# acc : 0.9824561403508771
# LogisticRegression 정확도 : 0.9737
# KNeighborsClassifier 정확도 : 0.9912
# DecisionTreeClassifier 정확도 : 0.9474