import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder, StandardScaler




path = 'c:/study/_data/AIFac_pollution/'
save_path = 'c:/study/_save/AIFac_pollution/'

# _data
#     TRAIN
#     TRAIN_AWS
#     TEST_INPUT
#     TEST_AWS
#     Meta
#     answer_sample.csv

#변수명과 random state바꿔라


train_files = glob.glob(path + "TRAIN/*.csv")
# print(train_files)
test_input_files = glob.glob(path + 'test_input/*.csv') 
# print(test_input_files)


################## train folder ##########################
li = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
# print(li) #35064...7개
# print(len(li))  #17

train_dataset = pd.concat(li, axis=0,
                          ignore_index=True)
# print(train_dataset)
#[596088 rows x 4 columns]

################## test folder #################################

li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
# 
# 

test_input_dataset = pd.concat(li, axis=0,
                          ignore_index=True)
#
#

################# 측정소 라벨 인코더 #########################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])
print(train_dataset)  #[596088 rows x 5 columns]
print(test_input_dataset)  #[131376 rows x 5 columns]
train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1)
print(train_dataset)  #[596088 rows x 4 columns]
print(test_input_dataset)  #[131376 rows x 4 columns]





    
    
    





