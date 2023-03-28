# 챕터 11, 삼성전자 주가 예측입니다.

import numpy as np
import pandas as pd

df1 = pd.read_csv(".\kospi200\data\kospi200.csv", index_col=0, 
                  header=0, encoding='cp949', sep=',')
# 경로에 주의해 주세요. 경로에서 꼭 틀립니다.                     
print(df1)
print(df1.shape)

df2 = pd.read_csv(".\kospi200\data\samsung.csv", index_col=0, 
                  header=0, encoding='cp949', sep=',')   
print(df2)
print(df2.shape)

# kospi200의 거래량
for i in range(len(df1.index)):     # 거래량 str -> int 변경
        df1.iloc[i,4] = int(df1.iloc[i,4].replace(',', ''))  
# 삼성전자의 모든 데이터 
for i in range(len(df2.index)):     # 모든 str -> int 변경
        for j in range(len(df2.iloc[i])):
                df2.iloc[i,j] = int(df2.iloc[i,j].replace(',', ''))  

df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)

df1 = df1.values
df2 = df2.values
print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('./kospi200/data/kospi200.npy', arr=df1)
np.save('./kospi200/data/samsung.npy', arr=df2)




'''  

               시가  ...       거래량
일자                 ...
2018-05-04  53000  ...  39565391
2018-05-08  52600  ...  23104720
2018-05-09  52600  ...  16128305
2018-05-10  51700  ...  13905263
2018-05-11  52000  ...  10314997
...           ...  ...       ...
2020-01-23  61800  ...  14916555
2020-01-28  59400  ...  23664541
2020-01-29  59100  ...  16446102
2020-01-30  58800  ...  20821939
2020-01-31  57800  ...  19749457

[426 rows x 5 columns]

<class 'numpy.ndarray'> <class 'numpy.ndarray'>
(426, 5) (426, 5)

# cp949
# 인덱스 column을 0번째 줄로 정하겠다는 의미이다.
# header=None  헤더가 없는 파일 불러올때 쓴다. 변수명이 없는 파일.
#####################################################################################

df = df.sort_values(['일자'], ascending=[True]) # 일자별로 정렬. 1996-1-3 ~ 2018-10-11

df["만기일"] = 0                # 만기일 컬럼 추가.

######################### 옵션 만기일 계산 로직 ######################################
# aaa = pd.date_range(df.index[0], df.index[-1], freq = 'WOM-2THU')   # 매월 둘째 목요일 (옵션 만기일)
ac = 0
for i in range(len(df.index)):
    if(pd.date_range(df.index[i], df.index[i], freq = 'WOM-2THU') == df.index[i]):
        # 목요일이 공휴일일 경우, 그 전일, 그전일이 공휴일일경우 그 전전일인 로직을 너어줘야한다.
        # 조건은 그 월에 2째목요일에 주가 데이터가 없으면 공휴일이다. ㅋㅋ
        df.iloc[i,14] = 111          # 만기일 칼럼(14)에 1을 넣는다.
        ac = ac + 1
#####################################################################################        

print(df)
print(len(df.index))        # 5760

print("전체 목요일의 수 : ", ac)


for i in range(len(df.index)):                              # 거래량 str -> int 변경
        df.iloc[i,7] = int(df.iloc[i,7].replace(',', ''))       

# print(int(aaa[0,5].replace(',', '')))

print(df['만기일'])
print(df.iloc[3][7])
print(df.iloc[3, 7])

print(type(df['거래량']))


df.to_csv("./Project6_kospi200/data/kospi200_update.csv", encoding='cp949') # header=False, index=False

# print("=============================")
# print(pd.date_range(df.index[0], df.index[60], freq = 'WOM-2THU' ))


x0_data = df.index
x1_data = df.loc[:, "시가":"현재가"] # 5760 * 4 
x2_data = df.iloc[:, 7]     # "거래량"
x3_data = df['만기일']      # 만기일

x0_data = x0_data.values
x1_data = x1_data.values        # np.ndarray로 변환.
x2_data = x2_data.values        # str으로 되어있어 int로 바꿔야됨        
x3_data = x3_data.values

x0_data = np.reshape(x0_data, (len(x0_data),1)) # (i,) -> (i,1) 로 reshape
x2_data = np.reshape(x2_data, (len(x2_data),1))
x3_data = np.reshape(x3_data, (len(x3_data),1))
# print(x0_data)
# print(type(x0_data))
# print(x0_data.shape)

# print(x1_data)
# print(type(x1_data))
# print(x1_data.shape)

# print(x2_data)
# print(type(x2_data))
# print(x2_data.shape)

x_data = np.hstack((x0_data, x1_data, x2_data, x3_data))
# print(x_data)
# print(type(x_data))
# print(x_data.shape)

# np.savetxt('./Project6_kospi200/data/kospi200_update.csv', x_data, delimiter=',')
# 시간 데이터가 있어서 오류가 나는듯하다.


np.save("./Project6_kospi200/data/kospi200_update.npy", arr=x_data) # 넘파이로 저장.
#aaa = np.load("./Project6_kospi200/data/kospi200_update.npy")      # 로드하기.


'''
'''

# print("=======================")
# a = np.arange(1,11)                 # 1~ 10
# print(a)
# b = np.log(a)                       # log(1) ~ log(10)
# print(b)
# c = np.log(130) - np.log(100)       # log()함수는 밑이 2인 자연로그, log10()함수는 밑이 10인 상용로그
# print(c)

# 데이터 전처리 도중에 98.7.25일자 데이터가 0 인 문제 발견
# log 0 이 되므로 inf 가 발생한다. 이 날짜를 지우게 됨.

y1_data = x1_data[:, 3]
print("x1_data.shape : ", x1_data.shape)        #(5760, 4)
print(y1_data)              
print("y1_data.shape : ", y1_data.shape)        #(5760,)        # DNN 계산용 x1, y1

log_x1_data = np.log(y1_data)
print("log_x1_data : \n", log_x1_data[5700:-1])

log_x2_data = log_x1_data[1:]    
print(log_x1_data.shape)       # (5760,)
print(log_x2_data.shape)       # (5759,)
print(log_x2_data)

print(np.log(287.85))   # 마지막 -1 값      # 검산용
print(np.log(277.05))   # 마지막 값

log_y_data = (log_x2_data - log_x1_data[0:5759])           # 자연로그로 로그수익률을 계산. 전날자 대비 
print(log_y_data[5700:])
print(log_y_data.shape)

print(np.log(287.85)- np.log(289.91))
print(np.log(277.05)- np.log(287.85))   # 마지막 log수익률      # 검산용

############## 정규화 ##################
log_xy_data = (log_y_data - min(log_y_data)) / (max(log_y_data) - min(log_y_data))
print(log_xy_data)
print(np.max(log_y_data))       # 검산  로그수익률 최대값 115.39701865221019
print(np.min(log_y_data))       # 검산  로그수익률 최소값 -127.38951128442366
########################################
print(log_xy_data.shape)



# xy = np.loadtxt(".\Project5_kospi200\data\kospi200.csv", delimiter=',', skiprows=32, skipcolms=1, dtype = np.float32)
# MACD 계산으로 인해 2006년 2월15일까지의 데이터는 사용 못해서 32줄을 스킵함.
# 원래 3073개의 데이터  // ~2500개까지가 트레이닝 이후가 테스트데이터


##########################################  선형 회귀용 데이터 ########################################
# x1_data = xy[:2500, [0]]    # 주가
# x2_data = xy[:2500, [-1]]   # MACD값
# x3_data = xy[:2500, [4]]     # MACD 시그널 값
# y_data = xy[1:2501, [0]]        # y값이당.  x1의 다음날 가격
# x123_data = np.hstack([x1_data, x2_data, x3_data])  # n행3열 데이터로 변환함.
# # hstack : 행의 수가 같은 두 개 이상의 배열을 옆으로 연결하여 열의 수가 더 많은 배열을 만든다.
# # 연결할 배열은 하나의 리스트에 담아야 한다.

# # 테스트용 데이트 분리함.
# x1_test_data = xy[2501:-1, [0]]    # 주가
# x2_test_data = xy[2501:-1, [-1]]   # MACD값
# x3_test_data = xy[2501:-1, [4]]     # MACD 시그널 값
# y_test_data = xy[2502:, [0]]        # y값이당.  x1의 다음날 가격
# x123_test_data  = np.hstack([x1_test_data, x2_test_data, x3_test_data])

# # Make sure the shape and data are OK
# print(x1_data.shape, len(x1_data))      # (2500, 1)
# print(x2_data.shape, len(x2_data))
# print(x3_data.shape, len(x3_data))
# print(y_data.shape, len(y_data))                        
# print(x123_data.shape, len(x123_data))  # (2500, 3)

# print(x1_test_data.shape)   #(571, 1)
# print(x2_test_data.shape)
# print(x3_test_data.shape)
# print(y_test_data.shape)    #(571, 1)
#######################################################################################################


# # 데이터셋 생성 함수
# def seq2dataset(seq, window_size):  # seq의 갯수를 window_size +1 만큼으로 잘라서 [seq-window_size, window_size+1] 행렬로 재정의.
#     dataset = []
#     for i in range(len(seq)-window_size):       # 54 - 4
#         subset = seq[i:(i+window_size + 1 )]       # list / 0:0+4+1 /   0:5 ~ 1:6 ~... # +1 이 중요        
#         dataset.append([code2idx[item] for item in subset]) # 숫자로 변환되서 5개씩 입력됨
#     return np.array(dataset)

# 2. 데이터 셋 생성하기
# dataset = seq2dataset(seq, window_size = 6) # 5일선을 기준으로 계산하기 위해 6개씩 자른다.


#%%

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils

#%%
# a = np.array([11,12,13,14,15,16,17,18,19,20])
a = np.array(range(11,91))
batch_size =1 
window_size = 6 # 자르는 사이즈

def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기
    aaa = []
    for i in range(len(seq)-window_size +1):                 # 열
        subset = seq[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

dataset = split_5(log_xy_data, window_size)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다.
print("===========================")
print(dataset)
print(dataset.shape)    # 5754,6


#입력과 출력을 분리시키기  5개와 1개로

X_train = dataset[:,0:5]
y_train = dataset[:,5]

# X_train = np.reshape(X_train, (len(X_train), window_size-1, batch_size))  # 76,4,1
print(X_train.shape)    # (5754, 5, 1)
print(y_train.shape)    # (5754, )

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score         # 7:3 으로 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=1, test_size = 0.3)
print('X_train.shape : ', X_train.shape )   # (4027, 5)
print('X_test.shape : ', X_test.shape)      # (1727, 5)
print('y_train.shape : ', y_train.shape)    # (4027,)
print('y_test.shape : ', y_test.shape)      # (1727,)


print(X_train[:5])
print(y_train[:5])




# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# import matplotlib.pyplot as plt

# SVG(model_to_dot(model, show_shapes=True, show_layer_names=False).create(prog='dot', format='svg'))

# temp0003 = 0   # 적중률 카운트
# for i in range(len_accuracy_data-1) :    # len(accuracy_data) 는 135개 캐스팅 까먹어서 일단 숫자로 씀
#                         # 135개로 하면 out of size가 걸리므로 134개만 함.
#     temp0004 = accuracy_data[i+1] - accuracy_data[i]
#     #print(accuracy_data[i+1] - accuracy_data[i])
#     #print(temp0004[0], temp0004[1])   # 윗줄이랑 같은 의미 정상작동하는지 확인용
  
#     if temp0004[0] >0 and temp0004[1] >0 :
#         temp0003 = temp0003 + 1   # 적중 카운트 숫자를 센다.
#     elif temp0004[0] <0 and temp0004[1] <0 : 
#         temp0003 = temp0003 + 1   # 적중 카운트 숫자를 센다
         
#     #result_upDown.append(accuracy_data[i+1] - accuracy_data[i])

# print("==================================")
# print("전체데이터 : ", len_accuracy_data-1 )
# print("적  중  율 : ", temp0003 )
# print("예  측  도 : ", temp0003 / (len_accuracy_data-1)) 


'''
