import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print("dataset.shape : ", dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset): # 수정
            break
        tmp_x = dataset[i:x_end_number, :] # 수정
        tmp_y = dataset[x_end_number:y_end_number, :] # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 2)
print(x, "\n", y)
print(x.shape)
print(y.shape)

'''
[[[ 1 11 21]
  [ 2 12 22]
  [ 3 13 23]]

 [[ 2 12 22]
  [ 3 13 23]
  [ 4 14 24]]

 [[ 3 13 23]
  [ 4 14 24]
  [ 5 15 25]]

 [[ 4 14 24]
  [ 5 15 25]
  [ 6 16 26]]

 [[ 5 15 25]
  [ 6 16 26]
  [ 7 17 27]]

 [[ 6 16 26]
  [ 7 17 27]
  [ 8 18 28]]]
 [[[ 4 14 24]
  [ 5 15 25]]

 [[ 5 15 25]
  [ 6 16 26]]

 [[ 6 16 26]
  [ 7 17 27]]

 [[ 7 17 27]
  [ 8 18 28]]

 [[ 8 18 28]
  [ 9 19 29]]

 [[ 9 19 29]
  [10 20 30]]]
(6, 3, 3)
(6, 2, 3)
'''