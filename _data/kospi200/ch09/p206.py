#1. 데이터
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
       end_number = i + time_steps
       if end_number > len(dataset) -1:
          break
       tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
       x.append(tmp_x)
       y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 4)
print(x, "\n", y)

'''
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
 [ 5  6  7  8  9 10]
''' 