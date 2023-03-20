import numpy as np

a = np.array([[1,2,3], [6,4,5], [7,9,2], [3,2,1], [2,3,1]])
print(a)
print(a.shape) #(5, 3)
print(np.argmax(a)) #7
print(np.argmax(a, axis=0)) #[2 2 1] 0은 행이다,그래서행끼리 비교
print(np.argmax(a, axis=1)) #[2 0 1 0 1] 1은 열이며, 그래서행끼리 비교
print(np.argmax(a, axis=-1)) #[2 0 1 0 1] -1늠 가장 마지막 이란 뜻
  #가장 마지막축, 이건2차원 이니까 가장 마지막 축은
  #그래서 -1을 쓰면 데이터는 1과 동일
