import numpy as np


dataset = np.array(range(1, 11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)

# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
# (6, 5)

# x = bbb[:,:4]
x = bbb[:, :-1]
y = bbb[:,-1]
print(x)
print(y)
