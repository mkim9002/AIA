import numpy as np

datasets = np.array(range(1,41)).reshape(10,4)  #1차원의 1-40 까지 벡터
# print(datasets)
# print(datasets.shape) #(10,4)

x_data = datasets[:,:-1]
y_data = datasets[:, -1]

# print(x_data)
# print(y_data)
# print(x_data.shape, y_data.shape)

timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps ):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(x_data, timesteps)
# print(x_data)
print(x_data.shape) #(5, 5, 3)

##########  y 만들기   ##########

y_data = y_data[timesteps:]
print(y_data)
print(y_data.shape)




