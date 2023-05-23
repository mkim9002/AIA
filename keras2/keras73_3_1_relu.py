import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0,x)

relu = lambda x: np.maximum(0,x)

x = np.arange(-5,5,0.1)
y =relu(x)

plt.plot(x,y)
plt.grid()
plt.show()

#3_2, 3_3, 3_4...
# elu, selu, reaky_relu,...


