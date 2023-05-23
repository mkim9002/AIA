import numpy as np
import matplotlib.pyplot as plt

# def elu(x, alpha):
#     return np.where(x > 0, x, alpha * (np.exp(x) - 1))

elu = lambda x, alpha: np.where(x > 0, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = elu(x, 1.0)  # alpha 값은 1.0으로 설정

plt.plot(x, y)
plt.grid()
plt.show()
