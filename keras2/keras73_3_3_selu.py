import numpy as np
import matplotlib.pyplot as plt

# def selu(x, scale, alpha):
#     return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

selu = lambda x, scale, alpha: np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = selu(x, 1.0507, 1.67326)  # scale 값은 1.0507, alpha 값은 1.67326으로 설정

plt.plot(x, y)
plt.grid()
plt.show()
