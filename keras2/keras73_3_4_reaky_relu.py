import numpy as np
import matplotlib.pyplot as plt

# def leaky_relu(x, alpha):
#     return np.where(x > 0, x, alpha * x)

leaky_relu = lambda x, alpha: np.where(x > 0, x, alpha * x)

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x, 0.1)  # alpha 값은 0.1로 설정

plt.plot(x, y)
plt.grid()
plt.show()
