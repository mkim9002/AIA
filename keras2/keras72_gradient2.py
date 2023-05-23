import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6
gradient = lambda x: 2*x - 4

x = -10.0  # 초기값
epochs = 20
learning_rate = 0.25

# 변수 초기화
x_list = [x]
f_list = [f(x)]

# 경사 하강법 반복
for i in range(epochs):
    x = x - learning_rate * gradient(x)
    x_list.append(x)
    f_list.append(f(x))

# 그래프 그리기
plt.plot(x_list, f_list, marker='o', linestyle='-', color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.grid(True)
plt.show()

