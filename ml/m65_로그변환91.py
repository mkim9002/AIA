import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)

#로그변환
log_data = np.log(data)

#원본 데이터 히스토그램 그려
plt.subplot(1, 2, 1)
plt.hist(data, bins =50, color='blue', alpha=0.5)
plt.title('origonal')

#원본 데이터 히스토그램 그려
plt.subplot(1, 2, 1)
plt.hist(data, bins =50, color='blue', alpha=0.5)
plt.title('origonal')

plt.subplot(1, 2, 2)
plt.hist(log_data, bins =50, color='blue', alpha=0.5)
plt.title('log transfered data')

plt.show()
