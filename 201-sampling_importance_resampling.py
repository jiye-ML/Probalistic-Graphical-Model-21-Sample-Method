'''
2019年04月26日20:49:11

重要采样
'''

import matplotlib.pyplot as plt
import numpy as np

from rv import Gaussian
from sampling import sir

np.random.seed(1234)

# 真实数据的分布
def func(x):
    return np.exp(-x ** 2) + 3 * np.exp(-(x - 3) ** 2)

x = np.linspace(-5, 10, 100)
# 从高斯分布中采样出一些数据
rv = Gaussian(mu=np.array([2.]), var=np.array([2.]))

# 重要重采样
samples = sir(func, rv, n=100)
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()