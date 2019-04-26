'''
2019年04月26日19:57:26

展示两个分布， 一个是容易采样的分布， 一个是不容易采样的分布，我们的目的就是利用拒绝采样工具，
从容易采样的分布中采样出需要的分布的数据
'''

import matplotlib.pyplot as plt
import numpy as np

from rv import Gaussian
from sampling import rejection_sampling

np.random.seed(1234)

# 真实数据的分布
def func(x):
    return np.exp(-x ** 2) + 3 * np.exp(-(x - 3) ** 2)


x = np.linspace(-5, 10, 100)
# 从高斯分布中采样出一些数据
rv = Gaussian(mu=np.array([2.]), var=np.array([2.]))

# 利用拒绝采样工具，从容易采样的分布中采样出数据，
samples = rejection_sampling(func, rv, k=15, n=100)

# 展示
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()
