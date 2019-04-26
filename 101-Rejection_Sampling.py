'''
2019年04月26日19:57:26

展示两个分布， 一个是容易采样的分布， 一个是不容易采样的分布，我们的目的就是利用拒绝采样工具，
从容易采样的分布中采样出需要的分布的数据
'''

import matplotlib.pyplot as plt
import numpy as np

from rv import Gaussian

np.random.seed(1234)


def func(x):
    return np.exp(-x ** 2) + 3 * np.exp(-(x - 3) ** 2)

x = np.linspace(-5, 10, 100)
rv = Gaussian(mu=np.array([2.]), var=np.array([2.]))

# 需要预测的分布
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
# 一直的高斯分布
plt.plot(x, 15 * rv.pdf(x), label=r"$kq(z)$")
plt.fill_between(x, func(x), 15 * rv.pdf(x), color="gray")
plt.legend(fontsize=15)
plt.show()
