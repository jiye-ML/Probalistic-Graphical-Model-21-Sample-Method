import numpy as np
from rv.rv import RandomVariable


'''
多变量高斯分布
'''
class MultivariateGaussian(RandomVariable):
    """
    @ 表示矩阵或者向量乘法 A @ B np.multi
    The multivariate Gaussian distribution
    p(x|mu, cov)
    = exp{-0.5 * (x - mu)^T @ cov^-1 @ (x - mu)}
      / (2pi)^(D/2) / |cov|^0.5
    """

    def __init__(self, mu=None, cov=None, tau=None):
        super().__init__()
        self.mu = mu
        if cov is not None:
            self.cov = cov
        elif tau is not None:
            self.tau = tau
        else:
            self.cov = None
            self.tau = None
        pass

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 1
            self.parameter["mu"] = mu
        else:
            assert mu is None
            self.parameter["mu"] = None
        pass

    @property
    def cov(self):
        return self.parameter["cov"]

    @cov.setter
    def cov(self, cov):
        if isinstance(cov, np.ndarray):
            assert cov.ndim == 2
            self.tau_ = np.linalg.inv(cov)
            self.parameter["cov"] = cov
        else:
            assert cov is None
            self.tau_ = None
            self.parameter["cov"] = None
        pass

    @property
    def tau(self):
        return self.tau_

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, np.ndarray):
            assert tau.ndim == 2
            self.parameter["cov"] = np.linalg.inv(tau)
            self.tau_ = tau
        else:
            assert tau is None
            self.tau_ = None
            self.parameter["cov"] = None

    @property
    def ndim(self):
        if hasattr(self.mu, "ndim"):
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mu, "size"):
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape
        else:
            return None

    def _fit(self, X):
        self.mu = np.mean(X, axis=0)
        # 求解x的协方差, bias=True则除以N来归一化，否则除以N-1来归一化
        # 保证最少是二维数据
        self.cov = np.atleast_2d(np.cov(X.T, bias=True))
        pass

    def _pdf(self, X):
        d = X - self.mu
        # tau是协方差的逆
        return (
            np.exp(-0.5 * np.sum(np.matmul(d, self.tau) * d, axis=-1))
            * np.sqrt(np.linalg.det(self.tau))
            / np.power(2 * np.pi, 0.5 * self.size))

    def _draw(self, sample_size=1):
        return np.random.multivariate_normal(self.mu, self.cov, sample_size)
