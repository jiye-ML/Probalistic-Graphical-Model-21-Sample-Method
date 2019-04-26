import numpy as np


class RandomVariable(object):
    """
    base class for random variables
    """

    def __init__(self):
        self.parameter = {}
        pass

    # 打印的时候，字符串
    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += " " * 4
            if isinstance(value, RandomVariable):
                string += f"{key}={value:8}"
            else:
                string += f"{key}={value}"
            string += "\n"
        string += ")"
        return string

    # 格式化
    def __format__(self, indent="4"):
        indent = int(indent)
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * indent)
            if isinstance(value, RandomVariable):
                string += f"{key}=" + value.__format__(str(indent + 4))
            else:
                string += f"{key}={value}"
            string += "\n"
        string += (" " * (indent - 4)) + ")"
        return string

    def fit(self, X, **kwargs):
        """
        estimate parameter(s) of the distribution

        Parameters
        ----------
        X : np.ndarray
            observed data
        """
        # x 必须是一个 np.ndarray
        self._check_input(X)
        # 调用子类实现
        if hasattr(self, "_fit"):
            self._fit(X, **kwargs)
        else:
            raise NotImplementedError

    # def ml(self, X, **kwargs):
    #     """
    #     maximum likelihood estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_ml"):
    #         self._ml(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def map(self, X, **kwargs):
    #     """
    #     maximum a posteriori estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_map"):
    #         self._map(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def bayes(self, X, **kwargs):
    #     """
    #     bayesian estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_bayes"):
    #         self._bayes(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # 概率密度函数
    def pdf(self, X):
        """
        compute probability density function
        p(X|parameter)

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input of the function

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        self._check_input(X)
        if hasattr(self, "_pdf"):
            return self._pdf(X)
        else:
            raise NotImplementedError

    # 从分布中采样数据
    def draw(self, sample_size=1):
        """

        Parameters
        ----------
        sample_size : int
            sample size

        Returns
        -------
        sample : (sample_size, ndim) np.ndarray
            generated samples from the distribution
        """
        assert isinstance(sample_size, int)
        if hasattr(self, "_draw"):
            return self._draw(sample_size)
        else:
            raise NotImplementedError
        pass

    def _check_input(self, X):
        assert isinstance(X, np.ndarray)
        pass
