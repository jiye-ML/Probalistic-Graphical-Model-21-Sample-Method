import numpy as np

# 重要采样
def sir(func, rv, n):
    """
    sampling-importance-resampling

    Parameters
    ----------
    func : callable
        (un)normalized distribution to be sampled from
    rv : RandomVariable
        distribution to generate sample
    n : int
        number of samples to draw

    Returns
    -------
    sample : (n, ndim) ndarray
        generated sample
    """
    assert hasattr(rv, "draw"), "the distribution has no method to draw random samples"
    # 采样数据
    sample_candidate = rv.draw(n * 10)
    # 权值
    weight = np.squeeze(func(sample_candidate) / rv.pdf(sample_candidate))
    assert weight.shape == (n * 10,), weight.shape
    # 归一化
    weight /= np.sum(weight)
    # 根据权值采样数据
    index = np.random.choice(n * 10, n, p = weight)
    sample = sample_candidate[index]
    return sample
