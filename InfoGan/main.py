import numpy as np

def rnd_categorical(n, n_categorical):
    """
    :param n: batch size
    :param n_categorical:  分类数量
    :return: one_hot编码和indices索引
    """
    indices = np.random.randint(n_categorical, size=n)
    one_hot = np.zeros((n, n_categorical))
    one_hot[np.arange(n), indices] = 1
    return one_hot, indices


a,b = rnd_categorical(20,10)
c_categorical = np.asarray(a, dtype=np.float32)
categories = np.asarray(b, dtype=np.int32)
print(a)
print(b)