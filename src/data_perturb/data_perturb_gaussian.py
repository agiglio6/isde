import numpy as np

from data_perturb import CDataPerturb


class CDataPerturbGaussian(CDataPerturb):

    def __init__(self, sigma=100.0, min_value=0, max_value=255):
        super().__init__()
        self.sigma = sigma
        self.min_value = min_value
        self.max_value = max_value

    @property
    def sigma(self):
        return self._sigma

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @sigma.setter
    def sigma(self, value):
        self._sigma = int(value)

    @min_value.setter
    def min_value(self, value):
        self._min_value = int(value)

    @max_value.setter
    def max_value(self, value):
        self._max_value = int(value)

    def data_perturbation(self, x):
        if x.size != x.shape[0]:
            raise TypeError("x is not flattened!")

        xp = x.copy().ravel()
        xp += self.sigma * np.random.randn(xp.size)

        xp[x < self.min_value] = self.min_value
        xp[x > self.max_value] = self.max_value

        return xp
