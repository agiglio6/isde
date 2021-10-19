from abc import ABC, abstractmethod

import numpy as np


class CConvKernel(ABC):

    def __init__(self, kernel_size=3):
        self._mask = None
        self.kernel_size = kernel_size

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def mask(self):
        return self._mask

    @kernel_size.setter
    def kernel_size(self, value):
        if int(value) % 2 == 0:
            raise ValueError("Kernel size must be an odd integer.")
        self._kernel_size = int(value)
        self.kernel_mask()

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError("kernel_mask not implemented")

    def kernel(self, x):
        if x.size != x.shape[0]:
            raise TypeError("x must be a flattened array.")

        if self._mask is None:
            raise TypeError("Kernel mask not defined.")

        xp = np.convolve(x, self._mask, 'same')

        return xp
