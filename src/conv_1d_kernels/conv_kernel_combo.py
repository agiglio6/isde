from abc import ABC
from copy import deepcopy

from .conv_kernel import CConvKernel


class CConvKernelCombo(CConvKernel, ABC):

    def __init__(self, kernels_seq, kernel_size=3):
        self._kernel_size = kernel_size
        self._mask = None
        self._kernels_seq = kernels_seq

    @property
    def kernels_seq(self):
        return self._kernels_seq

    @kernels_seq.setter
    def kernels_seq(self, value):
        self._kernels_seq = value

    def kernel_mask(self):
        pass

    def kernel(self, x):
        xp = deepcopy(x)
        for conv_kernel in self.kernels_seq:
            xp = conv_kernel.kernel(xp)
        return xp
