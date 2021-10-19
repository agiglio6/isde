import numpy as np

from .conv_kernel import CConvKernel


class CConvKernelMovingAverage(CConvKernel):

    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)

    def kernel_mask(self):
        x = np.ones(self.kernel_size)
        self._mask = x / np.sum(x)
