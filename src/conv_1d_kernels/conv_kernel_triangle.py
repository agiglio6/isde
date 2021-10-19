import numpy as np

from .conv_kernel import CConvKernel


class CConvKernelTriangle(CConvKernel):

    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)

    def kernel_mask(self):
        x = np.arange(1, (self._kernel_size+1)/2)
        y = np.arange((self._kernel_size+1)/2, 0, -1)
        z = np.concatenate((x, y))
        self._mask = z / np.sum(z)
