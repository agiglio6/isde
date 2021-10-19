import random
import numpy as np
import matplotlib.pyplot as plt

from conv_1d_kernels import CConvKernelMovingAverage, CConvKernelTriangle, CConvKernelCombo
from utils import load_mnist_data

x, y = load_mnist_data()

mov_avg_kernel = CConvKernelMovingAverage()
tri_kernel = CConvKernelTriangle()

combo_kernel = CConvKernelCombo(kernels_seq=(mov_avg_kernel, tri_kernel, mov_avg_kernel))

rand_idx = np.zeros(np.unique(y).size)
for i, label in enumerate(np.unique(y)):
    idx = np.array(range(0, y.shape[0]))[y == label]
    rand_idx[i] = random.choice(idx)

for i, k in enumerate(rand_idx):
    img = x[int(k), :].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.title("Label: " + str(y[int(k)]))
        plt.savefig('digit' + str(i) + '.pdf')

for i, k in enumerate(rand_idx):
    img = mov_avg_kernel.kernel(x[int(k)].ravel()).reshape(28, 28)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.title("Label: " + str(y[int(k)]))
        plt.savefig('digit' + str(i) + '_moving_avg.pdf')

for i, k in enumerate(rand_idx):
    img = tri_kernel.kernel(x[int(k)].ravel()).reshape(28, 28)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.title("Label: " + str(y[int(k)]))
        plt.savefig('digit' + str(i) + '_triangle.pdf')

for i, k in enumerate(rand_idx):
    img = combo_kernel.kernel(x[int(k)].ravel()).reshape(28, 28)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.title("Label: " + str(y[int(k)]))
        plt.savefig('digit' + str(i) + '_combo.pdf')
