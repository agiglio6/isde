import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from utils import load_mnist_data, load_mnist_data_openml, split_data, plot_ten_digits
from nmc import NMC
from data_perturb import CDataPerturbRandom, CDataPerturbGaussian

x, y = load_mnist_data()

"""
plot_ten_digits(x, y)

data_pert = CDataPerturbRandom()
xp = data_pert.perturb_dataset(x)
plot_ten_digits(xp, y)

data_pert = CDataPerturbGaussian()
xp = data_pert.perturb_dataset(x)
plot_ten_digits(xp, y)

plot_ten_digits(x, y)
"""

clf = NMC()
splitter = ShuffleSplit(train_size=0.6)
class_accuracy = np.zeros(shape=(splitter.n_splits,))

for i, (tr_idx, ts_idx) in enumerate(splitter.split(x, y)):
    x_tr, y_tr = x[tr_idx, :], y[tr_idx]
    x_ts, y_ts = x[ts_idx, :], y[ts_idx]
    clf.fit(x_tr, y_tr)
    ypred = clf.predict(x_ts)
    class_accuracy[i] = (ypred == y_ts).mean()
print("Classification accuracy not perturbed model: " + str(class_accuracy.mean()))

K = [0, 10, 20, 50, 100, 200, 500]
Sigma = [10, 20, 200, 200, 500]

for k in K:
    data_pert = CDataPerturbRandom(K=k)
    xp = data_pert.perturb_dataset(x)
    for i, (tr_idx, ts_idx) in enumerate(splitter.split(xp, y)):
        x_tr, y_tr = xp[tr_idx, :], y[tr_idx]
        x_ts, y_ts = xp[ts_idx, :], y[ts_idx]
        clf.fit(x_tr, y_tr)
        ypred = clf.predict(x_ts)
        class_accuracy[i] = (ypred == y_ts).mean()
    print("Classification accuracy random perturbed model with K=" + str(k) + ": " + str(class_accuracy[i]))

for sigma in Sigma:
    data_pert = CDataPerturbGaussian(sigma=sigma)
    xp = data_pert.perturb_dataset(x)
    for i, (tr_idx, ts_idx) in enumerate(splitter.split(xp, y)):
        x_tr, y_tr = xp[tr_idx, :], y[tr_idx]
        x_ts, y_ts = xp[ts_idx, :], y[ts_idx]
        clf.fit(x_tr, y_tr)
        ypred = clf.predict(x_ts)
        class_accuracy[i] = (ypred == y_ts).mean()
    print("Classification accuracy gaussian perturbed model with sigma=" + str(sigma) + ": " + str(class_accuracy[i]))
