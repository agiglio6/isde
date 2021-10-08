import numpy as np

from utils import load_mnist_data, load_mnist_data_openml, split_data, plot_ten_digits
from nmc import NMC

n_rep = 10
x, y = load_mnist_data_openml()

test_error = np.zeros(shape=(n_rep,))
for r in range(n_rep):
    x_tr, y_tr, x_ts, y_ts = split_data(x, y, n_tr=20000)
    clf = NMC()
    clf.fit(x_tr, y_tr)
    # plot_ten_digits(clf.centroids)
    ypred = clf.predict(x_ts)
    test_error[r] = (ypred != y_ts).mean()

print(test_error.mean(), test_error.std())
