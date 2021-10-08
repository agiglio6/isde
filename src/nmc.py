import numpy as np
from sklearn.metrics import pairwise_distances


class NMC:
    """"Class implementing Nearest Mean Centroid (NMC) classification algorithm"""

    def __init__(self):
        self._centroids = None  # init centroids

    @property # decorator, in this case it allows to use a function as a parameter"""
    def centroids(self):
        """in this way the user can read centroids but not modifying it"""
        return self._centroids

    def fit(self, x_tr, y_tr):
        """Fit the model to the data (estimating centroids)"""
        n_classes = np.unique(y_tr).size
        n_features = x_tr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k in range(n_classes):
            # extract only images of 0 from x_tr
            xk = x_tr[y_tr == k, :]
            self._centroids[k, :] = np.mean(xk, axis=0)
        return self

    def predict(self, x_ts):
        dist = pairwise_distances(x_ts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
