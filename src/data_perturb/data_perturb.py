from abc import ABC, abstractmethod

import numpy as np


class CDataPerturb(ABC):
    """Abstract interface to define data perturbation models"""
    def __init__(self):
        pass

    @abstractmethod
    def data_perturbation(self, x):
        """

        :param x: a flat vector containing n_features elements
        """
        raise NotImplementedError("data_perturbation not implemented")

    def perturb_dataset(self, x):
        """

        :param x: matrix of shape =(n_samples, n_features)
        """
        xp = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            xp[i, :] = self.data_perturbation(x[i, :])
        return xp
