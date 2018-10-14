from __future__ import division
import numpy as np
import math
import sys


def mean_squared_error(y_true, y_pred):
    """return the mean squared error between y_true and y_pred"""
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)

