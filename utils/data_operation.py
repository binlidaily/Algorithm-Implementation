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


def calculate_entropy(y):
	log2 = lambda x: math.log(x) / math.log(2)
	entropy = 0
	unique_labels = np.unique(y)

	for label in unique_labels:
		cnt = len(y[y == label])
		prob = float(cnt) / len(y)
		entropy += - prob * log2(prob)

	return entropy

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_variance(X):
    """ Return the Variance of features in Dataset X"""
    mean = np.ones(np.shape(X)) * X.mean(0)   # mean of every columns
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance