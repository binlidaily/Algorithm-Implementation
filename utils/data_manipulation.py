from __future__ import division
from itertools import combinations_with_replacement
import numpy as np


def divide_on_feature(X, feature_i, threshold):
    split_func = None

    # sample is a row of X, and feature_i is to locate the feature and choose the value comparing to threshold
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    Xy_1 = np.array([sample for sample in X if split_func(sample)])
    Xy_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([Xy_1, Xy_2])


def shuffle_data(X, y, seed=None):
    """ random shuffle of samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # split the training data from test data in the ratio of specific test size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def normalize(X, axis=-1, order=2):
    """Normalize the training data X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def polynomial_features(X, degree):
    """
    Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or
    equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b],
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    :param X:
    :param degree:
    :return:
    """
    n_samples, n_features = np.shape(X)

    def index_combinations():
        # # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
        # get the combinations with length degree
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        # Return the product of array elements over a given axis.
        # >> > np.prod([[1., 2.], [3., 4.]], axis=1)
        # array([2., 12.])

        # take a look: https://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

def standardize(X):
    """ Standardize the dataset X """
    X_std = X

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # std = sqrt(mean(abs(x - x.mean())**2))

    # $$
    # y _ { i } = \frac { x _ { i } - \overline { x } } { s }
    # $$

    for col in range(np.shape(X)[1]):
        # comfirm that std is not zero
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

    return X_std

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def make_diagonal(x):
    """ Converts a vector into an diagonal matrix """
    m = np.zeros((len(x), len(x)))
    # for i in range(len(m[0])):
    for i in range(len(x)):
        m[i, i] = x[i]
    return m