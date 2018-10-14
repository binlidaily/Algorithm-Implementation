import math
import numpy as np
import sys
sys.path.insert(0, '/Users/Bin/Dropbox/Codes/ML-From-Scratch/ml-algs/utils/')
from data_manipulation import polynomial_features, normalize

class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)
    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)
    def grad(self, w):
        return self.alpha * w

class l1_l2_regularization():
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)

class Regression(object):
    """
    Base regression model. Models the relationship between a scalar dependent variable y and the independent
        variables X.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        # self.regularization = lambda x: 0
        # self.regularization.grad = lambda x: 0
        self.training_errors = []

    def initialize_weights(self, n_features):
        """
        Initialize weights randomly [-1/sqrt(N), 1/sqrt(N)]
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.normal(-limit, limit, (n_features,))

    def fit(self, X, y):
        # add bias weights (set 1 default) to training data X
        X = np.insert(X, 0, 1, axis=1)

        # self.training_errors = []
        # store the training errors for plotting
        self.initialize_weights(n_features=X.shape[1])

        # we use matrix operation
        # do gradient descent for n_features
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)

            # calculate l2 loss, // regularization is a class, need initialize in the successors
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
            self.training_errors.append(mse)

            # gradient of l2 loss w.r.t. w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)

            # update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # add bias weights (set 1 default) to training data X
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """
    Linear model.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.

    """

    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_desent = gradient_descent
        # regularization is a class
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_desent:
            # add bias weights (set 1 default) to training data x
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            # Use Projection Matrix calculate weights
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).fit(X, y)
        else:
            super(LinearRegression, self).fit(X, y)

class PolynomialRegression(Regression):
    """
    Perform a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.

    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        self.degree = degree
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)


class LocallyWeightedLinearRegression(object):
    """
    Locally weighted linear regression
    * implementation 1: https://medium.com/100-days-of-algorithms/day-97-locally-weighted-regression-c9cfaff087fb
    * implementation 2: https://github.com/arokem/lowess
    """

    def __init__(self, tau=0.1):
        self.tau = tau
        # beta is the parameters of this algorithm
        self.beta = None

    def radial_kernel(self, x0, X):
        # there is axis=1, so the number of result is as same as X
        return np.exp(np.sum((X-x0)**2, axis=1) / (-2 * self.tau * self.tau))

    def fit(self, x0, X, y):
        # add bias term
        x0 = np.insert(x0, 0, 1)
        X = np.insert(X, 0, 1, axis=1)

        # intermedial variable
        xw = X.T * self.radial_kernel(x0, X)
        self.beta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(xw, X)), xw), y)

    def predict(self, x0, X, y):
        self.fit(x0, X, y)
        # print np.shape(x0), np.shape(self.beta)
        return np.matmul(np.insert(x0, 0, 1), self.beta)

class RidgeRegression(Regression):
    """
    Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, reg_factor, n_iterations=100, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations=n_iterations,
                                              learning_rate=learning_rate)

class PolynomialRidgeRegression(Regression):
    """
    Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations=n_iterations,
                                                        learning_rate=learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).predict(X)


class LassoRegression(Regression):
    """
    Linear regression model with a regularization factor which does both variable selection
    and regularization. Model that tries to balance the fit of the model with respect to the training
    data and the complexity of the model. A large regularization factor with decreases the variance of
    the model and do para.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations,
                                              learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X)

class ElasticNet(Regression):
    """ Regression where a combination of l1 and l2 regularization are used. The
    ratio of their contributions are set with the 'l1_ratio' parameter.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    l1_ration: float
        Weighs the contribution of l1 and l2 regularization.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000,
                learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations,
                                        learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)