import numpy as np
from sklearn import linear_model
from numpy import genfromtxt
import sys

sys.path.insert(0, '/Users/Bin/Dropbox/Codes/ml-algs/utils/')
from compare_sklearn import compare_sklearn


class GradientDescent(object):
    """Gradient Descent"""

    def __init__(self, iter_max=1000, W=None, b=None, alpha=0.01):
        super(GradientDescent, self).__init__()
        self.iter_max = iter_max
        self.W = W
        self.b = b
        self.alpha = alpha

    def _sse(self, X, y):
        m = len(X)
        cost_sum = 0
        for i in range(m):
            cost_sum += pow(np.matmul(X[i], self.W) + self.b - y[i], 2)
        return cost_sum / (2 * m)

    def _derivative_W(self, X, y):
        m = len(X)
        der_sum = 0
        for i in range(m):
            # der_sum += (np.matmul(X[i], self.W)+self.b - y[i])*X[i]
            der_sum += (np.matmul(X[i], self.W) + self.b - y[i]) * X[i]
        return der_sum

    def _derivative_b(self, X, y):
        m = len(X)
        der_sum = 0
        for i in range(m):
            der_sum += (np.matmul(X[i], self.W) + self.b - y[i])
        return der_sum

    def fit(self, X, y):
        m, n = np.shape(X)
        self.W = [0] * n
        self.b = 0
        for j in range(self.iter_max):
            pre_sse = self._sse(X, y)
            print("---------------------iteration: %d------------------------" % (j + 1))
            print("before update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))
            print("derivative of (W, b)=(%lf, %lf)" % (self._derivative_W(X, y), self._derivative_b(X, y)))

            self.W = self.W - 1.0 / m * self.alpha * self._derivative_W(X, y)
            self.b = self.b - 1.0 / m * self.alpha * self._derivative_b(X, y)
            print("after update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))
            now_sse = self._sse(X, y)

            print("pre_sse = %lf, now_sse = %lf)" % (pre_sse, now_sse))
            if (abs(pre_sse - now_sse) < 0.001):
                break

        # print log
        print("function is: y = %lfx + %lf" % (self.W, self.b))

    def fit_lr(self, X, y):
        m, n = np.shape(X)
        self.W = [0] * n
        self.b = 0

        b_lr = 0.0
        W_lr = 0.0

        for j in range(self.iter_max):
            pre_sse = self._sse(X, y)
            print("---------------------iteration: %d------------------------" % (j + 1))
            print("before update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))
            print("derivative of (W, b)=(%lf, %lf)" % (self._derivative_W(X, y), self._derivative_b(X, y)))

            W_grad = self._derivative_W(X, y)
            b_grad = self._derivative_b(X, y)

            W_lr = W_lr + W_grad ** 2
            b_lr = b_lr + b_grad ** 2

            self.W = self.W - self.alpha / np.sqrt(W_lr) * W_grad
            self.b = self.b - self.alpha / np.sqrt(b_lr) * b_grad

            print("after update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))
            now_sse = self._sse(X, y)

            print("pre_sse = %lf, now_sse = %lf)" % (pre_sse, now_sse))
            if (abs(pre_sse - now_sse) < 0.001):
                break

        # print log
        print("function is: y = %lfx + %lf" % (self.W, self.b))

    def predict(self, Xi):
        return np.matmul(Xi, self.W) + self.b


def main():
    points = genfromtxt('../datasets/gradient_descent.csv', delimiter=',')
    X = points[:, :-1]
    y = points[:, -1]

    GD = GradientDescent(iter_max=100, alpha=0.0001)
    GD.fit(X, y)

    LR = linear_model.LinearRegression()
    LR.fit(X, y)

    compare_sklearn(LR.predict, GD.predict, 1.8, 4.9)


if __name__ == '__main__':
    main()
