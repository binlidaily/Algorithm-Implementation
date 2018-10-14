import numpy as np
import random
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt
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

    def _derivative_W(self, X, y, rdn):
        return (np.matmul(X[rdn], self.W) + self.b - y[rdn]) * X[rdn]

    def _derivative_b(self, X, y, rdn):
        return (np.matmul(X[rdn], self.W) + self.b - y[rdn])

    def fit(self, X, y):
        m, n = np.shape(X)
        self.W = [0] * n
        self.b = 0
        for j in range(self.iter_max):
            pre_sse = self._sse(X, y)
            print("---------------------iteration: %d------------------------" % (j + 1))
            print("before update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))

            rdn = random.randint(0, m)
            print("derivative of (W, b)=(%lf, %lf)" % (self._derivative_W(X, y, rdn), self._derivative_b(X, y, rdn)))

            self.W = self.W - 1.0 * self.alpha * self._derivative_W(X, y, rdn)
            self.b = self.b - 1.0 * self.alpha * self._derivative_b(X, y, rdn)
            print("after update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))
            now_sse = self._sse(X, y)

            print("pre_sse = %lf, now_sse = %lf)" % (pre_sse, now_sse))
            if (abs(pre_sse - now_sse) < 0.001):
                break

        # print log
        print("function is: y = %lfx + %lf" % (self.W, self.b))

    def fit_travel(self, X, y):
        m, n = np.shape(X)
        self.W = [0] * n
        self.b = 0

        # 	plt.plot(X, y, 'b.')
        # plt.xlabel('House Area')
        # plt.ylabel('House Price')
        # plt.show(block = False)

        hl, = plt.plot([], [], 'r-')

        for j in range(self.iter_max):
            print("---------------------iteration: %d------------------------" % (j + 1))
            for rdn in range(m):
                pre_sse = self._sse(X, y)
                print("before update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))

                print(
                "derivative of (W, b)=(%lf, %lf)" % (self._derivative_W(X, y, rdn), self._derivative_b(X, y, rdn)))

                self.W = self.W - 1.0 * self.alpha * self._derivative_W(X, y, rdn)
                self.b = self.b - 1.0 * self.alpha * self._derivative_b(X, y, rdn)
                print("after update (W, b)=(%lf, %lf)" % (np.array(self.W[0]), self.b))
                now_sse = self._sse(X, y)

                print("pre_sse = %lf, now_sse = %lf)" % (pre_sse, now_sse))

                # y_hat = np.matmul(X, self.W) + self.b
                # # print y_hat
                # hl.set_xdata(np.append(hl.get_xdata(), X))
                # hl.set_ydata(np.append(hl.get_ydata(), y_hat))
                # plt.draw()

                if (abs(pre_sse - now_sse) < 0.001):
                    break

        # print log
        print("function is: y = %lfx + %lf" % (self.W, self.b))

    def predict(self, Xi):
        return np.matmul(Xi, self.W) + self.b

    # def plot_data(X, y, xlabel, ylabel):
    # 	plt.scatter(X, y)
    # 	plt.xlabel(xlabel)
    # 	plt.ylabel(ylabel)
    # 	plt.show()


# hl, = plt.plot([], [])

# def update_line(hl, new_data):
#     hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
#     hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
#     plt.draw()

def main():
    points = genfromtxt('../datasets/gradient_descent.csv', delimiter=',')
    X = points[:, :-1]
    y = points[:, -1]

    # plot_data(X, y, 'House Area', 'House Price')

    GD = GradientDescent(iter_max=100, alpha=0.0001)
    GD.fit_travel(X, y)

    LR = linear_model.LinearRegression()
    LR.fit(X, y)

    compare_sklearn(LR.predict, GD.predict, 1.8, 4.9)


if __name__ == '__main__':
    main()
