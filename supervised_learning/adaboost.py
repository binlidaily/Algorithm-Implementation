from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import sys
sys.path.append(r'../')

from utils.data_manipulation import train_test_split
from utils.data_operation import accuracy_score
from utils.misc import Plot

# Decision stump used as weak classifier in this impl. of Adaboost
class DecisionStump():
	def __init__(self):
		# Determines if sample shall be classified as -1 or 1 given threshold
		self.polarity = 1
		# The index of the feature used to make classification
		self.feature_index = None
		# The threshold is against with the feataure should be measured
		self.threshold = None
		self.alpha = None


class Adaboost():
	def __init__(self, n_clf):
		self.n_clf = n_clf

	def fit(self, X, y):
		n_samples, n_features = np.shape(X)

		# initialize the weight to 1/N at first
		w = np.full(n_samples, (1.0 / n_samples))

		self.clfs = []

		for _ in range(self.n_clf):
			clf = DecisionStump()
			min_error = float('inf')

			for feature_i in range(n_features):
				# all feature values in one specific feature
				feature_values = np.expand_dims(X[:, feature_i], axis=1)
				unique_values = np.unique(feature_values)

				for threshold in unique_values:
					p = 1

					prediction = np.ones(np.shape(y))
					prediction[X[:, feature_i] < threshold] = -1

					error = sum(w[y != prediction])

					if error > 0.5:
						error = 1 - error
						p = -1

					if error < min_error:
						clf.polarity = p
						clf.feature_index = feature_i
						clf.threshold = threshold
						min_error = error

			clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
			predictions = np.ones(np.shape(y))
			negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
			predictions[negative_idx] = -1

			w *= np.exp(-clf.alpha * y * predictions)
			w /= np.sum(w)
			self.clfs.append(clf)


	def predict(self, X):
		n_samples, n_features = np.shape(X)
		y_pred = np.zeros((n_samples, 1))

		for clf in self.clfs:
			predictions = np.ones(np.shape(y_pred))
			negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
			predictions[negative_idx] = -1

			y_pred += clf.alpha * predictions

		# Return sign of prediction sum
		y_pred = np.sign(y_pred).flatten()
		return y_pred

def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # Change labels to {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Adaboost", accuracy=accuracy)


if __name__ == "__main__":
    main()