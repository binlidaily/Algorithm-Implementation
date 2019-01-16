from __future__ import division, print_function
import numpy as np
import progressbar

import sys
sys.path.append(r'../')

from deep_learning.loss_functions import SquareLoss, CrossEntropy
from supervised_learning.decision_tree import RegressionTree
from utils.misc import bar_widgets
from utils.data_manipulation import to_categorical

class GradientBoosting(object):
	"""docstring for GradientBoosting"""
	def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, regression):
		super(GradientBoosting, self).__init__()
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.min_samples_split = min_samples_split
		self.min_impurity = min_impurity
		self.max_depth = max_depth
		self.regression = regression
		self.bar = progressbar.ProgressBar(widgets=bar_widgets)

		# Square loss for regression
        # Log loss for classification
		self.loss = SquareLoss()
		if not self.regression:
			self.loss = CrossEntropy()
        # Initailize the trees
		self.trees = []
		for _ in range(n_estimators):
			tree = RegressionTree(
						min_samples_split=self.min_samples_split,
						min_impurity=min_impurity,
						max_depth=self.max_depth)
			self.trees.append(tree)

	def fit(self, X, y):
		# r_im is also the y_pred for the current tree
		r_im = np.full(np.shape(y), np.mean(y, axis=0))

		for i in self.bar(range(self.n_estimators)):
			gradient = self.loss.gradient(y, r_im)
			self.trees[i].fit(X, gradient)
			update = self.trees[i].predict(X)

			# update the residual
			r_im -= np.multiply(self.learning_rate, update)

	def predict(self, X):
		y_pred = np.array([])
		for tree in self.trees:
			update = tree.predict(X)
			update = np.multiply(self.learning_rate, update)
			y_pred = -update if not y_pred.any() else y_pred - update

		if not self.regression:
			y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
			y_pred = np.argmax(y_pred, axis=1)

		return y_pred

class GradientBoostingRegressor(GradientBoosting):
	def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
				min_var_red=1e-7, max_depth=4, debug=False):
		super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, 
					learning_rate=learning_rate, 
					min_samples_split=min_samples_split, 
					min_impurity=min_var_red,
					max_depth=max_depth,
					regression=True)

class GradientBoostingClassifier(GradientBoosting):
	def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
				min_info_gain=1e-7, max_depth=2, debug=False):
		super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, 
					learning_rate=learning_rate, 
					min_samples_split=min_samples_split, 
					min_impurity=min_info_gain,
					max_depth=max_depth,
					regression=False)

	def fit(self, X, y):
		y = to_categorical(y)
		super(GradientBoostingClassifier, self).fit(X, y)

