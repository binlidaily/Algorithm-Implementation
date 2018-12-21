import numpy as np
import sys

sys.path.insert(0, '../until')

from data_manipulation import divide_on_feature
from data_operation import calculate_entropy, calculate_variance


class DecisionNode(object):
    """docstring for DecesionNode"""

    def __init__(self, feature_i=None, threshold=None, value=None,
                 true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        # the leaf node has the value variable
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(DecisionNode):
    """docstring for DecisionTree"""

    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float('inf')):
        # The root node at start
        self.root = None
        # The maximum number of depth for splitting
        self.max_depth = max_depth
        # The  minimum number of samples to justify the tree
        self.min_samples_split = min_samples_split
        # Check if y is one-hot encoded
        self.one_dim = None
        # Different Decision Tree has a different impurity calculation method
        self._impurity_calculation = None
        # Different Decision Tree has a different leaf value calculation method
        self._leaf_value_calculation = None
        # Set the minimum impurity, we build the tree when the impurity we get is greater than this minima
        self.min_impurity = min_impurity

    def _build_tree(self, X, y, current_depth=0):
        # print('Start building the tree')
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        largest_impurity = 0

        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)

        if current_depth <= self.max_depth and n_samples >= self.min_samples_split:
            # for-loop: Calculate the impurity for each feature
            for feature_i in range(n_features):
                # Get all unique values in curent feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    # comfirm that the divide is done, if there is no subtree here, we just skip it
                    if len(Xy1) > 0 and len(Xy2) > 0:

                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate the current impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {'feature_i': feature_i, 'threshold': threshold}
                            best_sets = {
                                'leftX': Xy1[:, :n_features],
                                'lefty': Xy1[:, n_features:],
                                'rightX': Xy2[:, :n_features],
                                'righty': Xy2[:, n_features:]
                            }

        # we don't need exactly stop the process until we cannot get any impurity impoving, we just set a min_impurity
        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            # true_branch is just 'left' branch
            true_branch = self._build_tree(best_sets['leftX'], best_sets['lefty'], current_depth + 1)
            false_branch = self._build_tree(best_sets['rightX'], best_sets['righty'], current_depth + 1)
            return DecisionNode(feature_i=best_criteria['feature_i'], threshold=best_criteria['threshold'],
                                true_branch=true_branch, false_branch=false_branch)

        # we should know where return the result, including the leaf node and the decision node!

        # we are at leaf node, so we determine the value
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def fit(self, X, y):
        # To use len(np.shape(y)) is to find the dimension of the array, if the size of list is (150,) the dimension
        # is 1 if the size of list is (150, 2) or (150, 1) the dimension is 2 if the size of list is (150, 2,
        # 1) the dimension is 3

        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        # current node
        feature_value = x[tree.feature_i]

        # left or right?
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.true_branch:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=' '):
        pass


class ClassificationTree(DecisionTree):
    """docstring for DecisionTreeClassifier"""

    def _calculate_information_gain(self, y, y1, y2):
        entropy = calculate_entropy(y)
        p = len(y1) / len(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        unique_labels = np.unique(y)

        for label in unique_labels:
            cnt = len(y[y == label])
            if cnt > max_count:
                max_count = cnt
                most_common = label
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


class RegressionTree(DecisionTree):
    """docstring for RegressionTree"""
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)  

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]
        
    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)