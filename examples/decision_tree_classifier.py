from __future__ import division, print_function
# compare with sklearn
from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.insert(0, '../supervised_learning/')


sys.path.insert(0, '../utils/')

from data_manipulation import train_test_split
from data_operation import accuracy_score
from decision_tree import ClassificationTree

def main():
	print ('-- Classification Tree')

	data = datasets.load_iris()
	X = data.data
	y = data.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	clf = ClassificationTree()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)

	print('Accuracy: ', accuracy)

	dt = DecisionTreeClassifier()
	dt.fit(X_train, y_train)
	y_val = dt.predict(X_test)

	acc = accuracy_score(y_test, y_val)

	print('sklearn score:', acc)


if __name__ == '__main__':
	main()