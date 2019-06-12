import sys
sys.path.insert(0, '../utils/')
from misc import Plot
sys.path.insert(0, '../unsupervised_learning/')
from kmeans import KMeans
from sklearn import datasets

def main():
	X, y = datasets.make_blobs()

	clf = KMeans(k=3)
	# clf.fit(X)
	y_pred = clf.predict(X)

	p = Plot()
	p.plot_in_2d(X, y_pred, title="K-Means Clustering")
	p.plot_in_2d(X, y, title="Actual Clustering")

if __name__ == '__main__':
	main()