import numpy as np
from sklearn import datasets
import sys
sys.path.insert(0, '../utils/')
from misc import Plot

class KMeans(object):
	"""docstring for KMeans"""
	def __init__(self, k, max_iter=1000):
		super(KMeans, self).__init__()
		self.k = k
		self.centers = []
		self.max_iter = max_iter
		
	def init_k_centers(self, X):
		n_samples, n_features = np.shape(X)
		for i in range(self.k):
			centroid = X[np.random.choice(range(n_samples))]
			self.centers.append(centroid)

	def euclidean_distance(self, a, b):
		n_features = len(a)
		dist = 0

		for i in range(n_features):
			dist += (a[i] - b[i]) ** 2

		return dist



	def update_centers(self, clusters):
		idx = []
		for i in range(self.k):
			if clusters[i] == []:
				idx.append(i)


		while len(idx) > 0:
			pop_idx = None
			while True:
				sub = np.random.randint(self.k)

				if sub not in idx:

					if len(clusters[sub]) > 1:
						pop_idx = sub
						break


			clusters[idx[-1]].append(clusters[pop_idx].pop())
			idx.pop()


		for i in range(self.k):
			self.centers[i] = np.sum(clusters[i], 0)

		

	def fit(self, X):
		n_samples, n_features = np.shape(X)
		self.init_k_centers(X)

		for i in range(self.max_iter):
			clusters = []
			for _ in range(self.k):
				clusters.append([])
			min_dist = float('inf')
			for m in range(n_samples):
				idx = -1
				for n in range(self.k):
					dist = self.euclidean_distance(X[m], self.centers[n])

					if min_dist > dist:
						min_dist = dist
						idx = n

				clusters[idx].append(X[m])
			if i % 100 == 0:
				print clusters
			self.update_centers(clusters)
			if i % 100 == 0:
				print clusters

	def predict(self, X):
		n_samples, _ = np.shape(X)
		Y = []
		for i in range(n_samples):
			min_dist = float('inf')
			idx = None
			for j in range(self.k):
				dist = self.euclidean_distance(X[i], self.centers[j])
				if min_dist > dist:
					min_dist = dist
					idx = j
			Y.append(idx)
		return Y

def main():
	X, y = datasets.make_blobs()

	clf = KMeans(k=3)
	clf.fit(X)
	y_pred = clf.predict(X)

	p = Plot()
	p.plot_in_2d(X, y_pred, title="K-Means Clustering")
	p.plot_in_2d(X, y, title="Actual Clustering")

if __name__ == '__main__':
	main()