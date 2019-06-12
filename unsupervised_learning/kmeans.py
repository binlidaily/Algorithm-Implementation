import numpy as np


class KMeans(object):
	"""docstring for KMeans"""
	def __init__(self, k, max_iter=1000):
		super(KMeans, self).__init__()
		self.k = k
		self.max_iter = max_iter
		
	def init_k_centroids(self, X):
		""" Initialize the centroids as k random samples of X"""
		n_samples, n_features = np.shape(X)
		centroids = np.zeros((self.k, n_features))
		for i in range(self.k):
			centroid = X[np.random.choice(range(n_samples))]
			centroids[i] = centroid
		return centroids

	def euclidean_distance(self, a, b):
		n_features = len(a)
		dist = 0

		for i in range(n_features):
			dist += (a[i] - b[i]) ** 2

		return dist


	def update_centroids(self, clusters, X):
		# heuristic method
		# here if the cluster is empty, pick a sample from another cluster randomly.
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

		centroids = np.zeros((self.k, np.shape(X)[1]))
		for i, cluster in enumerate(clusters):
			centroids[i] = np.mean(X[cluster], axis=0)

		return centroids

	def create_clusters(self, centroids, X):
		clusters = []
		for i in range(self.k):
			clusters.append([])

		for sample_i, sample in enumerate(X):
			min_dist = float('inf')
			idx = None
			for n in range(self.k):
				dist = self.euclidean_distance(sample, centroids[n])

				if min_dist > dist:
					min_dist = dist
					idx = n

			clusters[idx].append(sample_i)
		return clusters

	def get_cluster_label(self, clusters, X):
		y_pred = np.zeros(np.shape(X)[0])
		for cluster_i, cluster in enumerate(clusters):
			for sample_i in cluster:
				y_pred[sample_i] = cluster_i
		return y_pred

	def predict(self, X):
		n_samples, n_features = np.shape(X)
		centroids = self.init_k_centroids(X)

		for i in range(self.max_iter):
			
			clusters = self.create_clusters(centroids, X)
			pre_centroids = centroids
			centroids = self.update_centroids(clusters, X)

			diff = centroids - pre_centroids
			if not diff.any():
				break

		y_pred = self.get_cluster_label(clusters, X)
		return y_pred


# ------------------------------------------
from sklearn import datasets
import sys
sys.path.insert(0, '../utils/')
from misc import Plot

def main():
	X, y = datasets.make_blobs()

	clf = KMeans(k=10)
	y_pred = clf.predict(X)

	p = Plot()
	p.plot_in_2d(X, y_pred, title="K-Means Clustering")
	p.plot_in_2d(X, y, title="Actual Clustering")

if __name__ == '__main__':
	main()