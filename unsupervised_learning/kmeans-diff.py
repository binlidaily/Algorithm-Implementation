import numpy as np


class KMeans(object):
	"""docstring for KMeans"""
	def __init__(self, k, max_iter=1000):
		super(KMeans, self).__init__()
		self.k = k
		self.max_iter = max_iter
		
	def init_k_centroids(self, X):
		n_samples, n_features = np.shape(X)
		centroids = []
		for i in range(self.k):
			centroid = X[np.random.choice(range(n_samples))]
			centroids.append(centroid)
		
		return centroids

	def _init_random_centroids(self, X):
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

		centroids = [[] for _ in range(self.k)]
		for i, cluster in enumerate(clusters):
			centroids[i] = np.mean(X[cluster], axis=0)

		return centroids

	def _calculate_centroids(self, clusters, X):
		""" Calculate new centroids as the means of the samples in each cluster  """
		n_features = np.shape(X)[1]
		centroids = np.zeros((self.k, n_features))
		for i, cluster in enumerate(clusters):
			centroid = np.mean(X[cluster], axis=0)
			centroids[i] = centroid
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

	def _closest_centroid(self, sample, centroids):
		""" Return the index of the closest centroid to the sample """
		closest_i = 0
		closest_dist = float('inf')
		for i, centroid in enumerate(centroids):
			distance = euclidean_distance(sample, centroid)
			if distance < closest_dist:
				closest_i = i
				closest_dist = distance
		return closest_i

	def _create_clusters(self, centroids, X):
		""" Assign the samples to the closest centroids to create clusters """
		n_samples = np.shape(X)[0]
		clusters = [[] for _ in range(self.k)]
		for sample_i, sample in enumerate(X):
			centroid_i = self._closest_centroid(sample, centroids)
			clusters[centroid_i].append(sample_i)
		return clusters


	def _get_cluster_labels(self, clusters, X):
		""" Classify samples as the index of their clusters """
		# One prediction for each sample
		y_pred = np.zeros(np.shape(X)[0])
		for cluster_i, cluster in enumerate(clusters):
			for sample_i in cluster:
				y_pred[sample_i] = cluster_i
		return y_pred

	def predict(self, X):

		centroids = self._init_random_centroids(X)

		for i in range(self.max_iter):
			
			# clusters = self.create_clusters(centroids, X)
			clusters = self._create_clusters(centroids, X)

			pre_centroids = centroids
			# centroids = self.update_centroids(clusters, X)
			centroids = self._calculate_centroids(clusters, X)
			
			print type(centroids), type(pre_centroids)
			diff = centroids - pre_centroids
			if not diff.any():
				break

		y_pred = self._get_cluster_labels(clusters, X)
		return y_pred


# ------------------------------------------
from sklearn import datasets
import sys
sys.path.insert(0, '../utils/')
from misc import Plot
import math
def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def main():
	X, y = datasets.make_blobs()

	clf = KMeans(k=3)
	y_pred = clf.predict(X)

	p = Plot()
	p.plot_in_2d(X, y_pred, title="K-Means Clustering")
	p.plot_in_2d(X, y, title="Actual Clustering")

if __name__ == '__main__':
	main()