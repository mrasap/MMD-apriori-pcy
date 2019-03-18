from clustering.lib import *
from clustering.config import *
import os
import sys
import random
from clustering.hierarchical_cluster_algorithm import HierarchicalClustering
from clustering.cluster import Cluster

dirname = os.path.dirname(os.path.abspath(__file__))


class KMeansCluster(Cluster):
    """
    Represents a clustering of data points that are mutating
    """

    def __init__(self, array):
        super().__init__(array)
        self.compute_centroid()

    def reset(self):
        """
        Reset the cluster.
        """
        self.points = None

    def add(self, new_point: np.array):
        """
        Add a new point to the clustering.

        :param new_point: A new point to be added to the clustering
        """
        self.points = new_point if self.points is None else np.vstack((self.points, new_point))

    def compute_centroid(self) -> float:
        """
        Overwritten to compute the difference in the new and old centroid

        :return: the euclidean distance between the old and the new centroid
        """
        new_centroid = mean_vector_of_matrix(self.points)

        if self.centroid is None:
            diff = new_centroid
        else:
            diff = euclidean_distance_between_vectors(self.centroid, new_centroid)

        self.centroid = new_centroid
        return diff


class KMeansClustering(HierarchicalClustering):

    def __init__(self, array: np.ndarray):
        super().__init__(array)

    def initialize_clusters(self):
        self.initialize_random_clusters()

    def initialize_random_clusters(self):
        """
        Just take random starting clusters
        # TODO: Sample a subset with hierarchical clustering to generate better initial clusters
        """
        n = self.points.shape[0] - 1
        self.clusters = [KMeansCluster(self.points[random.randint(0, n), :]) for _ in range(self.k)]

    def compute_clusters(self, k: int = None):
        diff = sys.maxsize
        i = 0

        while diff >= KMEANS_DIFF_CUTOFF:
            print('i:', i, ', diff:', diff)
            self.reset_clusters()

            self.redistribute_datapoints()

            diff = self.compute_centroids()

            i += 1
            print('after diff:', diff)

    def reset_clusters(self):
        """
        Reset the clusters.
        """
        for cluster in self.clusters:
            cluster.reset()

    def redistribute_datapoints(self):
        """
        Redistribute all datapoints to the closest centroids
        """
        for point in self.points:
            min_distance, min_cluster = sys.maxsize, None
            for cluster in self.clusters:
                dist = cluster.euclidean_distance(point)
                if dist < min_distance:
                    min_distance, min_cluster = dist, cluster
            min_cluster.add(point)

    def compute_centroids(self) -> int:
        """
        Compute the centroids of the clusters.

        :return: the difference in distance between the previous and current centroids
        """
        diff = 0
        for cluster in self.clusters:
            diff += cluster.compute_centroid()

        return diff


if __name__ == '__main__':
    path = os.path.join(dirname, '../data/clustering/example1.txt')
    data = import_txt_as_matrix(path)
    # data = data[:100, :]
    runner = KMeansClustering(data)
    runner.compute_clusters(3)
    runner.plot()
