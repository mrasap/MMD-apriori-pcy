from clustering.cluster import Cluster
from clustering.lib import *
import sys
import os
from itertools import cycle

dirname = os.path.dirname(os.path.abspath(__file__))


class HierarchicalCluster(Cluster):

    def __init__(self, array):
        super().__init__(array)
        self.compute_centroid()

    def merge(self, other):
        """
        Merge the other clustering with this one.

        :param other: Cluster: the clustering that should be merged with this one
        """
        self.points = np.vstack((self.points, other.points))
        # TODO: add weighted means instead of recomputing
        self.centroid = mean_vector_of_matrix(self.points)


class HierarchicalClustering(object):
    """
    Create a hierarchical clustering of the datapoints.
    """

    def __init__(self, array: np.ndarray, k: int = 3):
        """
        Initialize a hierarchical clustering algorithm

        :param array: a numpy array, where rows are the data points and cols are the dimensions
        :param k: the amount of clusters that should be calculated.
        """
        self.points = array
        self.clusters = None
        self.k = k
        self.initialize_clusters()

    def initialize_clusters(self):
        """
        Initialize the starting clusters. This is a separate function to allow it to be overwritten.
        """
        self.clusters = [HierarchicalCluster(self.points[i, :]) for i in range(self.points.shape[0])]

    def plot(self):
        """
        Plot out the clusters
        """
        cycol = cycle('bgrcmk')

        for cluster in self.clusters:
            color = next(cycol)
            points = cluster.points if len(cluster.points.shape) > 1 else np.array([cluster.points])
            plt.scatter(points[:, 0], points[:, 1], c=color, marker=".")

            centroid = cluster.get_centroid()
            plt.scatter(centroid[0], centroid[1], c=color, marker="X")
        plt.show()

    def compute_clusters(self, k: int = None) -> list:
        """
        Compute the k clusters of the data.

        :return: list[Cluster]: the k clusters.
        """
        assert k is None or k <= self.points.shape[0]
        if k is not None:
            self.k = k

        while len(self.clusters) != self.k:
            # find closest pair
            min_distance = sys.maxsize
            a = b = None
            # TODO: O(nlogn) instead of O(n**2)
            # TODO: Don't recompute all distances every cycle,
            #  only update the distances of the updated clusters
            # brute force
            for i, c1 in enumerate(self.clusters):
                for c2 in self.clusters[i + 1:]:
                    distance = c1.euclidean_distance(c2.get_centroid())
                    if min_distance > distance:
                        min_distance = distance
                        (a, b) = (c1, c2)

            # merge
            self.clusters.remove(b)
            a.merge(b)

        return self.clusters


if __name__ == '__main__':
    path = os.path.join(dirname, '../data/clustering/example1.txt')
    data = import_txt_as_matrix(path)
    data = data[:100, :]
    runner = HierarchicalClustering(data)
    runner.compute_clusters(3)
    runner.plot()
