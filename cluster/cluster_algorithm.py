from cluster.lib import *
import sys
import os
from itertools import cycle

dirname = os.path.dirname(os.path.abspath(__file__))


class Cluster(object):
    """
    Represents a cluster of data points and its corresponding centroid
    """

    def __init__(self, array):
        self.points = array
        self.centroid = None
        self.compute_centroid()

    def compute_centroid(self):
        """
        Calculate the centroid of the data
        """
        self.centroid = mean_vector_of_matrix(self.points)

    def euclidean_distance(self, other) -> int:
        """
        Calculate euclidean distance between the centroids of the two clusters

        :param other: the other cluster
        :return: euclidean distance
        """
        return euclidean_distance_between_vectors(self.centroid, other.centroid)

    def merge(self, other):
        """
        Merge the other cluster with this one.

        :param other: Cluster: the cluster that should be merged with this one
        """
        self.points = np.vstack((self.points, other.points))
        # TODO: add weighted means instead of recomputing
        self.centroid = mean_vector_of_matrix(self.points)


class HierarchicalClustering(object):
    """
    Create a hierarchical clustering of the datapoints.
    """

    def __init__(self, array: np.ndarray):
        self.points = array
        self.clusters = [Cluster(self.points[i][:]) for i in range(self.points.shape[0])]

    def plot(self):
        """
        Plot out the clusters
        """
        cycol = cycle('bgrcmk')

        for cluster in self.clusters:
            color = next(cycol)
            points = cluster.points if len(cluster.points.shape) > 1 else np.array([cluster.points])
            plt.scatter(points[:, 0], points[:, 1], c=color, marker=".")

            centroid = cluster.centroid
            plt.scatter(centroid[0], centroid[1], c=color, marker="X")
        plt.show()

    def compute_clusters(self, k: int) -> list:
        """
        Compute the k clusters of the data.

        :param k: the amount of clusters that should be calculated.
        :return: list[Cluster]: the k clusters.
        """
        assert k <= self.points.shape[0]

        while len(self.clusters) != k:
            # find closest pair
            min_distance = sys.maxsize
            a = b = None
            # TODO: O(nlogn) instead of O(n**2)
            # TODO: Don't recompute all distances every cycle,
            #  only update the distances of the updated clusters
            # brute force
            for i, c1 in enumerate(self.clusters):
                for c2 in self.clusters[i + 1:]:
                    distance = c1.euclidean_distance(c2)
                    if min_distance > distance:
                        min_distance = distance
                        (a, b) = (c1, c2)

            # merge
            self.clusters.remove(b)
            a.merge(b)

        return self.clusters


class KMeansClustering(HierarchicalClustering):

    def compute_clusters(self, k: int):
        pass


if __name__ == '__main__':
    path = os.path.join(dirname, '../data/cluster/example1.txt')
    data = import_txt_as_matrix(path)
    # data = data[:100, :]
    runner = HierarchicalClustering(data)
    runner.compute_clusters(3)
    runner.plot()
