from clustering.lib import *
import os
from clustering.hierarchical_cluster_algorithm import HierarchicalClustering
from clustering.cluster import Cluster

dirname = os.path.dirname(os.path.abspath(__file__))


class KMeansCluster(Cluster):
    """
    Represents a clustering of data points that are mutating
    """

    def reset(self):
        self.points = np.array([])
        self.centroid = None

    def add(self, newPoint):
        """
        Add a new point to the clustering.
        :param newPoint: A new point to be added to the clustering
        """
        assert newPoint.shape[0] == self.points.shape[1]

        self.points = np.vstack(self.points, newPoint)


class KMeansClustering(HierarchicalClustering):

    def compute_clusters(self, k: int):
        pass


if __name__ == '__main__':
    path = os.path.join(dirname, '../data/clustering/example1.txt')
    data = import_txt_as_matrix(path)
    # data = data[:100, :]
    runner = KMeansClustering(data)
    runner.compute_clusters(3)
    runner.plot()
