from clustering.lib import mean_vector_of_matrix, euclidean_distance_between_vectors


class Cluster(object):
    """
    Represents a clustering of data points and its corresponding centroid
    """

    def __init__(self, array):
        self.points = array
        self.centroid = None

    def compute_centroid(self):
        """
        Calculate the centroid of the data
        """
        self.centroid = mean_vector_of_matrix(self.points)

    def get_centroid(self):
        """
        Safety check to ensure that the centroid is computed.

        :return: the centroid
        """
        if self.centroid is None:
            self.compute_centroid()
        return self.centroid

    def euclidean_distance(self, other) -> int:
        """
        Calculate euclidean distance between the centroids of the two clusters

        :param other: the centroid of the other clustering
        :return: euclidean distance
        """
        return euclidean_distance_between_vectors(self.get_centroid(), other)
