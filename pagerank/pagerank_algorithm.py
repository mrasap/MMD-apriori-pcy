import numpy as np
from pagerank import lib


class PageRank:

    def __init__(self, matrix: np.ndarray):
        assert matrix.shape[0] == matrix.shape[1]

        self.matrix = matrix
        self.n = matrix.shape[0]

    def generate_starting_surfer(self) -> np.array:
        """
        Generate a random surfer that will be used as a starting point to find the eigenvector.

        :return: np.array: stochastic vector of size of all nodes.
        """
        return np.array([[1 / self.n] for _ in range(self.n)])

    def calculate_page_rank(self, k: int) -> np.array:
        """
        Apply power iterations to approximate the eigenvector of the matrix.

        :param k: int: the amount of iterations
        :return: np.array: the approximation of the eigenvector. The values represent the
        stochastic rank of the particular node in the graph.
        """
        vector = self.generate_starting_surfer()
        for _ in range(k):
            vector = self.matrix.dot(vector)

        return vector

    def calculate_page_rank_with_teleport(self, k: int, b: int = 0.85) -> np.array:
        """
        Avoid spider traps and dead ends with teleports

        :param k: int: the amount of iterations
        :return: np.array: the approximation of the eigenvector. The values represent the
        stochastic rank of the particular node in the graph.
        """
        random_teleport = ((1 - b)/self.n)*np.ones((self.n, 1))
        vector = self.generate_starting_surfer()
        for _ in range(k):
            vector = b*self.matrix.dot(vector) + random_teleport

        return vector


if __name__ == '__main__':
    runner = PageRank(lib.import_csv_as_adjacency_matrix('../data/pagerank/example1.csv'))
    print(runner.calculate_page_rank(10))
    print(runner.calculate_page_rank_with_teleport(10))

