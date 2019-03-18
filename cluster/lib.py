import numpy as np
import matplotlib.pyplot as plt


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def import_txt_as_matrix(path: str) -> np.ndarray:
    """
    Import a dataset and transform it into a matrix.

    :param path: the path to the file
    :return: np.ndarray: matrix where cols represent the dimensions and rows represent the points
    """
    with open(path, 'r') as file:
        points = file.read().split('\n')
        coords = [[float(coordinate) for coordinate in point.split(' ')] for point in points]
        return np.array(coords)


def mean_vector_of_vectors(a: np.array, b: np.array) -> np.array:
    """
    Calculate the mean of the two vectors.
    """
    assert len(a.shape) == len(b.shape) == 1
    assert a.shape[0] == b.shape[0]

    return mean_vector_of_matrix(np.vstack([a, b]).T)


def mean_vector_of_matrix(m: np.ndarray) -> np.array:
    """
    Calculate the mean of each column in the matrix

    :param m: np.ndarray: the matrix, where a row is a point and a column is a dimension
    :return: np.array: a vector with the mean of each column of the matrix
    """
    if len(m.shape) == 1:
        return m
    else:
        return m.mean(axis=0)


def euclidean_distance_between_vectors(a: np.array, b: np.array) -> int:
    """
    Calculate the euclidean distance between the two vectors.

    Note, for matrices the implementation by Nico Schlömer is best:
    https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

    :param a: np.array: vector 1
    :param b: np.array: vector 2
    :return: the euclidean distance between the two vectors
    """
    assert a.shape == b.shape

    return np.sqrt(np.sum((a - b) ** 2))
