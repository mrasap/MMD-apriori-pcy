from cluster.cluster_algorithm import *
import os
import pytest

dirname = os.path.dirname(os.path.abspath(__file__))

example1 = os.path.join(dirname, '../../data/cluster/test_example1.txt')
example1_expected = np.array([
        [1.9957, 2.7363],
        [1.5573, 2.4469],
        [2.5028, 0.7232]
    ])

a = np.array([3, 3, 4])
b = np.array([1, 9, 3])
m = np.vstack([a, b]).T
e = 0.01


def test_mean_vector_of_vectors():
    result = mean_vector_of_vectors(a, b)

    assert np.allclose(result, np.array([3.33333333, 4.33333333]))


@pytest.mark.parametrize('expected,data', [
    (np.array([3.33333333, 4.33333333]), m),
    (a, a),
    (b, b)
])
def test_mean_vector_of_matrix(expected, data):
    result = mean_vector_of_matrix(data)

    assert np.allclose(result, expected)


def test_euclidean_distance_between_vectors():
    result = euclidean_distance_between_vectors(a, b)

    assert 6.403 - e < result
    assert result < 6.403 + e


def test_import_txt_as_matrix():
    result = import_txt_as_matrix(example1)

    assert result.shape == (3, 2)
    assert np.allclose(result, example1_expected)
