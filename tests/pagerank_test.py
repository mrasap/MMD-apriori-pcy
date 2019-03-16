from collections import OrderedDict
from pagerank import lib
import numpy as np
import pytest
import os

dirname = os.path.dirname(os.path.abspath(__file__))
example1 = os.path.join(dirname, '../data/pagerank/example1.csv')


def test_import_data():
    expected = OrderedDict([('C', {'A'}), ('B', {'D', 'A'}), ('D', {'B', 'C'}), ('A', {'D', 'C'})])
    result = lib.import_csv_as_adjacency_dict(example1)

    assert expected.keys() == result.keys()

    for key in result.keys():
        assert expected[key] == result[key]


def test_adjacency_matrix(adjacency_dict_pagerank):
    expected = np.array([[0., 0.5, 1., 0.],
                         [0., 0., 0., 0.5],
                         [0.5, 0., 0., 0.5],
                         [0.5, 0.5, 0., 0.]])
    result = lib.construct_adjacency_matrix_from_dict(adjacency_dict_pagerank)

    assert np.allclose(expected, result)


def test_adjacency_matrix_is_stochastic(adjacency_dict_pagerank):
    result = lib.construct_adjacency_matrix_from_dict(adjacency_dict_pagerank)

    for i in range(result.shape[1]):
        assert np.isclose(np.sum(result[:, i]), 1)


@pytest.mark.parametrize('k', [10, 20, 100])
def test_pagerank_is_stochastic(pagerank_runner, k):
    result = pagerank_runner.calculate_page_rank(k)

    assert np.isclose(np.sum(result), 1)


@pytest.mark.parametrize('k', [10, 20, 100])
def test_pagerank_with_teleport_is_stochastic(pagerank_runner, k):
    result = pagerank_runner.calculate_page_rank_with_teleport(k)

    assert np.isclose(np.sum(result), 1)
