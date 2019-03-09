from collections import OrderedDict
from pagerank import lib
import numpy as np


def test_import_data():
    expected = OrderedDict([('C', {'A'}), ('B', {'D', 'A'}), ('D', {'B', 'C'}), ('A', {'D', 'C'})])
    result = lib.import_csv_as_adjacency_dict('data/pagerank/example1.csv')

    assert expected.keys() == result.keys()

    for key in result.keys():
        assert expected[key] == result[key]


def test_adjacency_matrix(adjacency_dict_pagerank):
    expected = np.array([[0.5, 0.5, 1., 0.5],
                         [0.5, 0.5, 0., 0.5],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.]])
    result = lib.construct_adjacency_matrix_from_dict(adjacency_dict_pagerank)

    assert np.allclose(expected, result)
