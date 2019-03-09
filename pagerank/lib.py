from collections import OrderedDict
import csv
import itertools
import numpy as np


def import_csv_as_adjacency_dict(path: str) -> OrderedDict:
    """
    Import the csv file and return an adjacency dict of the graph.

    Expects the csv file to be of the format:
    'node from, node to, weight'
    where weight is currently set to 1

    :param path: the path to the data file
    :return: OrderedDict: a dict of all nodes as keys, and a set of nodes that are are
    discovered by the outgoing edges of the key as values.
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        edges = [row[0:2] for row in reader]

        # this operation is equivalent to flatten.distinct in scala
        set_of_nodes = set(itertools.chain.from_iterable(edges))

        data = OrderedDict({node: set() for node in set_of_nodes})

        for edge in edges:
            data[edge[0]].add(edge[1])

        return data


def construct_adjacency_matrix_from_dict(data: OrderedDict) -> np.ndarray:
    """
    See function name.

    :param data: OrderedDict: adjacency dict
    :return: np.ndarray: adjacency matrix
    """
    arr = np.zeros((len(data), len(data)))

    for i, (node_from, lst) in enumerate(data.items()):
        n_outgoing_edges = len(lst)
        for j, node_to in enumerate(lst):
            arr[j][i] = 1 / n_outgoing_edges

    return arr


if __name__ == '__main__':
    lst = import_csv_as_adjacency_dict('../data/pagerank/example1.csv')
    print(lst)
    matrix = construct_adjacency_matrix_from_dict(lst)
    print(matrix)
