import pytest
from apriori.lib import generate_documents
from pagerank.lib import *
from pagerank.pagerank_algorithm import *
import os

dirname = os.path.dirname(os.path.abspath(__file__))
example1 = os.path.join(dirname, '../data/pagerank/example1.csv')


@pytest.fixture(scope='module')
def data_apriori():
    baskets = generate_documents()
    yield baskets
    # nothing to close, but I think its better practice to yield rather than to return in case
    # you really need to close something


@pytest.fixture(scope='module')
def adjacency_dict_pagerank():
    data = import_csv_as_adjacency_dict(example1)
    yield data


@pytest.fixture(scope='module')
def pagerank_runner():
    data = import_csv_as_adjacency_matrix(example1)
    runner = PageRank(data)
    yield runner
