import pytest
from apriori.lib import generate_documents
from pagerank.lib import *
from pagerank.pagerank_algorithm import *


@pytest.fixture(scope='module')
def data_apriori():
    baskets = generate_documents()
    yield baskets
    # nothing to close, but I think its better practice to yield rather than to return in case
    # you really need to close something


@pytest.fixture(scope='module')
def adjacency_dict_pagerank():
    data = import_csv_as_adjacency_dict('data/pagerank/example1.csv')
    yield data


@pytest.fixture(scope='module')
def pagerank_runner():
    data = import_csv_as_adjacency_matrix('data/pagerank/example1.csv')
    runner = PageRank(data)
    yield runner
