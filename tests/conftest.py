import pytest
from apriori.lib import generate_documents
from pagerank.lib import import_csv_as_adjacency_dict


@pytest.fixture(scope='module')
def data_apriori():
    baskets = generate_documents()
    yield baskets
    print('wrapping up test fixture')


@pytest.fixture(scope='module')
def adjacency_dict_pagerank():
    data = import_csv_as_adjacency_dict('data/pagerank/example1.csv')
    yield data
    print('wrapping up test fixture')
