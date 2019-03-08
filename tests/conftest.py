import pytest
from apriori.lib import generateDocuments

@pytest.fixture(scope='session')
def data():
    baskets = generateDocuments()
    yield baskets
    print('wrapping up test fixture')
