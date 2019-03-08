from apriori.apriori_algorithm import Apriori
from apriori.pcy_algorithm import PCY
import pytest

testdata = [
    (2, 3, 256, [frozenset({'and', 'cat'}), frozenset({'and', 'dog'}), frozenset({'cat', 'dog'}),
            frozenset({'a', 'cat'}), frozenset({'a', 'dog'})]),
    (3, 3, 256, [frozenset({'cat', 'and', 'dog'}), frozenset({'cat', 'a', 'dog'})])
]


@pytest.mark.parametrize("k,support,bucket,expected", testdata)
def test_apriori(data, k, support, bucket, expected):
    runner = Apriori(baskets=data, k=k, support_threshold=support)
    assert set(runner.get_frequent_sets()) == set(expected)


@pytest.mark.parametrize("k,support,bucket,expected", testdata)
def test_pcy(data, k, support, bucket, expected):
    runner = PCY(baskets=data, k=k, support_threshold=support, bucket_size=bucket)
    assert set(runner.get_frequent_sets()) == set(expected)
