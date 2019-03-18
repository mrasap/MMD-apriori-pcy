import pytest

from clustering.hierarchical_cluster_algorithm import *


def test_hierarchical_cluster_init(cluster_runner):
    result = cluster_runner.clusters

    assert len(result) == 3
    for cluster in result:
        assert isinstance(cluster, Cluster)
        assert cluster.points.shape == (2,)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_hierarchical_cluster_compute(cluster_runner, k):
    # TODO: currently only checking vague terms, this should check an actual example
    result = cluster_runner.compute_clusters(k)

    assert len(result) == k
