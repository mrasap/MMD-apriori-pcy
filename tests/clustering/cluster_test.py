import pytest

from clustering.hierarchical_cluster_algorithm import *
from clustering.kmeans_cluster_algorithm import *


def test_hierarchical_cluster_init(hierarchical_cluster_runner):
    result = hierarchical_cluster_runner.clusters

    assert len(result) == 3
    for cluster in result:
        assert isinstance(cluster, HierarchicalCluster)
        assert cluster.points.shape == (2,)
        assert cluster.get_centroid().shape == (2,)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_hierarchical_cluster_compute(hierarchical_cluster_runner, k):
    # TODO: currently only checking vague terms, this should check an actual example
    result = hierarchical_cluster_runner.compute_clusters(k)

    assert len(result) == k


def test_kmeans_cluster_init(kmeans_cluster_runner):
    result = kmeans_cluster_runner.clusters

    assert len(result) == 3
    for cluster in result:
        assert isinstance(cluster, KMeansCluster)
        assert cluster.points.shape == (2,)
        assert cluster.get_centroid().shape == (2,)


@pytest.mark.parametrize('init, add, expected', [
    ([3, 4], [4, 4], [[3, 4],
                      [4, 4]]),
    ([[3, 4],
      [4, 3]], [4, 4], [[3, 4],
                        [4, 3],
                        [4, 4]]),
])
def test_kmeans_cluster_add(init, add, expected):
    init = np.array(init)
    add = np.array(add)
    expected = np.array(expected)
    cluster = KMeansCluster(init)
    cluster.add(add)

    assert np.array_equal(cluster.points, expected)


def test_kmeans_cluster_add_multiple():
    init = np.array([3, 4])
    add = np.array([4, 4])
    expected = np.array([[3, 4],
                         [4, 4],
                         [4, 4]])
    cluster = KMeansCluster(init)
    cluster.add(add)
    cluster.add(add)

    assert np.array_equal(cluster.points, expected)


def test_kmeans_cluster_reset_then_add():
    init = np.array([3, 4])
    add = np.array([4, 4])
    expected = np.array([4, 4])
    cluster = KMeansCluster(init)
    cluster.reset()
    cluster.add(add)

    assert np.array_equal(cluster.points, expected)
