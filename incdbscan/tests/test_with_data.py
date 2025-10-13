import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles

from incdbscan import IncrementalDBSCAN
from testutils import (
    are_lists_isomorphic,
    read_handl_data
)


@pytest.mark.slow
def test_same_results_as_sklearn_dbscan():
    EPS = 1
    MIN_PTS = 5

    data = read_handl_data()
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan_1 = incdbscan.insert(data).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_1)

    labels_incdbscan_2 = \
        incdbscan.insert(data).delete(data).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_2)

    np.random.seed(123)
    noise = np.random.uniform(-14, 14, (1000, 2))
    labels_incdbscan_3 = \
        incdbscan.insert(noise).delete(noise).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_3)


@pytest.mark.parametrize("eps,min_pts", [
    (0.5, 3),
    (1.0, 3),
    (1.0, 5),
    (1.5, 4),
    (2.0, 5),
])
def test_sklearn_consistency_with_different_parameters(eps, min_pts):
    """Test consistency with various eps and min_pts combinations"""
    data = read_handl_data()

    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


@pytest.mark.parametrize("n_samples,n_centers,cluster_std", [
    (300, 3, 0.5),
    (500, 5, 1.0),
    (1000, 10, 0.8),
])
def test_sklearn_consistency_with_blob_data(n_samples, n_centers, cluster_std):
    """Test consistency with synthetic blob datasets"""
    EPS = 2.0
    MIN_PTS = 5

    data, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=2,
        cluster_std=cluster_std,
        random_state=42
    )

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_moons_data():
    """Test consistency with moon-shaped data"""
    EPS = 0.3
    MIN_PTS = 5

    data, _ = make_moons(n_samples=500, noise=0.05, random_state=42)

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_circles_data():
    """Test consistency with circular data"""
    EPS = 0.2
    MIN_PTS = 5

    data, _ = make_circles(n_samples=400, noise=0.05,
                           factor=0.5, random_state=42)

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_incremental_insertion():
    """Test consistency when data is inserted incrementally in batches"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    # Insert data in 4 batches
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    batch_size = len(data) // 4
    for i in range(4):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < 3 else len(data)
        incdbscan.insert(data[start_idx:end_idx])

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_partial_deletion():
    """Test consistency after partial deletion of data"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()

    # Split data into two parts
    split_idx = len(data) // 2
    data_part1 = data[:split_idx]
    data_part2 = data[split_idx:]

    # Expected result: only first part
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data_part1)

    # Insert all, then delete second part
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data).delete(data_part2)
    labels_incdbscan = incdbscan.get_cluster_labels(data_part1)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_random_insertion_deletion():
    """Test consistency with random insertion and deletion order"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    # Shuffle data for random insertion
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data[shuffled_indices]

    # Insert in random order
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(shuffled_data)

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_higher_dimensions():
    """Test consistency with higher dimensional data"""
    EPS = 3.0
    MIN_PTS = 5

    # Generate 5-dimensional blob data
    data, _ = make_blobs(
        n_samples=500,
        centers=5,
        n_features=5,
        cluster_std=1.0,
        random_state=42
    )

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_sparse_data():
    """Test consistency with sparse data (mostly noise)"""
    EPS = 0.5
    MIN_PTS = 3

    np.random.seed(42)
    # Generate sparse random data
    data = np.random.uniform(-10, 10, (200, 2))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_multiple_insert_delete_cycles():
    """Test consistency after multiple insertion and deletion cycles"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)

    # Multiple insert-delete cycles with random noise
    np.random.seed(42)
    noise_batches = []
    for _ in range(3):
        noise = np.random.uniform(-14, 14, (500, 2))
        noise_batches.append(noise)
        incdbscan.insert(noise)

    incdbscan.insert(data)

    # Delete the same noise batches we inserted
    for noise in noise_batches:
        incdbscan.delete(noise)

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)
