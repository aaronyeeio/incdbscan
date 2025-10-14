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


def test_sklearn_consistency_with_sample_weights():
    """Test consistency with sample weights"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)
    weights = np.random.uniform(0.5, 2.0, len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(
        data, sample_weight=weights).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


@pytest.mark.parametrize("weight_range", [
    (0.5, 1.5),
    (1.0, 3.0),
    (0.1, 5.0),
])
def test_sklearn_consistency_with_different_weight_ranges(weight_range):
    """Test consistency with various weight ranges"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)
    weights = np.random.uniform(weight_range[0], weight_range[1], len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(
        data, sample_weight=weights).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_weighted_insert_delete():
    """Test consistency with weighted insertion and deletion"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    split_idx = len(data) // 2
    data_part1 = data[:split_idx]
    data_part2 = data[split_idx:]

    np.random.seed(42)
    weights_part1 = np.random.uniform(0.5, 2.0, len(data_part1))
    weights_part2 = np.random.uniform(0.5, 2.0, len(data_part2))

    # Expected result: only first part with weights
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data_part1, sample_weight=weights_part1)

    # Insert both parts with weights, then delete second part with same weights
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data_part1, sample_weight=weights_part1)
    incdbscan.insert(data_part2, sample_weight=weights_part2)
    incdbscan.delete(data_part2, sample_weight=weights_part2)
    labels_incdbscan = incdbscan.get_cluster_labels(data_part1)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_incremental_weighted_insertion():
    """Test consistency when weighted data is inserted incrementally"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)
    weights = np.random.uniform(0.5, 2.0, len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    # Insert weighted data in 4 batches
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    batch_size = len(data) // 4
    for i in range(4):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < 3 else len(data)
        incdbscan.insert(
            data[start_idx:end_idx],
            sample_weight=weights[start_idx:end_idx]
        )

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_mixed_weighted_operations():
    """Test consistency with complex mix of weighted insertions and deletions"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)
    weights = np.random.uniform(0.5, 2.0, len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)

    # Insert some noise with weights
    noise = np.random.uniform(-14, 14, (500, 2))
    noise_weights = np.random.uniform(0.5, 2.0, len(noise))
    incdbscan.insert(noise, sample_weight=noise_weights)

    # Insert actual data with weights
    incdbscan.insert(data, sample_weight=weights)

    # Insert more noise with weights
    noise2 = np.random.uniform(-14, 14, (300, 2))
    noise2_weights = np.random.uniform(0.5, 2.0, len(noise2))
    incdbscan.insert(noise2, sample_weight=noise2_weights)

    # Delete noise with same weights
    incdbscan.delete(noise, sample_weight=noise_weights)
    incdbscan.delete(noise2, sample_weight=noise2_weights)

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_integer_weights():
    """Test consistency with integer weights simulating duplicate points"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)
    # Integer weights simulate having multiple copies of the same point
    weights = np.random.randint(1, 4, len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan = incdbscan.insert(
        data, sample_weight=weights).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_partial_weight_changes():
    """Test behavior when inserting same points with different weights multiple times"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Insert with weight 1, then add more weight to same points
    initial_weights = np.ones(len(data))
    additional_weights = np.random.uniform(0.5, 1.5, len(data))
    final_weights = initial_weights + additional_weights

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=initial_weights)
    incdbscan.insert(data, sample_weight=additional_weights)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_partial_weight_deletion():
    """Test partial weight deletion (delete less weight than inserted)"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Insert with total weight, then delete partial weight
    total_weights = np.random.uniform(2.0, 4.0, len(data))
    delete_weights = np.random.uniform(0.5, 1.5, len(data))
    remaining_weights = total_weights - delete_weights

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=remaining_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=total_weights)
    incdbscan.delete(data, sample_weight=delete_weights)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_multiple_weight_accumulations():
    """Test multiple weight accumulations on same points"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Accumulate weights through multiple insertions
    weight1 = np.random.uniform(0.3, 0.7, len(data))
    weight2 = np.random.uniform(0.4, 0.8, len(data))
    weight3 = np.random.uniform(0.5, 0.9, len(data))
    final_weights = weight1 + weight2 + weight3

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=weight1)
    incdbscan.insert(data, sample_weight=weight2)
    incdbscan.insert(data, sample_weight=weight3)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_weight_fluctuations():
    """Test alternating insert/delete operations with weights"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Final state: insert 3.0, delete 1.0, insert 0.5 = 2.5 total
    weight1 = np.ones(len(data)) * 3.0
    weight2 = np.ones(len(data)) * 1.0
    weight3 = np.ones(len(data)) * 0.5
    final_weights = weight1 - weight2 + weight3

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=weight1)
    incdbscan.delete(data, sample_weight=weight2)
    incdbscan.insert(data, sample_weight=weight3)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_mixed_new_and_weight_updates():
    """Test inserting new points while updating weights of existing points"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    split_idx = len(data) // 2
    data_part1 = data[:split_idx]
    data_part2 = data[split_idx:]

    np.random.seed(42)
    initial_weights_part1 = np.ones(len(data_part1))
    additional_weights_part1 = np.random.uniform(0.5, 1.5, len(data_part1))
    weights_part2 = np.random.uniform(1.0, 2.0, len(data_part2))

    final_weights = np.concatenate([
        initial_weights_part1 + additional_weights_part1,
        weights_part2
    ])

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    # First insert part1 with initial weights
    incdbscan.insert(data_part1, sample_weight=initial_weights_part1)
    # Then insert part2 (new points) AND update part1 weights simultaneously
    incdbscan.insert(data_part2, sample_weight=weights_part2)
    incdbscan.insert(data_part1, sample_weight=additional_weights_part1)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_boundary_weight_transitions():
    """Test weight changes around min_pts boundary"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Create weights that will cause some points to cross min_pts threshold
    # Start with small weights (some points won't be cores)
    small_weights = np.random.uniform(0.1, 0.3, len(data))
    # Add weights to push some over the threshold
    push_weights = np.random.uniform(0.2, 0.5, len(data))
    final_weights = small_weights + push_weights

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=small_weights)
    incdbscan.insert(data, sample_weight=push_weights)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_complex_weight_mixing():
    """Test complex scenario with multiple batches and weight operations"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Split into 3 parts
    n = len(data) // 3
    part1 = data[:n]
    part2 = data[n:2*n]
    part3 = data[2*n:]

    w1_initial = np.random.uniform(0.5, 1.0, len(part1))
    w1_add = np.random.uniform(0.3, 0.7, len(part1))
    w2 = np.random.uniform(1.0, 2.0, len(part2))
    w2_reduce = np.random.uniform(0.2, 0.5, len(part2))
    w3 = np.random.uniform(1.5, 2.5, len(part3))

    final_weights = np.concatenate([
        w1_initial + w1_add,
        w2 - w2_reduce,
        w3
    ])

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    # Complex sequence of operations
    incdbscan.insert(part1, sample_weight=w1_initial)
    incdbscan.insert(part2, sample_weight=w2)
    incdbscan.insert(part3, sample_weight=w3)
    incdbscan.delete(part2, sample_weight=w2_reduce)
    incdbscan.insert(part1, sample_weight=w1_add)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_noise_weight_transitions():
    """Test points transitioning between noise and cluster membership via weights"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(123)  # Different seed for variety

    # Use varied weights that create interesting boundary cases
    weights = np.random.choice([0.5, 1.0, 1.5, 2.0, 3.0], size=len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    # Build up to same weights incrementally
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    base_weights = np.ones(len(data)) * 0.5
    incdbscan.insert(data, sample_weight=base_weights)

    # Add additional weight to points that need it
    additional_weights = weights - base_weights
    mask = additional_weights > 0
    if np.any(mask):
        incdbscan.insert(data[mask], sample_weight=additional_weights[mask])

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_multiple_deletions_same_points():
    """Test multiple delete operations on the same points"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Insert with weight 5.0, then delete in two steps: -2.0, -1.5
    # Final weight should be 1.5
    insert_weight = np.ones(len(data)) * 5.0
    delete_weight1 = np.ones(len(data)) * 2.0
    delete_weight2 = np.ones(len(data)) * 1.5
    final_weights = insert_weight - delete_weight1 - delete_weight2

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=insert_weight)
    incdbscan.delete(data, sample_weight=delete_weight1)
    incdbscan.delete(data, sample_weight=delete_weight2)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_multiple_varied_deletions():
    """Test multiple deletions with different weights per point"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Start with varied weights, delete in multiple steps
    insert_weight = np.random.uniform(3.0, 5.0, len(data))
    delete_weight1 = np.random.uniform(0.5, 1.0, len(data))
    delete_weight2 = np.random.uniform(0.3, 0.8, len(data))
    delete_weight3 = np.random.uniform(0.2, 0.6, len(data))
    final_weights = insert_weight - delete_weight1 - delete_weight2 - delete_weight3

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=insert_weight)
    incdbscan.delete(data, sample_weight=delete_weight1)
    incdbscan.delete(data, sample_weight=delete_weight2)
    incdbscan.delete(data, sample_weight=delete_weight3)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_interleaved_insert_delete_same_points():
    """Test alternating insert and delete operations on the same points"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Complex sequence: +3.0, -1.0, +1.5, -0.8, +0.5 = 3.2 final
    w1 = np.ones(len(data)) * 3.0
    d1 = np.ones(len(data)) * 1.0
    w2 = np.ones(len(data)) * 1.5
    d2 = np.ones(len(data)) * 0.8
    w3 = np.ones(len(data)) * 0.5
    final_weights = w1 - d1 + w2 - d2 + w3

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=w1)
    incdbscan.delete(data, sample_weight=d1)
    incdbscan.insert(data, sample_weight=w2)
    incdbscan.delete(data, sample_weight=d2)
    incdbscan.insert(data, sample_weight=w3)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_deletion_across_threshold():
    """Test deletions that cause points to cross the min_pts threshold multiple times"""
    EPS = 1.0
    MIN_PTS = 5

    data = read_handl_data()
    np.random.seed(42)

    # Start with high weights (points are cores), then gradually reduce
    # Some points will transition from core -> border -> noise -> border -> core
    insert_weight = np.random.uniform(2.0, 3.0, len(data))
    delete1 = np.random.uniform(0.3, 0.5, len(data))
    delete2 = np.random.uniform(0.2, 0.4, len(data))
    delete3 = np.random.uniform(0.1, 0.3, len(data))
    # Then add back some weight
    insert2 = np.random.uniform(0.5, 1.0, len(data))
    final_weights = insert_weight - delete1 - delete2 - delete3 + insert2

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan.insert(data, sample_weight=insert_weight)
    incdbscan.delete(data, sample_weight=delete1)
    incdbscan.delete(data, sample_weight=delete2)
    incdbscan.delete(data, sample_weight=delete3)
    incdbscan.insert(data, sample_weight=insert2)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


# ============================================================================
# Cosine Distance Metric Tests
# ============================================================================

def test_sklearn_consistency_with_cosine_metric():
    """Test consistency with cosine distance metric"""
    EPS = 0.3
    MIN_PTS = 5

    # Generate normalized data for cosine similarity
    data, _ = make_blobs(
        n_samples=500,
        centers=5,
        n_features=10,
        cluster_std=0.5,
        random_state=42
    )
    # Normalize data for cosine distance
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


@pytest.mark.parametrize("eps,min_pts", [
    (0.2, 3),
    (0.3, 5),
    (0.4, 4),
    (0.5, 6),
])
def test_sklearn_consistency_with_cosine_different_parameters(eps, min_pts):
    """Test consistency with cosine metric and various parameters"""
    # Generate normalized data
    data, _ = make_blobs(
        n_samples=300,
        centers=4,
        n_features=8,
        cluster_std=0.4,
        random_state=42
    )
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    dbscan = DBSCAN(eps=eps, min_samples=min_pts, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts, metric='cosine')
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_cosine_incremental_insertion():
    """Test consistency with cosine metric when data is inserted incrementally"""
    EPS = 0.3
    MIN_PTS = 5

    data, _ = make_blobs(
        n_samples=400,
        centers=5,
        n_features=10,
        cluster_std=0.5,
        random_state=42
    )
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data)

    # Insert data in 4 batches
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    batch_size = len(data) // 4
    for i in range(4):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < 3 else len(data)
        incdbscan.insert(data[start_idx:end_idx])

    labels_incdbscan = incdbscan.get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_cosine_insert_delete():
    """Test consistency with cosine metric for insertion and deletion"""
    EPS = 0.3
    MIN_PTS = 5

    data, _ = make_blobs(
        n_samples=400,
        centers=4,
        n_features=8,
        cluster_std=0.5,
        random_state=42
    )
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    # Split data
    split_idx = len(data) // 2
    data_part1 = data[:split_idx]
    data_part2 = data[split_idx:]

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data_part1)

    # Insert all, then delete second part
    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    incdbscan.insert(data)
    incdbscan.delete(data_part2)
    labels_incdbscan = incdbscan.get_cluster_labels(data_part1)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_cosine_and_weights():
    """Test consistency with cosine metric and sample weights"""
    EPS = 0.3
    MIN_PTS = 5

    data, _ = make_blobs(
        n_samples=300,
        centers=4,
        n_features=8,
        cluster_std=0.5,
        random_state=42
    )
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    np.random.seed(42)
    weights = np.random.uniform(0.5, 2.0, len(data))

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data, sample_weight=weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    labels_incdbscan = incdbscan.insert(
        data, sample_weight=weights).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_cosine_weight_updates():
    """Test consistency with cosine metric and weight updates"""
    EPS = 0.3
    MIN_PTS = 5

    data, _ = make_blobs(
        n_samples=300,
        centers=4,
        n_features=8,
        cluster_std=0.5,
        random_state=42
    )
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    np.random.seed(42)
    initial_weights = np.ones(len(data))
    additional_weights = np.random.uniform(0.5, 1.5, len(data))
    final_weights = initial_weights + additional_weights

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data, sample_weight=final_weights)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    incdbscan.insert(data, sample_weight=initial_weights)
    incdbscan.insert(data, sample_weight=additional_weights)
    labels_incdbscan = incdbscan.get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_cosine_high_dimensional():
    """Test consistency with cosine metric on high-dimensional data"""
    EPS = 0.4
    MIN_PTS = 5

    # High-dimensional data (50 dimensions)
    data, _ = make_blobs(
        n_samples=200,
        centers=3,
        n_features=50,
        cluster_std=0.5,
        random_state=42
    )
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)


def test_sklearn_consistency_with_cosine_text_like_data():
    """Test consistency with cosine metric on sparse text-like data"""
    EPS = 0.3
    MIN_PTS = 3

    # Simulate sparse text-like data (many zeros, like TF-IDF vectors)
    np.random.seed(42)
    data = np.random.rand(200, 100)
    # Make it sparse (80% zeros)
    mask = np.random.rand(200, 100) < 0.8
    data[mask] = 0
    # Normalize for cosine
    row_norms = np.linalg.norm(data, axis=1, keepdims=True)
    # Avoid division by zero
    row_norms[row_norms == 0] = 1
    data = data / row_norms

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS, metric='cosine')
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, metric='cosine')
    labels_incdbscan = incdbscan.insert(data).get_cluster_labels(data)

    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan)
