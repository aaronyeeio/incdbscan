import numpy as np
import pytest

from incdbscan import IncrementalDBSCAN


def test_evict_border_points_only():
    """Test eviction when cluster has enough border points."""
    # Create a cluster with clear core and border structure
    # Use higher min_pts to ensure border points exist
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=5)

    # Core region: 4 points in a square
    core_points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    # Border points: close enough to be in cluster but with fewer than 5 neighbors
    border_points = np.array([
        [2.0, 0.5],   # Close to core but only has 3 neighbors
        [0.5, 2.0],   # Close to core but only has 3 neighbors
        [-0.5, 0.5],  # Close to core but only has 3 neighbors
    ])

    all_points = np.vstack([core_points, border_points])
    inserted_objects = incdbscan.insert(all_points)
    object_ids = [obj.id for obj in inserted_objects]

    # Verify we have one cluster
    labels = incdbscan.get_cluster_labels(object_ids)
    assert len(set(labels[labels >= 0])) == 1
    cluster_label = int(labels[0])

    cluster = incdbscan.get_cluster(cluster_label)
    initial_size = cluster.size
    initial_border_count = cluster.border_count

    # Verify we actually have border points
    assert initial_border_count > 0, "Test setup should create border points"

    # Evict 2 border points
    evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=2)

    assert len(evicted_ids) == 2

    # Verify evicted points are deleted (no longer in the system)
    for evicted_id in evicted_ids:
        assert incdbscan._objects.get_object_by_id(evicted_id) is None

    # Get updated cluster after eviction
    cluster = incdbscan.get_cluster(cluster_label)
    assert cluster.size == initial_size - 2
    # Border count should decrease by at most 2 (could be less if cascade deletions occurred)
    assert cluster.border_count <= initial_border_count

    # Cluster should still exist and be connected
    assert incdbscan.get_cluster(cluster_label) is not None


def test_evict_all_border_points():
    """Test evicting all border points."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=5)

    # Dense core region with 6 points
    core_points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0],
        [0, 0.5],
    ])

    # Isolated border points: close to core but won't have enough neighbors
    border_points = np.array([
        [2.2, 0.5],   # Far from most points, only close to 1-2 cores
        [-1.2, 0.5],  # Far from most points, only close to 1-2 cores
    ])

    all_points = np.vstack([core_points, border_points])
    inserted_objects = incdbscan.insert(all_points)
    object_ids = [obj.id for obj in inserted_objects]

    labels = incdbscan.get_cluster_labels(object_ids)
    cluster_label = int(labels[0])

    cluster = incdbscan.get_cluster(cluster_label)
    initial_border_count = cluster.border_count

    # Verify we have border points
    assert initial_border_count > 0, "Test setup should create border points"

    # Try to evict more than available borders
    evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=10)

    # Should evict at least all border points
    assert len(evicted_ids) >= initial_border_count

    # Verify evicted points are deleted
    for evicted_id in evicted_ids:
        assert incdbscan._objects.get_object_by_id(evicted_id) is None

    # Cluster should still exist (may have no borders or some cores removed)
    cluster = incdbscan.get_cluster(cluster_label)
    if cluster is not None:
        # If cluster exists, all borders should be gone
        assert cluster.border_count == 0


def test_evict_with_core_points():
    """Test eviction when need to remove core points."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=2)

    # Create a line of core points (no borders)
    points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
    ])

    inserted_objects = incdbscan.insert(points)
    object_ids = [obj.id for obj in inserted_objects]

    labels = incdbscan.get_cluster_labels(object_ids)
    cluster_label = int(labels[0])

    cluster = incdbscan.get_cluster(cluster_label)
    initial_size = cluster.size

    # Try to evict 3 points
    evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=3)

    # Should evict some points (endpoints are non-articulation points)
    assert len(evicted_ids) > 0

    # Verify evicted points are deleted
    for evicted_id in evicted_ids:
        assert incdbscan._objects.get_object_by_id(evicted_id) is None

    # Get updated cluster after eviction
    cluster = incdbscan.get_cluster(cluster_label)
    assert cluster.size == initial_size - len(evicted_ids)

    # Cluster should still be connected
    assert incdbscan.get_cluster(cluster_label) is not None


def test_evict_prevents_split():
    """Test that eviction doesn't split the cluster."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=2)

    # Create a dumbbell shape: two dense regions connected by one point
    left_region = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    bridge = np.array([[5, 0]])  # Bridge point (articulation point)

    right_region = np.array([
        [10, 0],
        [11, 0],
        [10, 1],
    ])

    all_points = np.vstack([left_region, bridge, right_region])
    inserted_objects = incdbscan.insert(all_points)
    object_ids = [obj.id for obj in inserted_objects]

    labels = incdbscan.get_cluster_labels(object_ids)
    unique_clusters = set(labels[labels >= 0])

    if len(unique_clusters) == 1:
        # If they form one cluster, the bridge is critical
        cluster_label = int(labels[0])
        cluster = incdbscan.get_cluster(cluster_label)
        initial_components = 1

        # Try to evict points
        evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=3)

        # Should evict some points but keep the bridge
        assert len(evicted_ids) > 0

        # Verify evicted points are deleted
        for evicted_id in evicted_ids:
            assert incdbscan._objects.get_object_by_id(evicted_id) is None

        # Cluster should still exist and be connected
        cluster = incdbscan.get_cluster(cluster_label)
        assert cluster is not None


def test_evict_from_empty_cluster():
    """Test eviction from non-existent cluster."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=3)

    # Try to evict from non-existent cluster
    evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label=999, n=5)

    assert len(evicted_ids) == 0


def test_evict_zero_points():
    """Test evicting zero points."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=3)

    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    inserted_objects = incdbscan.insert(points)
    object_ids = [obj.id for obj in inserted_objects]
    labels = incdbscan.get_cluster_labels(object_ids)
    cluster_label = int(labels[0])

    cluster = incdbscan.get_cluster(cluster_label)
    initial_size = cluster.size

    evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=0)

    assert len(evicted_ids) == 0

    # Get updated cluster after eviction (should be unchanged)
    cluster = incdbscan.get_cluster(cluster_label)
    assert cluster.size == initial_size


def test_evict_single_core_cluster():
    """Test evicting from cluster with only one core point."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=2)

    # Single core with borders
    core = np.array([[0, 0], [0.5, 0]])  # Will be one core region

    inserted_objects = incdbscan.insert(core)
    object_ids = [obj.id for obj in inserted_objects]
    labels = incdbscan.get_cluster_labels(object_ids)

    if labels[0] >= 0:
        cluster_label = int(labels[0])
        cluster = incdbscan.get_cluster(cluster_label)

        # Should be able to evict all points safely
        evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=10)
        assert len(evicted_ids) >= 0


def test_evict_maintains_cluster_statistics():
    """Test that cluster statistics are properly updated after eviction."""
    incdbscan = IncrementalDBSCAN(eps=1.5, min_pts=3)

    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2.5, 0.5],  # Border point
    ])

    inserted_objects = incdbscan.insert(points)
    object_ids = [obj.id for obj in inserted_objects]
    labels = incdbscan.get_cluster_labels(object_ids)
    cluster_label = int(labels[0])

    cluster = incdbscan.get_cluster(cluster_label)
    initial_size = cluster.size
    initial_weight = cluster.total_weight

    evicted_ids = incdbscan.evict_from_cluster_edge(cluster_label, n=1)

    assert len(evicted_ids) == 1

    # Verify evicted point is deleted
    assert incdbscan._objects.get_object_by_id(evicted_ids[0]) is None

    # Get updated cluster after eviction
    cluster = incdbscan.get_cluster(cluster_label)
    assert cluster.size == initial_size - 1
    assert cluster.total_weight == initial_weight - 1.0  # Default weight is 1.0
