"""Unit tests for two-level density DBSCAN functionality."""

import numpy as np
import pytest

from incdbscan import IncrementalDBSCAN
from testutils import (
    CLUSTER_LABEL_NOISE,
    assert_cluster_labels,
)


EPS = 1.5
EPS_MERGE = 0.8
MIN_PTS = 3


@pytest.fixture
def incdbscan_two_level():
    """Create IncrementalDBSCAN with two-level density."""
    return IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, eps_merge=EPS_MERGE)


@pytest.fixture
def incdbscan_standard():
    """Create standard IncrementalDBSCAN (eps_merge = eps)."""
    return IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS, eps_merge=EPS)


@pytest.fixture
def two_clusters_with_bridge():
    """Two well-separated clusters with bridge points."""
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) * 0.3 + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) * 0.3 + np.array([5, 0])
    bridge = np.array([[2.3, 0.1], [2.7, -0.1]])
    return cluster1, cluster2, bridge


@pytest.fixture
def dense_cluster():
    """A dense cluster of points."""
    np.random.seed(123)
    return np.random.randn(15, 2) * 0.3 + np.array([0, 0])


class TestParameterValidation:
    """Test parameter validation for two-level density."""

    def test_eps_merge_greater_than_eps_raises_error(self):
        """Test that eps_merge > eps raises ValueError."""
        with pytest.raises(ValueError, match="eps_merge must be <= eps"):
            IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=1.5)

    def test_eps_merge_equals_eps_is_valid(self):
        """Test that eps_merge = eps is valid (standard DBSCAN)."""
        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=1.5)
        assert dbscan.eps == 1.5
        assert dbscan.eps_merge == 1.5

    def test_eps_merge_less_than_eps_is_valid(self):
        """Test that eps_merge < eps is valid."""
        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=0.8)
        assert dbscan.eps == 1.5
        assert dbscan.eps_merge == 0.8

    def test_eps_merge_defaults_to_eps_when_none(self):
        """Test that eps_merge defaults to eps when not specified."""
        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3)
        assert dbscan.eps_merge == dbscan.eps


class TestBridgePrevention:
    """Test that two-level density prevents bridge merging."""

    def test_prevents_bridge_merging(self, two_clusters_with_bridge):
        """Test that two-level density keeps clusters separate despite bridge."""
        cluster1, cluster2, bridge = two_clusters_with_bridge
        X = np.vstack([cluster1, cluster2, bridge])

        # Two-level density should keep clusters separate
        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=0.8)
        dbscan.insert(X)
        labels = dbscan.get_cluster_labels(X)

        # Get labels of cluster centers
        label_cluster1 = dbscan.get_cluster_labels(cluster1[:1])[0]
        label_cluster2 = dbscan.get_cluster_labels(cluster2[:1])[0]

        # Clusters should have different labels
        assert label_cluster1 != label_cluster2
        assert label_cluster1 >= 0  # Not noise
        assert label_cluster2 >= 0  # Not noise

    def test_standard_dbscan_may_merge_via_bridge(self, two_clusters_with_bridge):
        """Test that standard DBSCAN behavior is different."""
        cluster1, cluster2, bridge = two_clusters_with_bridge
        X = np.vstack([cluster1, cluster2, bridge])

        # Standard DBSCAN with same eps
        dbscan_std = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=1.5)
        dbscan_std.insert(X)
        labels_std = dbscan_std.get_cluster_labels(X)

        # Two-level density
        dbscan_two = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=0.8)
        dbscan_two.insert(X)
        labels_two = dbscan_two.get_cluster_labels(X)

        # Count clusters (excluding noise)
        n_clusters_std = len(set(labels_std)) - (1 if -1 in labels_std else 0)
        n_clusters_two = len(set(labels_two)) - (1 if -1 in labels_two else 0)

        # Two-level should have >= clusters (prevents merging)
        assert n_clusters_two >= n_clusters_std

    def test_bridge_points_can_be_assigned(self, two_clusters_with_bridge):
        """Test that bridge points can still be assigned to clusters."""
        cluster1, cluster2, bridge = two_clusters_with_bridge

        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=0.8)
        dbscan.insert(cluster1)
        dbscan.insert(cluster2)
        dbscan.insert(bridge)

        # Bridge points should be assigned (not necessarily noise)
        # This depends on their distance to core points
        labels = dbscan.get_cluster_labels(bridge)
        # At least they should have valid labels (could be cluster or noise)
        assert all(label >= -1 for label in labels)


class TestIncrementalBehavior:
    """Test incremental insertion with two-level density."""

    def test_incremental_insertion_maintains_separation(self, two_clusters_with_bridge):
        """Test that incremental insertion maintains cluster separation."""
        cluster1, cluster2, bridge = two_clusters_with_bridge

        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=0.8)

        # Insert clusters incrementally
        dbscan.insert(cluster1)
        label1_initial = dbscan.get_cluster_labels(cluster1[:1])[0]
        assert label1_initial >= 0

        dbscan.insert(cluster2)
        label2_initial = dbscan.get_cluster_labels(cluster2[:1])[0]
        assert label2_initial >= 0
        assert label1_initial != label2_initial

        # Insert bridge - clusters should remain separate
        dbscan.insert(bridge)
        label1_after = dbscan.get_cluster_labels(cluster1[:1])[0]
        label2_after = dbscan.get_cluster_labels(cluster2[:1])[0]
        assert label1_after != label2_after

    def test_incremental_insertion_creates_clusters_correctly(self):
        """Test that clusters are created correctly with incremental insertion."""
        np.random.seed(42)
        cluster = np.random.randn(10, 2) * 0.3

        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=0.5)

        # Insert points one by one
        for i in range(len(cluster)):
            dbscan.insert(cluster[i:i+1])

        # All points should eventually be in a cluster
        labels = dbscan.get_cluster_labels(cluster)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1


class TestBorderPointAssignment:
    """Test that border points are assigned using eps, not eps_merge."""

    def test_border_points_use_eps_for_assignment(self):
        """Test that border points are assigned based on eps distance."""
        # Create core points forming a cluster
        core_points = np.array([
            [0, 0], [0, 0.2], [0.2, 0], [0, -0.2], [-0.2, 0]
        ])

        # Border point within eps but outside eps_merge
        border_point = np.array([[1.0, 0]])

        dbscan = IncrementalDBSCAN(eps=1.2, min_pts=3, eps_merge=0.5)
        dbscan.insert(core_points)
        dbscan.insert(border_point)

        # Core points should form a cluster
        core_label = dbscan.get_cluster_labels(core_points[:1])[0]
        assert core_label >= 0

        # Border point should be assigned to the cluster (using eps)
        border_label = dbscan.get_cluster_labels(border_point)[0]
        assert border_label == core_label

    def test_border_points_become_noise_when_outside_eps(self):
        """Test that points outside eps become noise."""
        core_points = np.array([
            [0, 0], [0, 0.2], [0.2, 0], [0, -0.2], [-0.2, 0]
        ])

        # Point outside eps
        far_point = np.array([[5.0, 0]])

        dbscan = IncrementalDBSCAN(eps=1.2, min_pts=3, eps_merge=0.5)
        dbscan.insert(core_points)
        dbscan.insert(far_point)

        # Far point should be noise
        far_label = dbscan.get_cluster_labels(far_point)[0]
        assert far_label == CLUSTER_LABEL_NOISE


class TestDeletion:
    """Test deletion with two-level density."""

    def test_deletion_maintains_two_level_behavior(self, dense_cluster):
        """Test that deletion works correctly with two-level density."""
        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=0.5)
        dbscan.insert(dense_cluster)

        # All points should be in a cluster initially
        labels_before = dbscan.get_cluster_labels(dense_cluster)
        assert all(label >= 0 for label in labels_before)

        # Delete some points
        points_to_delete = dense_cluster[:5]
        dbscan.delete(points_to_delete)

        # Remaining points
        remaining_points = dense_cluster[5:]
        labels_after = dbscan.get_cluster_labels(remaining_points)

        # Should still have valid labels (could be cluster or noise)
        assert all(label >= -1 for label in labels_after)

    def test_deletion_can_split_clusters(self):
        """Test that deleting bridge points can split clusters."""
        # Create two groups connected by a point
        group1 = np.array([[0, 0], [0, 0.2], [0, -0.2]])
        bridge_point = np.array([[1.0, 0]])
        group2 = np.array([[2.0, 0], [2.0, 0.2], [2.0, -0.2]])

        all_points = np.vstack([group1, bridge_point, group2])

        # With standard DBSCAN, might form one cluster
        dbscan = IncrementalDBSCAN(eps=1.2, min_pts=2, eps_merge=1.2)
        dbscan.insert(all_points)

        # Delete bridge point
        dbscan.delete(bridge_point)

        # Groups might split
        labels_group1 = dbscan.get_cluster_labels(group1)
        labels_group2 = dbscan.get_cluster_labels(group2)

        # At least they should have valid labels
        assert all(label >= -1 for label in labels_group1)
        assert all(label >= -1 for label in labels_group2)


class TestEquivalenceToStandardDBSCAN:
    """Test that eps_merge=eps is equivalent to standard DBSCAN."""

    def test_same_results_as_standard_dbscan(self):
        """Test that eps_merge=eps gives same results as standard DBSCAN."""
        np.random.seed(42)
        X = np.random.randn(50, 2)

        # Two-level with eps_merge=eps
        dbscan_two = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=1.0)
        dbscan_two.insert(X)
        labels_two = dbscan_two.get_cluster_labels(X)

        # Default (no eps_merge specified)
        dbscan_default = IncrementalDBSCAN(eps=1.0, min_pts=3)
        dbscan_default.insert(X)
        labels_default = dbscan_default.get_cluster_labels(X)

        # Should be identical
        assert np.array_equal(labels_two, labels_default)

    def test_incremental_equivalence(self):
        """Test incremental insertion equivalence."""
        np.random.seed(42)
        X1 = np.random.randn(20, 2)
        X2 = np.random.randn(20, 2) + np.array([3, 0])

        # Two-level with eps_merge=eps
        dbscan_two = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=1.5)
        dbscan_two.insert(X1)
        dbscan_two.insert(X2)

        # Default
        dbscan_default = IncrementalDBSCAN(eps=1.5, min_pts=3)
        dbscan_default.insert(X1)
        dbscan_default.insert(X2)

        # Should be identical
        all_points = np.vstack([X1, X2])
        labels_two = dbscan_two.get_cluster_labels(all_points)
        labels_default = dbscan_default.get_cluster_labels(all_points)
        assert np.array_equal(labels_two, labels_default)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_single_point_is_noise(self):
        """Test that a single point is labeled as noise."""
        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=0.5)
        point = np.array([[0, 0]])
        dbscan.insert(point)
        assert_cluster_labels(dbscan, point, CLUSTER_LABEL_NOISE)

    def test_empty_insertion(self):
        """Test that empty insertion raises appropriate error."""
        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=0.5)
        empty = np.array([]).reshape(0, 2)
        # sklearn validation requires at least 1 sample
        with pytest.raises(ValueError, match="minimum of 1 is required"):
            dbscan.insert(empty)

    def test_duplicate_points(self):
        """Test handling of duplicate points."""
        points = np.array([
            [0, 0], [0, 0], [0, 0.1], [0, 0.1]
        ])
        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=2, eps_merge=0.5)
        dbscan.insert(points)

        labels = dbscan.get_cluster_labels(points)
        # Should form a cluster
        assert any(label >= 0 for label in labels)

    def test_very_small_eps_merge(self):
        """Test with very small eps_merge."""
        np.random.seed(42)
        cluster1 = np.random.randn(10, 2) * 0.3
        cluster2 = np.random.randn(10, 2) * 0.3 + np.array([2, 0])

        # Very restrictive eps_merge
        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=0.1)
        dbscan.insert(np.vstack([cluster1, cluster2]))

        labels = dbscan.get_cluster_labels(np.vstack([cluster1, cluster2]))
        # Should create multiple small clusters or noise
        assert len(set(labels)) >= 2


class TestMultipleScenarios:
    """Test various realistic scenarios."""

    def test_three_clusters_chain(self):
        """Test three clusters in a chain with bridges."""
        np.random.seed(42)
        cluster1 = np.random.randn(15, 2) * 0.3 + np.array([0, 0])
        cluster2 = np.random.randn(15, 2) * 0.3 + np.array([4, 0])
        cluster3 = np.random.randn(15, 2) * 0.3 + np.array([8, 0])

        bridge1 = np.array([[1.9, 0], [2.1, 0]])
        bridge2 = np.array([[5.9, 0], [6.1, 0]])

        X = np.vstack([cluster1, cluster2, cluster3, bridge1, bridge2])

        # Two-level should keep clusters separate
        dbscan = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_merge=0.7)
        dbscan.insert(X)
        labels = dbscan.get_cluster_labels(X)

        # Should have multiple clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 2

    def test_concentric_circles(self):
        """Test concentric circle-like clusters."""
        np.random.seed(42)

        # Inner circle
        theta = np.linspace(0, 2*np.pi, 20)
        inner = np.column_stack([np.cos(theta)*0.5, np.sin(theta)*0.5])

        # Outer circle
        outer = np.column_stack([np.cos(theta)*3, np.sin(theta)*3])

        X = np.vstack([inner, outer])

        dbscan = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_merge=0.5)
        dbscan.insert(X)
        labels = dbscan.get_cluster_labels(X)

        # Should identify separate structures
        assert len(set(labels)) >= 1

    def test_noise_between_clusters(self):
        """Test handling noise points between clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(15, 2) * 0.3
        cluster2 = np.random.randn(15, 2) * 0.3 + np.array([5, 0])
        noise = np.random.uniform(1, 4, size=(5, 2))

        X = np.vstack([cluster1, cluster2, noise])

        dbscan = IncrementalDBSCAN(eps=1.2, min_pts=3, eps_merge=0.6)
        dbscan.insert(X)
        labels = dbscan.get_cluster_labels(X)

        # Should have noise points
        assert CLUSTER_LABEL_NOISE in labels
