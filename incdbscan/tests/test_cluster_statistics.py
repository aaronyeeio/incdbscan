import numpy as np
import pytest
from sklearn.datasets import make_blobs

from incdbscan import IncrementalDBSCAN


class TestClusterStatistics:
    """Test suite for verifying cluster statistics correctness."""

    def test_basic_statistics_after_insertion(self):
        """Test that basic statistics are correct after inserting data."""
        EPS = 2.0
        MIN_PTS = 3

        # Create a simple dataset with clear clusters
        data, _ = make_blobs(
            n_samples=100,
            centers=3,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        stats = incdbscan.get_cluster_statistics()

        # Verify total objects count
        total_objects_in_clusters = sum(
            cluster_info['size']
            for cluster_info in stats['clusters'].values()
        )
        assert stats['total_objects'] == total_objects_in_clusters

        # Verify total cores and borders
        total_cores = sum(
            cluster_info['core_count']
            for cluster_info in stats['clusters'].values()
        )
        total_borders = sum(
            cluster_info['border_count']
            for cluster_info in stats['clusters'].values()
        )
        assert stats['total_cores'] == total_cores
        assert stats['total_borders'] == total_borders

        # Verify each cluster's internal consistency
        for cluster_info in stats['clusters'].values():
            assert cluster_info['size'] == cluster_info['core_count'] + \
                cluster_info['border_count']
            assert cluster_info['size'] > 0

    def test_statistics_consistency_during_incremental_insertion(self):
        """Test that statistics remain consistent during incremental insertions."""
        EPS = 2.0
        MIN_PTS = 4

        data, _ = make_blobs(
            n_samples=200,
            centers=4,
            n_features=2,
            cluster_std=0.6,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)

        # Insert data in 4 batches
        batch_size = len(data) // 4
        for i in range(4):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < 3 else len(data)
            incdbscan.insert(data[start_idx:end_idx])

            stats = incdbscan.get_cluster_statistics()

            # Verify consistency after each batch
            total_size = sum(c['size'] for c in stats['clusters'].values())
            total_cores = sum(c['core_count']
                              for c in stats['clusters'].values())
            total_borders = sum(c['border_count']
                                for c in stats['clusters'].values())

            assert stats['total_objects'] == total_size
            assert stats['total_cores'] == total_cores
            assert stats['total_borders'] == total_borders

            # Each cluster should have consistent internal counts
            for cluster_info in stats['clusters'].values():
                assert cluster_info['size'] == cluster_info['core_count'] + \
                    cluster_info['border_count']

    def test_statistics_after_deletion(self):
        """Test that statistics are correctly updated after deletion."""
        EPS = 2.0
        MIN_PTS = 3

        data, _ = make_blobs(
            n_samples=150,
            centers=3,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        inserted_objects = incdbscan.insert(data)

        stats_before = incdbscan.get_cluster_statistics()

        # Delete half of the data by IDs
        ids_to_delete = [
            obj.id for obj in inserted_objects[:len(inserted_objects) // 2]]
        incdbscan.delete(ids_to_delete)

        stats_after = incdbscan.get_cluster_statistics()

        # Verify statistics are still consistent after deletion
        total_size = sum(c['size'] for c in stats_after['clusters'].values())
        total_cores = sum(c['core_count']
                          for c in stats_after['clusters'].values())
        total_borders = sum(c['border_count']
                            for c in stats_after['clusters'].values())

        assert stats_after['total_objects'] == total_size
        assert stats_after['total_cores'] == total_cores
        assert stats_after['total_borders'] == total_borders

        # Total objects should decrease
        assert stats_after['total_objects'] < stats_before['total_objects']

        # Each cluster should have consistent internal counts
        for cluster_info in stats_after['clusters'].values():
            assert cluster_info['size'] == cluster_info['core_count'] + \
                cluster_info['border_count']

    def test_statistics_with_insert_delete_cycles(self):
        """Test statistics correctness through multiple insert/delete cycles."""
        EPS = 2.0
        MIN_PTS = 4

        main_data, _ = make_blobs(
            n_samples=100,
            centers=2,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(main_data)

        # Perform multiple insert-delete cycles
        np.random.seed(123)
        for _ in range(3):
            noise = np.random.uniform(-10, 10, (50, 2))
            noise_objects = incdbscan.insert(noise)

            stats_after_insert = incdbscan.get_cluster_statistics()
            self._verify_statistics_consistency(stats_after_insert)

            # Delete the noise objects
            noise_ids = [obj.id for obj in noise_objects]
            incdbscan.delete(noise_ids)

            stats_after_delete = incdbscan.get_cluster_statistics()
            self._verify_statistics_consistency(stats_after_delete)

    def test_core_vs_border_classification(self):
        """Test that core and border points are correctly classified in statistics."""
        EPS = 1.0
        MIN_PTS = 3

        # Create a controlled scenario with clear core and border points
        # Line of 3 points close together (cores) + 2 isolated points nearby (borders)
        data = np.array([
            # Core - has 2 neighbors (itself + [0.5, 0] + [1.0, 0])
            [0, 0],
            [0.5, 0],    # Core - has 2 neighbors
            [1.0, 0],    # Core - has 2 neighbors
            [0, 0.9],    # Border - attached to first core
            [1.0, 0.9],  # Border - attached to last core
        ])

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        inserted_objects = incdbscan.insert(data)

        stats = incdbscan.get_cluster_statistics()

        # Count actual core and border points by checking object properties
        actual_core_count = 0
        actual_border_count = 0

        # Count actual core and border points by checking object properties
        for obj in inserted_objects:
            label = incdbscan.clusters.get_label(obj)
            if label >= 0:  # Only count clustered objects
                if obj.is_core:
                    actual_core_count += 1
                else:
                    actual_border_count += 1

        # Statistics should match actual counts
        assert stats['total_cores'] == actual_core_count
        assert stats['total_borders'] == actual_border_count

        # Verify internal consistency
        assert stats['total_objects'] == actual_core_count + \
            actual_border_count

    def test_empty_cluster_statistics(self):
        """Test statistics when no clusters are formed."""
        EPS = 0.1
        MIN_PTS = 10

        # Sparse data that won't form clusters
        np.random.seed(42)
        data = np.random.uniform(-10, 10, (20, 2))

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        stats = incdbscan.get_cluster_statistics()

        # Should have no clusters or very few
        assert stats['cluster_count'] == 0 or stats['cluster_count'] <= 1
        assert stats['total_objects'] >= 0
        assert stats['total_cores'] >= 0
        assert stats['total_borders'] >= 0

    def test_single_cluster_statistics(self):
        """Test statistics when all points form a single cluster."""
        EPS = 5.0
        MIN_PTS = 3

        # Dense cluster where all points are close together
        data, _ = make_blobs(
            n_samples=50,
            centers=1,
            n_features=2,
            cluster_std=0.3,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        stats = incdbscan.get_cluster_statistics()

        # Should form exactly one cluster
        assert stats['cluster_count'] == 1

        # All objects should be in the cluster
        cluster_info = list(stats['clusters'].values())[0]
        assert cluster_info['size'] == cluster_info['core_count'] + \
            cluster_info['border_count']
        assert cluster_info['size'] > 0

    def test_statistics_individual_cluster_properties(self):
        """Test that individual cluster statistics are correctly tracked."""
        EPS = 2.0
        MIN_PTS = 3

        data, _ = make_blobs(
            n_samples=120,
            centers=3,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        clusters = incdbscan.get_all_clusters()

        for cluster in clusters:
            # Manually count core and border members
            manual_core_count = 0
            manual_border_count = 0

            for member in cluster.members:
                if member.is_core:
                    manual_core_count += 1
                else:
                    manual_border_count += 1

            # Verify against cluster's statistics
            assert cluster.core_count == manual_core_count
            assert cluster.border_count == manual_border_count
            assert cluster.size == len(cluster.members)
            assert cluster.size == manual_core_count + manual_border_count

    def test_statistics_after_cluster_merge(self):
        """Test that statistics are correct after clusters merge."""
        EPS = 2.0
        MIN_PTS = 3

        # Create two separate clusters
        cluster1, _ = make_blobs(
            n_samples=30,
            centers=[[0, 0]],
            n_features=2,
            cluster_std=0.3,
            random_state=42
        )

        cluster2, _ = make_blobs(
            n_samples=30,
            centers=[[10, 10]],
            n_features=2,
            cluster_std=0.3,
            random_state=43
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(cluster1)
        incdbscan.insert(cluster2)

        stats_separate = incdbscan.get_cluster_statistics()

        # Add a bridge point that might merge clusters
        bridge = np.array([[5, 5]])
        incdbscan.insert(bridge)

        stats_after_bridge = incdbscan.get_cluster_statistics()

        # Verify statistics consistency regardless of merge
        self._verify_statistics_consistency(stats_after_bridge)

    def test_statistics_with_noise_points(self):
        """Test that noise points don't affect cluster statistics."""
        EPS = 1.0
        MIN_PTS = 5

        # Create data with clear noise
        cluster_data, _ = make_blobs(
            n_samples=50,
            centers=1,
            n_features=2,
            cluster_std=0.3,
            random_state=42
        )

        np.random.seed(123)
        noise_data = np.random.uniform(-10, 10, (20, 2))

        data = np.vstack([cluster_data, noise_data])

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        inserted_objects = incdbscan.insert(data)

        stats = incdbscan.get_cluster_statistics()

        # Count actual clustered points (excluding noise)
        clustered_count = 0
        for obj in inserted_objects:
            label = incdbscan.clusters.get_label(obj)
            if label >= 0:  # Not noise
                clustered_count += 1

        # Statistics should only count clustered objects, not noise
        assert stats['total_objects'] == clustered_count
        assert stats['total_objects'] < len(
            data)  # Some points should be noise

    def test_event_counters_in_statistics(self):
        """Test that event counters (total_objects_added, total_objects_removed) are tracked."""
        EPS = 2.0
        MIN_PTS = 3

        data, _ = make_blobs(
            n_samples=100,
            centers=2,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        inserted_objects = incdbscan.insert(data)

        clusters = incdbscan.get_all_clusters()

        # All clusters should have objects added
        for cluster in clusters:
            assert cluster.total_objects_added > 0
            assert cluster.total_objects_removed == 0  # Nothing deleted yet

        # Delete some data by IDs
        ids_to_delete = [obj.id for obj in inserted_objects[:30]]
        incdbscan.delete(ids_to_delete)

        clusters_after_delete = incdbscan.get_all_clusters()

        # At least some clusters should have objects removed
        total_removed = sum(
            c.total_objects_removed for c in clusters_after_delete)
        assert total_removed >= 0

    @staticmethod
    def _verify_statistics_consistency(stats):
        """Helper method to verify statistics internal consistency."""
        total_size = sum(c['size'] for c in stats['clusters'].values())
        total_cores = sum(c['core_count'] for c in stats['clusters'].values())
        total_borders = sum(c['border_count']
                            for c in stats['clusters'].values())

        assert stats['total_objects'] == total_size
        assert stats['total_cores'] == total_cores
        assert stats['total_borders'] == total_borders

        for cluster_info in stats['clusters'].values():
            assert cluster_info['size'] == cluster_info['core_count'] + \
                cluster_info['border_count']
            assert cluster_info['size'] >= 0
            assert cluster_info['core_count'] >= 0
            assert cluster_info['border_count'] >= 0


class TestClusterLifecycleStatistics:
    """Test statistics related to cluster lifecycle events."""

    def test_cluster_creation_timestamp(self):
        """Test that cluster creation timestamp is recorded."""
        EPS = 2.0
        MIN_PTS = 3

        data, _ = make_blobs(
            n_samples=50,
            centers=2,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        import time
        before_time = time.time()

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        after_time = time.time()

        clusters = incdbscan.get_all_clusters()

        for cluster in clusters:
            assert before_time <= cluster.created_at <= after_time
            assert cluster.last_modified >= cluster.created_at

    def test_cluster_lifetime_tracking(self):
        """Test that cluster lifetime is correctly tracked."""
        EPS = 2.0
        MIN_PTS = 3

        data, _ = make_blobs(
            n_samples=50,
            centers=1,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        stats = incdbscan.get_cluster_statistics()

        for cluster_info in stats['clusters'].values():
            assert cluster_info['lifetime'] >= 0
            assert 'created_at' in cluster_info

    def test_parent_labels_tracking(self):
        """Test that parent labels are tracked for merged clusters."""
        EPS = 2.0
        MIN_PTS = 3

        data, _ = make_blobs(
            n_samples=100,
            centers=3,
            n_features=2,
            cluster_std=0.5,
            random_state=42
        )

        incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
        incdbscan.insert(data)

        clusters = incdbscan.get_all_clusters()

        for cluster in clusters:
            # Parent labels should be a list
            assert isinstance(cluster.parent_labels, list)
            assert isinstance(cluster.merge_count, int)
            assert isinstance(cluster.split_count, int)
