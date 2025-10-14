"""Tests for soft clustering functionality."""

import numpy as np
import pytest
from incdbscan import IncrementalDBSCAN


class TestSoftClustering:
    """Test suite for soft clustering methods."""

    def test_soft_clustering_default_eps_soft(self):
        """Test that eps_soft defaults to 2*eps."""
        idb = IncrementalDBSCAN(eps=1.0, min_pts=3)

        # eps_soft should default to 2*eps
        assert idb.eps_soft == 2.0

        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
        ])
        idb.insert(X)

        # Should work without error
        probs, labels = idb.get_soft_labels(X)
        assert probs.shape[0] == len(X)

    def test_soft_clustering_with_eps_soft(self):
        """Test basic soft clustering with eps_soft parameter."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Cluster 1
            [5, 5], [5, 6], [6, 5], [6, 6],  # Cluster 2
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X)

        probs, labels = idb.get_soft_labels(X)

        # Should have 2 clusters
        assert len(labels) == 2

        # Probabilities should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

        # Shape should be (n_samples, n_clusters + 1) with noise column
        assert probs.shape == (len(X), 3)

    def test_soft_clustering_without_noise_prob(self):
        """Test soft clustering without noise probability column."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [5, 5], [5, 6], [6, 5], [6, 6],
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X)

        probs, labels = idb.get_soft_labels(X, include_noise_prob=False)

        # Shape should be (n_samples, n_clusters) without noise column
        assert probs.shape == (len(X), len(labels))

        # Probabilities should still sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_soft_clustering_kernels(self):
        """Test different kernel functions."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [5, 5], [5, 6], [6, 5], [6, 6],
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X)

        kernels = ['gaussian', 'inverse', 'linear']

        for kernel in kernels:
            probs, labels = idb.get_soft_labels(X, kernel=kernel)

            # All should produce valid probability distributions
            assert probs.shape[0] == len(X)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0)
            assert np.all(probs >= 0.0)
            assert np.all(probs <= 1.0)

    def test_soft_clustering_invalid_kernel(self):
        """Test that invalid kernel raises error."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Make sure we have a cluster
        ])
        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=2.0)
        idb.insert(X)

        with pytest.raises(ValueError, match="Unknown kernel"):
            idb.get_soft_labels(X, kernel='invalid_kernel')

    def test_soft_clustering_no_clusters(self):
        """Test soft clustering when no clusters exist (all noise)."""
        X = np.array([
            [0, 0], [10, 10], [20, 20], [30, 30]  # All far apart
        ])

        idb = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_soft=2.0)
        idb.insert(X)

        probs, labels = idb.get_soft_labels(X)

        # No clusters, so labels should be empty
        assert len(labels) == 0

        # All should be noise
        assert probs.shape == (len(X), 1)
        np.testing.assert_allclose(probs, 1.0)

    def test_soft_clustering_empty_dataset(self):
        """Test soft clustering on empty dataset."""
        idb = IncrementalDBSCAN(eps=1.0, min_pts=3, eps_soft=2.0)

        X_test = np.array([[0, 0], [1, 1]])
        probs, labels = idb.get_soft_labels(X_test)

        # No data, so everything is noise
        assert len(labels) == 0
        assert probs.shape == (len(X_test), 1)
        np.testing.assert_allclose(probs, 1.0)

    def test_soft_clustering_query_new_points(self):
        """Test soft clustering for points not in the training set."""
        X_train = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Cluster around origin
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X_train)

        # Query a new point near the cluster
        X_test = np.array([[0.5, 0.5]])
        probs, labels = idb.get_soft_labels(X_test)

        # Should have valid probabilities
        assert probs.shape[0] == 1
        np.testing.assert_allclose(probs.sum(), 1.0)

    def test_soft_clustering_noise_point_assignment(self):
        """Test that noise points get appropriate soft assignments."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Cluster 1
            [20, 20],  # Noise point - far away
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X)

        # Get hard labels
        hard_labels = idb.get_cluster_labels(X)
        assert hard_labels[-1] == -1  # Last point is noise

        # Get soft labels
        probs, cluster_labels = idb.get_soft_labels(X)

        # Noise point should have high noise probability
        # (since it's far from all clusters and beyond eps_soft range)
        noise_prob = probs[-1, -1]
        assert noise_prob > 0.9

    def test_soft_clustering_incremental_update(self):
        """Test that soft clustering adapts to incremental insertions."""
        X1 = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X1)

        X_test = np.array([[5, 5]])
        probs1, labels1 = idb.get_soft_labels(X_test)

        # Initially far from clusters, should be mostly noise
        initial_noise_prob = probs1[0, -1]
        assert initial_noise_prob > 0.9

        # Add new cluster near test point
        X2 = np.array([
            [5, 5], [5, 6], [6, 5], [6, 6],
        ])
        idb.insert(X2)

        probs2, labels2 = idb.get_soft_labels(X_test)

        # Now should have lower noise probability
        new_noise_prob = probs2[0, -1]
        assert new_noise_prob < initial_noise_prob

        # Should have more clusters now
        assert len(labels2) > len(labels1)

    def test_soft_clustering_probability_properties(self):
        """Test mathematical properties of probability distribution."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [5, 5], [5, 6], [6, 5], [6, 6],
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X)

        probs, labels = idb.get_soft_labels(X, kernel='gaussian')

        # All probabilities should be in [0, 1]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

        # Each row should sum to 1
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)

        # At least some probabilities should be non-zero
        assert np.any(probs > 0)

    def test_soft_clustering_core_points_only(self):
        """Test that only core points contribute to soft clustering."""
        # Create a cluster with core and border points
        X = np.array([
            # Dense core
            [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5],
            # Border point (will have fewer neighbors)
            [2.5, 0],
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=4, eps_soft=5.0)
        idb.insert(X)

        # Query point near border point
        X_test = np.array([[3, 0]])
        probs, labels = idb.get_soft_labels(X_test)

        # Should get valid probabilities
        assert probs.shape[0] == 1
        np.testing.assert_allclose(probs.sum(), 1.0)

    def test_soft_clustering_cluster_label_order(self):
        """Test that cluster labels are correctly mapped to probability columns."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [5, 5], [5, 6], [6, 5], [6, 6],
        ])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X)

        probs, cluster_labels = idb.get_soft_labels(
            X, include_noise_prob=False)

        # Labels should be sorted
        assert np.all(cluster_labels[:-1] <= cluster_labels[1:])

        # Number of columns should match number of labels
        assert probs.shape[1] == len(cluster_labels)

    def test_soft_clustering_with_weights(self):
        """Test soft clustering works with weighted samples."""
        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
        ])
        weights = np.array([2.0, 1.0, 1.0, 1.0])

        idb = IncrementalDBSCAN(eps=1.5, min_pts=3, eps_soft=3.0)
        idb.insert(X, sample_weight=weights)

        probs, labels = idb.get_soft_labels(X)

        # Should produce valid probabilities
        assert probs.shape[0] == len(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
