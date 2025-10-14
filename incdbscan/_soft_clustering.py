"""Soft clustering functionality for Incremental DBSCAN.

This module provides efficient soft cluster assignment by maintaining
an incremental cache of core point labels.
"""
import numpy as np


class SoftClusteringCache:
    """Maintains incremental cache for fast soft clustering queries.

    The cache uses a two-level structure:
    1. _core_label_cache: object_id -> cluster_label (stable across index changes)
    2. _index_to_label: array mapping neighbor_searcher indices to labels (fast lookup)

    The cache is incrementally updated after insert/delete operations to maintain
    synchronization with the clustering state.
    """

    def __init__(self, objects, clusters):
        """Initialize the soft clustering cache.

        Parameters
        ----------
        objects : Objects
            The Objects instance managing the data points
        clusters : Clusters
            The Clusters instance managing cluster assignments
        """
        self.objects = objects
        self.clusters = clusters

        # Cache for soft clustering: maps object_id to cluster label (only for clustered core points)
        self._core_label_cache = {}
        self._index_to_label = np.array([], dtype=np.int32)  # Initialize empty

    def update_after_insert(self):
        """Incrementally update cache after insertion by scanning only affected objects."""
        # Scan all objects to update cache for those that became core and got clustered
        # This is simpler than tracking changes, and only runs after insert
        for obj_id in self.objects.neighbor_searcher.ids:
            obj = self.objects._get_object_from_object_id(obj_id)

            if obj.is_core:
                label = self.clusters.get_label(obj)
                if label >= 0:
                    # Add or update cache entry
                    self._core_label_cache[obj_id] = label
                elif obj_id in self._core_label_cache:
                    # Object lost its cluster (became noise)
                    del self._core_label_cache[obj_id]
            elif obj_id in self._core_label_cache:
                # Object lost core status
                del self._core_label_cache[obj_id]

        # Build fast index-to-label lookup array
        self._build_index_to_label_array()

    def update_after_delete(self, deleted_object_ids):
        """Incrementally update cache after deletion by removing invalidated entries.

        Parameters
        ----------
        deleted_object_ids : list
            Object IDs that were deleted
        """
        # Remove deleted objects from cache
        for obj_id in deleted_object_ids:
            self._core_label_cache.pop(obj_id, None)

        # After deletion, some objects may have lost core status or changed labels
        # Remove invalidated entries
        ids_to_remove = []
        for obj_id in self._core_label_cache:
            obj = self.objects._get_object_from_object_id(obj_id)
            if obj is None or not obj.is_core:
                ids_to_remove.append(obj_id)
            else:
                label = self.clusters.get_label(obj)
                if label < 0:
                    ids_to_remove.append(obj_id)
                else:
                    # Update label if it changed
                    self._core_label_cache[obj_id] = label

        for obj_id in ids_to_remove:
            del self._core_label_cache[obj_id]

        # Rebuild fast index-to-label lookup array
        self._build_index_to_label_array()

    def _build_index_to_label_array(self):
        """Build fast numpy array mapping neighbor_searcher index to cluster label."""
        n_objects = len(self.objects.neighbor_searcher.ids)
        # Use -1 for non-core or noise points
        self._index_to_label = np.full(n_objects, -1, dtype=np.int32)

        # Map each index to its label if it's a cached core point
        ids_list = self.objects.neighbor_searcher.ids
        for idx in range(n_objects):
            obj_id = ids_list[idx]
            if obj_id in self._core_label_cache:
                self._index_to_label[idx] = self._core_label_cache[obj_id]

    def get_soft_labels(self, X, eps_soft, kernel='gaussian', include_noise_prob=True):
        """Get soft cluster assignment probabilities for data points.

        For each point, computes membership probability to each cluster based on
        distance-weighted voting from nearby core points within eps_soft radius.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Query points to get soft labels for (already validated)

        eps_soft : float
            The radius for soft clustering queries

        kernel : str, optional (default='gaussian')
            Distance weighting kernel. Options:
            - 'gaussian': exp(-d^2 / (2*sigma^2)) where sigma = eps_soft/3
            - 'inverse': 1 / (1 + d/eps_soft)
            - 'linear': max(0, 1 - d/eps_soft)

        include_noise_prob : bool, optional (default=True)
            If True, includes probability of remaining noise as last column

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_clusters) or
                        (n_samples, n_clusters+1)
            Soft cluster membership probabilities. If include_noise_prob=True,
            last column is noise probability. Rows sum to 1.0.

        cluster_labels : ndarray of shape (n_clusters,)
            Cluster labels corresponding to columns (excluding noise column)
        """
        n_samples = len(X)

        # Get active cluster labels
        active_clusters = self.clusters.get_all_clusters()
        if not active_clusters:
            # No clusters exist
            if include_noise_prob:
                return np.ones((n_samples, 1)), np.array([])
            else:
                return np.zeros((n_samples, 0)), np.array([])

        cluster_labels = np.array(sorted([c.label for c in active_clusters]))
        n_clusters = len(cluster_labels)

        # Build fast label-to-column mapping array
        max_label = cluster_labels.max() if len(cluster_labels) > 0 else 0
        label_to_col_array = np.full(max_label + 1, -1, dtype=np.int32)
        for i, label in enumerate(cluster_labels):
            label_to_col_array[label] = i

        # Check if we have any objects
        if len(self.objects.neighbor_searcher.values) == 0:
            # No objects in dataset
            if include_noise_prob:
                return np.ones((n_samples, 1)), cluster_labels
            else:
                return np.zeros((n_samples, 0)), cluster_labels

        # Query neighbors with distances using neighbor_searcher
        distances, neighbor_indices = \
            self.objects.neighbor_searcher.neighbor_searcher.radius_neighbors(
                X, radius=eps_soft, return_distance=True
            )

        # Vectorized weight computation
        sigma = eps_soft / 3.0

        # Initialize probability matrix
        if include_noise_prob:
            probs = np.zeros((n_samples, n_clusters + 1))
        else:
            probs = np.zeros((n_samples, n_clusters))

        # For each query point
        for i in range(n_samples):
            indices = neighbor_indices[i]
            dists = distances[i]

            if len(indices) == 0:
                if include_noise_prob:
                    probs[i, -1] = 1.0
                continue

            # Fast lookup: get labels directly from index array
            neighbor_labels = self._index_to_label[indices]

            # Filter to core points (label >= 0)
            core_mask = neighbor_labels >= 0

            if not core_mask.any():
                if include_noise_prob:
                    probs[i, -1] = 1.0
                continue

            core_labels = neighbor_labels[core_mask]
            core_dists = dists[core_mask]

            # Vectorized weight computation
            if kernel == 'gaussian':
                weights = np.exp(-core_dists**2 / (2 * sigma**2))
            elif kernel == 'inverse':
                weights = 1.0 / (1.0 + core_dists / eps_soft)
            elif kernel == 'linear':
                weights = np.maximum(0, 1.0 - core_dists / eps_soft)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            # Map labels to column indices using array lookup
            col_indices = label_to_col_array[core_labels]

            # Accumulate weights per cluster using bincount
            cluster_weights = np.bincount(
                col_indices, weights=weights, minlength=n_clusters)

            # Normalize to probabilities
            total_weight = cluster_weights.sum()

            if total_weight > 0:
                probs[i, :n_clusters] = cluster_weights / total_weight
                if include_noise_prob:
                    probs[i, -1] = 0.0
            else:
                if include_noise_prob:
                    probs[i, -1] = 1.0

        return probs, cluster_labels
