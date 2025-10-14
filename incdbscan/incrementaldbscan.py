import warnings

import numpy as np

from ._clusters import Clusters
from ._deleter import Deleter
from ._inserter import Inserter
from ._objects import Objects
from ._utils import input_check


class IncrementalDBSCAN:
    """The incremental version of DBSCAN, a density-based clustering algorithm
    that handles outliers.

    After an initial clustering of an initial object set (i.e., set of data
    points), the object set can at any time be updated by increments of any
    size. An increment can be either the insertion or the deletion of objects.

    After each update, the result of the clustering is the same as if the
    updated object set (i.e., the initial object set modified by all
    increments) was clustered by DBSCAN. However, this result is reached by
    using information from the previous state of the clustering, and without
    the need of applying DBSCAN to the whole updated object set.

    Parameters
    ----------
    eps : float, optional (default=0.5)
        The radius of neighborhood calculation. An object is the neighbor of
        another if the distance between them is no more than eps.

    min_pts : int or float, optional (default=1)
        The minimum sum of neighbor weights that an object needs to have to be a
        core object of a cluster.

    eps_merge : float, optional (default=None)
        The radius for determining connectivity between core objects for cluster
        merging. Must be <= eps. If None, defaults to eps (standard DBSCAN).
        Using eps_merge < eps implements two-level density: border points can
        connect to clusters using eps, but clusters only merge when core points
        are within eps_merge of each other.

    metric : string or callable, optional (default='minkowski')
        The distance metric to use to calculate distance between data objects.
        Accepts metrics that are accepted by scikit-learn's NearestNeighbors
        class, excluding 'precomputed'. The default is 'minkowski', which is
        equivalent to the Euclidean distance if p=2.

    p : float or int, optional (default=2)
        Parameter for Minkowski distance if metric='minkowski'.

    eps_soft : float, optional (default=None)
        The radius for soft clustering queries. If None, defaults to 2*eps.
        Typically set larger than eps to capture broader context for 
        probabilistic cluster assignment.

    References
    ----------
    Ester et al. 1998. Incremental Clustering for Mining in a Data Warehousing
    Environment. In: Proceedings of the 24th International Conference on Very
    Large Data Bases (VLDB 1998).

    """

    def __init__(self, eps=1, min_pts=5, eps_merge=None, metric='minkowski', p=2, eps_soft=None):
        self.eps = eps
        self.eps_merge = eps_merge if eps_merge is not None else eps
        self.eps_soft = eps_soft if eps_soft is not None else 2 * eps
        self.min_pts = min_pts
        self.metric = metric
        self.p = p

        if self.eps_merge > self.eps:
            raise ValueError("eps_merge must be <= eps")

        self.clusters = Clusters()
        self._objects = Objects(self.eps, self.eps_merge, self.min_pts,
                                self.metric, self.p, self.clusters, self.eps_soft)
        self._inserter = Inserter(self.eps, self.eps_merge, self.min_pts,
                                  self._objects)
        self._deleter = Deleter(self.eps, self.eps_merge, self.min_pts,
                                self._objects)

    def insert(self, X, sample_weight=None):
        """Insert objects into the object set, then update clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to be inserted into the object set.

        sample_weight : array-like of shape (n_samples,), optional (default=None)
            Weight of each sample. If None, all samples have weight 1.

        Returns
        -------
        self

        """
        X = input_check(X)

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        else:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != len(X):
                raise ValueError(
                    f'sample_weight has {len(sample_weight)} elements, '
                    f'but X has {len(X)} samples.'
                )

        # Use batch insertion for all cases
        self._inserter.batch_insert(X, sample_weight)

        return self

    def delete(self, X, sample_weight=None):
        """Delete objects from object set, then update clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to be deleted from the object set.

        sample_weight : array-like of shape (n_samples,), optional (default=None)
            Weight of each sample. If None, all samples have weight 1.

        Returns
        -------
        self

        """
        X = input_check(X)

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        else:
            sample_weight = np.asarray(sample_weight)
            if len(sample_weight) != len(X):
                raise ValueError(
                    f'sample_weight has {len(sample_weight)} elements, '
                    f'but X has {len(X)} samples.'
                )

        objects_to_delete = []
        weights = []
        for ix, value in enumerate(X):
            obj = self._objects.get_object(value)
            if obj:
                objects_to_delete.append(obj)
                weights.append(sample_weight[ix])
            else:
                warnings.warn(IncrementalDBSCANWarning(
                    f'Object at position {ix} was not deleted because '
                    'there is no such object in the object set.'))

        if objects_to_delete:
            self._deleter.batch_delete(objects_to_delete, weights)

        return self

    def get_cluster_labels(self, X):
        """Get cluster labels of objects.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to get labels for.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
                 Cluster labels. Effective labels start from 0. -1 means the
                 object is noise. numpy.nan means the object was not in the
                 object set.

        """
        X = input_check(X)

        labels = np.zeros(len(X))

        for ix, value in enumerate(X):
            obj = self._objects.get_object(value)

            if obj:
                label = self.clusters.get_label(obj)

            else:
                label = np.nan
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'No label was retrieved for object at position {ix} '
                        'because there is no such object in the object set.'
                    )
                )

            labels[ix] = label

        return labels

    def get_all_clusters(self):
        """Get all active clusters.

        Returns
        -------
        clusters : list of Cluster
                   List of all active Cluster instances
        """
        return self.clusters.get_all_clusters()

    def get_cluster(self, label):
        """Get a specific cluster by label.

        Parameters
        ----------
        label : int
            The cluster label

        Returns
        -------
        cluster : Cluster or None
                  The Cluster instance, or None if not found
        """
        return self.clusters.get_cluster(label)

    def get_cluster_statistics(self):
        """Get statistics for all clusters.

        Returns
        -------
        stats : dict
                Dictionary containing cluster statistics
        """
        return self.clusters.get_statistics()

    def get_soft_labels(self, X, kernel='gaussian', include_noise_prob=True):
        """Get soft cluster assignment probabilities for data points.

        For each point, computes membership probability to each cluster based on
        distance-weighted voting from nearby core points within eps_soft radius.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points to get soft labels for

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
        X = input_check(X)
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
        label_to_col = {label: i for i, label in enumerate(cluster_labels)}

        # Check if we have any objects
        if len(self._objects.neighbor_searcher.values) == 0:
            # No objects in dataset
            if include_noise_prob:
                return np.ones((n_samples, 1)), cluster_labels
            else:
                return np.zeros((n_samples, 0)), cluster_labels

        # Query neighbors with distances using neighbor_searcher
        distances, neighbor_indices = \
            self._objects.neighbor_searcher.neighbor_searcher.radius_neighbors(
                X, radius=self.eps_soft, return_distance=True
            )

        # Compute weights using kernel function
        sigma = self.eps_soft / 3.0  # For gaussian kernel

        def compute_weight(dist):
            if kernel == 'gaussian':
                return np.exp(-dist**2 / (2 * sigma**2))
            elif kernel == 'inverse':
                return 1.0 / (1.0 + dist / self.eps_soft)
            elif kernel == 'linear':
                return np.maximum(0, 1.0 - dist / self.eps_soft)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

        # Initialize probability matrix
        if include_noise_prob:
            probs = np.zeros((n_samples, n_clusters + 1))
        else:
            probs = np.zeros((n_samples, n_clusters))

        # For each query point
        for i in range(n_samples):
            indices = neighbor_indices[i]
            dists = distances[i]

            # Collect core neighbors and their weights
            cluster_weights = np.zeros(n_clusters)

            for idx, dist in zip(indices, dists):
                obj_id = self._objects.neighbor_searcher.ids[int(idx)]
                obj = self._objects._get_object_from_object_id(obj_id)

                # Only consider core points
                if not obj.is_core:
                    continue

                label = self.clusters.get_label(obj)

                # Only consider clustered core points
                if label < 0:  # NOISE or UNCLASSIFIED
                    continue

                # Compute distance weight
                weight = compute_weight(dist)

                # Add weight to corresponding cluster
                col_idx = label_to_col[label]
                cluster_weights[col_idx] += weight

            # Normalize to probabilities
            total_weight = cluster_weights.sum()

            if total_weight > 0:
                probs[i, :n_clusters] = cluster_weights / total_weight
                if include_noise_prob:
                    # No noise probability if assigned to cluster
                    probs[i, -1] = 0.0
            else:
                # No core neighbors found - remains noise
                if include_noise_prob:
                    probs[i, -1] = 1.0
                # else: all zeros (no cluster membership)

        return probs, cluster_labels


class IncrementalDBSCANWarning(Warning):
    pass
