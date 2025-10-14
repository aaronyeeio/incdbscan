import warnings

import numpy as np

from ._clusters import Clusters
from ._deleter import Deleter
from ._inserter import Inserter
from ._objects import Objects
from ._soft_clustering import SoftClusteringCache
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

    nearest_neighbors : string, optional (default='torch_cuda')
        Backend for nearest neighbor search. Options:
        - 'torch_cuda': PyTorch with CUDA (GPU acceleration)
        - 'torch_cpu': PyTorch with CPU
        - 'sklearn': scikit-learn NearestNeighbors

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

    def __init__(self, eps=1, min_pts=5, eps_merge=None, metric='minkowski', p=2, nearest_neighbors='torch_cuda', eps_soft=None):
        self.eps = eps
        self.eps_merge = eps_merge if eps_merge is not None else eps
        self.eps_soft = eps_soft if eps_soft is not None else 2 * eps
        self.min_pts = min_pts
        self.metric = metric
        self.p = p
        self.nearest_neighbors = nearest_neighbors

        if self.eps_merge > self.eps:
            raise ValueError("eps_merge must be <= eps")

        self.clusters = Clusters()
        self._objects = Objects(self.eps, self.eps_merge, self.min_pts,
                                self.metric, self.p, self.clusters, self.eps_soft, self.nearest_neighbors)
        self._inserter = Inserter(self.eps, self.eps_merge, self.min_pts,
                                  self._objects)
        self._deleter = Deleter(self.eps, self.eps_merge, self.min_pts,
                                self._objects)

        # Soft clustering cache for fast probabilistic queries
        self._soft_clustering_cache = SoftClusteringCache(
            self._objects, self.clusters)

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

        # Incrementally update soft clustering cache
        self._soft_clustering_cache.update_after_insert()

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
            # Collect deleted object IDs
            deleted_ids = [obj.id for obj in objects_to_delete]

            self._deleter.batch_delete(objects_to_delete, weights)

            # Incrementally update soft clustering cache
            self._soft_clustering_cache.update_after_delete(deleted_ids)

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

        return self._soft_clustering_cache.get_soft_labels(
            X, self.eps_soft, kernel=kernel, include_noise_prob=include_noise_prob
        )


class IncrementalDBSCANWarning(Warning):
    pass
