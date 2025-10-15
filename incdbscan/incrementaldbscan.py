import warnings

import numpy as np
import rustworkx as rx

from ._clusters import Clusters, CLUSTER_LABEL_NOISE
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

    min_cluster_size : int or float, optional (default=1)
        The minimum total weight sum that a cluster needs to have. Clusters with
        total weight less than this value will be dissolved and marked as noise.

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

    def __init__(self, eps=1, min_pts=5, min_cluster_size=1, metric='minkowski', p=2, nearest_neighbors='torch_cuda', eps_soft=None):
        self.eps = eps
        self.eps_soft = eps_soft if eps_soft is not None else 2 * eps
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.p = p
        self.nearest_neighbors = nearest_neighbors

        self.clusters = Clusters()
        self._objects = Objects(self.eps, self.min_pts,
                                self.metric, self.p, self.clusters, self.eps_soft, self.nearest_neighbors)
        self._inserter = Inserter(self.eps, self.min_pts,
                                  self.min_cluster_size, self._objects)
        self._deleter = Deleter(self.eps, self.min_pts,
                                self.min_cluster_size, self._objects)

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

    def get_soft_labels(self, X, kernel='gaussian', include_noise_prob=True, target_clusters=None):
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

        target_clusters : array-like or None, optional (default=None)
            Specific cluster labels to compute probabilities for. If None,
            computes for all active clusters.

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
            X, self.eps_soft, kernel=kernel, include_noise_prob=include_noise_prob,
            target_clusters=target_clusters
        )

    def evict_from_cluster_edge(self, cluster_label, n):
        """Evict n points from cluster edge without causing cluster split.

        This method removes up to n points from the cluster's boundary in a way
        that guarantees the cluster remains connected. It prioritizes border
        points (non-core) first, then uses articulation point detection to
        safely remove core points that don't maintain cluster connectivity.

        Parameters
        ----------
        cluster_label : int
            The label of the cluster to evict points from

        n : int
            Maximum number of points to evict

        Returns
        -------
        evicted_count : int
            Actual number of points evicted (may be less than n if not enough
            non-critical points exist)

        Notes
        -----
        Evicted points are marked as noise (label=-1). The algorithm guarantees
        that the cluster will not split into disconnected components.

        Time complexity: O(V + E) where V is the number of core points and E
        is the number of edges between core points.
        """
        cluster = self.clusters.get_cluster(cluster_label)
        if not cluster or cluster.size == 0:
            return 0

        # Find points that can be safely evicted
        evict_candidates = self._find_evictable_points(cluster, n)

        # Mark evicted points as noise
        for obj in evict_candidates:
            self.clusters.set_label(obj, CLUSTER_LABEL_NOISE)

        return len(evict_candidates)

    def _find_max_evictable_with_stats(self, cluster, candidates, max_n):
        """Find maximum number of candidates that can be evicted while preserving cluster.

        Uses cumulative statistics to avoid recomputation. Time complexity: O(k) where
        k = min(max_n, len(candidates)).

        Parameters
        ----------
        cluster : Cluster
            The cluster being processed
        candidates : list
            Candidate points for eviction, ordered by priority
        max_n : int
            Maximum number to try evicting

        Returns
        -------
        list
            Evictable points (may be empty if none can be evicted)
        """
        if not candidates:
            return []

        k = min(max_n, len(candidates))

        # Precompute cumulative statistics for candidates
        cumulative_cores = []
        cumulative_weight = []

        cores_so_far = 0
        weight_so_far = 0.0

        for obj in candidates:
            cores_so_far += 1 if obj.is_core else 0
            weight_so_far += obj.weight
            cumulative_cores.append(cores_so_far)
            cumulative_weight.append(weight_so_far)

        # Find maximum evictable count by checking from k down to 0
        # remaining_cores = cluster.core_count - cumulative_cores[i-1]
        # remaining_weight = cluster.total_weight - cumulative_weight[i-1]
        for i in range(k, 0, -1):
            evict_cores = cumulative_cores[i - 1]
            evict_weight = cumulative_weight[i - 1]

            remaining_cores = cluster.core_count - evict_cores
            remaining_weight = cluster.total_weight - evict_weight

            if remaining_cores > 0 and remaining_weight >= self.min_cluster_size:
                return candidates[:i]

        return []

    def _find_evictable_points(self, cluster, n):
        """Find points to evict while ensuring a single cluster remains.

        Strategy:
        1. If n points can be evicted without touching articulation points,
           evict those n points from the periphery.
        2. If n points would require removing articulation points (causing split):
           - Identify which components would result from removing safe points
           - Keep only the largest/densest component
           - Evict all points not in that component to maintain single cluster
        3. If eviction would destroy the cluster entirely (no cores left),
           return empty list to preserve cluster.

        Points are prioritized by "peripherality":
        - Border points (non-core) sorted by neighbor_count (ascending)
        - Non-articulation core points sorted by neighbor_count (ascending)

        Parameters
        ----------
        cluster : Cluster
            The cluster to analyze

        n : int
            Target number of points to evict

        Returns
        -------
        evict_candidates : list
            List of Object instances to evict. May exceed n if the cluster
            would split (keeps only densest component). May be less than n
            or empty if evicting would destroy the cluster.
        """
        # Step 1: Collect and sort border points by peripherality
        border_points = [obj for obj in cluster.members if not obj.is_core]
        border_points.sort(key=lambda obj: obj.neighbor_count)

        # Step 2: Handle core points
        cores = [obj for obj in cluster.members if obj.is_core]

        if len(cores) <= 1:
            # Single core or no cores - can safely evict all border points
            # and the single core if needed
            all_evictable = border_points + cores
            return all_evictable[:n]

        # Step 3: Detect articulation points among core points
        core_node_ids = [obj.node_id for obj in cores]
        subgraph = self._objects.graph.subgraph(core_node_ids)
        subgraph_articulation_indices = rx.articulation_points(subgraph)

        # Map articulation points back to original objects
        articulation_node_ids = set()
        for subgraph_idx in subgraph_articulation_indices:
            original_node_id = subgraph[subgraph_idx].node_id
            articulation_node_ids.add(original_node_id)

        articulation_cores = [
            obj for obj in cores if obj.node_id in articulation_node_ids
        ]
        non_critical_cores = [
            obj for obj in cores if obj.node_id not in articulation_node_ids
        ]

        # Sort non-critical cores by peripherality
        non_critical_cores.sort(key=lambda obj: obj.neighbor_count)

        # Step 4: Try evicting n points without touching articulation points
        safe_evictable = border_points + non_critical_cores

        if len(safe_evictable) >= n:
            # Use cumulative stats to find maximum evictable
            return self._find_max_evictable_with_stats(cluster, safe_evictable, n)

        # Step 5: Would need to remove articulation points to reach n
        # This would cause split into multiple components
        # Strategy: Keep the largest/densest component, evict everything else

        # Case 5a: No articulation points (all cores non-critical)
        if not articulation_cores:
            return self._find_max_evictable_with_stats(cluster, safe_evictable, len(safe_evictable))

        # Case 5b: Check what happens if we remove all safe points
        remaining_cores = set(cores) - set(safe_evictable)

        if len(remaining_cores) <= 1:
            # Would leave ≤1 core - try to evict as many safe points as possible
            return self._find_max_evictable_with_stats(cluster, safe_evictable, len(safe_evictable))

        # Case 5c: Check if remaining cores are still connected
        remaining_core_node_ids = [obj.node_id for obj in remaining_cores]
        remaining_subgraph = self._objects.graph.subgraph(
            remaining_core_node_ids)
        components_node_ids = rx.connected_components(remaining_subgraph)

        if len(components_node_ids) == 1:
            # Still connected - evict all safe points if possible
            return self._find_max_evictable_with_stats(cluster, safe_evictable, len(safe_evictable))

        # Case 5d: Multiple components - keep only the densest one
        return self._shrink_to_best_component(
            cluster, components_node_ids, remaining_subgraph,
            non_critical_cores, border_points
        )

    def _shrink_to_best_component(self, cluster, components_node_ids,
                                  remaining_subgraph, non_critical_cores, border_points):
        """Shrink cluster to best component when split would occur.

        Parameters
        ----------
        cluster : Cluster
            The cluster being processed
        components_node_ids : list
            Connected components as lists of node IDs
        remaining_subgraph : PyGraph
            Subgraph of remaining cores
        non_critical_cores : list
            Non-articulation core points
        border_points : list
            Border points

        Returns
        -------
        list
            Points to evict (everything except best component)
        """
        # Map components from node IDs to objects
        components_objects = []
        for component_node_ids in components_node_ids:
            component_cores = {
                self._objects.graph[remaining_subgraph[subgraph_node_id].node_id]
                for subgraph_node_id in component_node_ids
            }
            components_objects.append(component_cores)

        # Select densest component (highest total neighbor_count)
        best_core_component = max(
            components_objects,
            key=lambda comp: sum(obj.neighbor_count for obj in comp)
        )

        # Expand component: add connected non-critical cores
        core_component = set(best_core_component)
        for obj in non_critical_cores:
            if any(neighbor in core_component for neighbor in obj.neighbors if neighbor.is_core):
                core_component.add(obj)

        # Expand component: add connected border points
        border_component = {
            obj for obj in border_points
            if any(neighbor in core_component for neighbor in obj.neighbors)
        }

        final_component = core_component | border_component

        # Verify final component forms a valid cluster using O(k) statistics
        component_cores = sum(1 for obj in final_component if obj.is_core)
        component_weight = sum(obj.weight for obj in final_component)

        if component_cores == 0 or component_weight < self.min_cluster_size:
            # Component too small - don't evict anything
            return []

        # Evict everything except the final component
        return [obj for obj in cluster.members if obj not in final_component]


class IncrementalDBSCANWarning(Warning):
    pass
