import numpy as np
from sortedcontainers import SortedList


class NeighborSearcher:
    def __init__(self, metric, p, nearest_neighbors='torch_cuda'):
        """Initialize NeighborSearcher with specified backend.

        Args:
            metric: Distance metric to use
            p: Power parameter for Minkowski metric
            nearest_neighbors: Backend to use. Options:
                - 'torch_cuda': PyTorch with CUDA (GPU)
                - 'torch_cpu': PyTorch with CPU
                - 'sklearn': sklearn NearestNeighbors
        """
        self.backend = nearest_neighbors

        if nearest_neighbors in ['torch_cuda', 'torch_cpu']:
            from ._nearest_neighbors_torch import NearestNeighbors
            device = 'cuda' if nearest_neighbors == 'torch_cuda' else 'cpu'
            self.neighbor_searcher = NearestNeighbors(
                metric=metric, p=p, device=device)
        elif nearest_neighbors == 'sklearn':
            from sklearn.neighbors import NearestNeighbors
            self.neighbor_searcher = NearestNeighbors(
                metric=metric, p=p, n_jobs=-1)
        else:
            raise ValueError(
                f"nearest_neighbors must be one of: 'torch_cuda', 'torch_cpu', 'sklearn', "
                f"got '{nearest_neighbors}'")

        self.values = np.array([])
        self.ids = SortedList()

    def batch_insert(self, new_values, new_ids):
        """Insert multiple values at once and rebuild tree only once."""
        if not new_ids:
            return

        # Build a map from old IDs to their values
        old_id_to_value = {}
        for i, old_id in enumerate(self.ids):
            old_id_to_value[old_id] = self.values[i]

        # Add all new IDs to SortedList
        for new_id in new_ids:
            self.ids.add(new_id)

        # Build a map from new IDs to their values
        new_id_to_value = dict(zip(new_ids, new_values))

        # Rebuild the values array in the new sorted order
        new_values_list = []
        for id_ in self.ids:
            if id_ in new_id_to_value:
                new_values_list.append(new_id_to_value[id_])
            else:
                new_values_list.append(old_id_to_value[id_])

        if new_values_list:
            self.values = np.array(new_values_list).reshape(
                len(new_values_list), -1)
            self.neighbor_searcher = self.neighbor_searcher.fit(self.values)
        else:
            self.values = np.array([])

    def _insert_into_array(self, new_value, position):
        extended = np.insert(self.values, position, new_value, axis=0)
        if not self.values.size:
            extended = extended.reshape(1, -1)
        self.values = extended

    def batch_query_neighbors(self, query_values, radius):
        """Query neighbors for multiple points at once.

        Args:
            query_values: values to query
            radius: search radius

        Returns: list of lists, where each inner list contains neighbor IDs
        """
        if len(self.values) == 0:
            return [[] for _ in range(len(query_values))]

        neighbor_indices_array = self.neighbor_searcher.radius_neighbors(
            query_values, radius=radius, return_distance=False)

        result = []
        for neighbor_indices in neighbor_indices_array:
            result.append([self.ids[ix] for ix in neighbor_indices])
        return result

    def batch_delete(self, ids_to_delete):
        """Delete multiple IDs at once and rebuild tree only once."""
        # Sort positions in descending order to delete from back to front
        positions_to_delete = []
        for id_ in ids_to_delete:
            if id_ in self.ids:
                positions_to_delete.append(self.ids.index(id_))

        positions_to_delete.sort(reverse=True)

        # Delete from back to front to maintain index validity
        for position in positions_to_delete:
            del self.ids[position]

        # Delete all positions from numpy array at once
        if positions_to_delete:
            # Re-sort in ascending order for numpy delete
            positions_to_delete.sort()
            self.values = np.delete(self.values, positions_to_delete, axis=0)

            # Rebuild the tree if there are still values
            if len(self.values) > 0:
                self.neighbor_searcher = self.neighbor_searcher.fit(
                    self.values)
