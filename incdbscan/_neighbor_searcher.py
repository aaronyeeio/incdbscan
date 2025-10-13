import numpy as np
from sklearn.neighbors import NearestNeighbors
from sortedcontainers import SortedList


class NeighborSearcher:
    def __init__(self, radius, metric, p):
        self.neighbor_searcher = \
            NearestNeighbors(radius=radius, metric=metric, p=p)
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

    def batch_query_neighbors(self, query_values):
        """Query neighbors for multiple points at once.

        Returns: list of lists, where each inner list contains neighbor IDs
        """
        if len(self.values) == 0:
            return [[] for _ in range(len(query_values))]

        neighbor_indices_array = self.neighbor_searcher.radius_neighbors(
            query_values, return_distance=False)

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
