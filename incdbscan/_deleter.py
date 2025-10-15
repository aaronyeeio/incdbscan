from collections import defaultdict

import rustworkx as rx

from ._bfscomponentfinder import BFSComponentFinder
from ._clusters import CLUSTER_LABEL_NOISE


class Deleter:
    def __init__(self, eps, min_pts, min_cluster_size, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.objects = objects
        self.clusters = objects.clusters  # Shorthand for clusters

    def batch_delete(self, ids_to_delete):
        """Delete multiple objects at once and update clustering.

        Args:
            ids_to_delete: list of object IDs to delete
        """
        # Get objects from IDs first to collect affected objects
        objects_to_delete = []
        for object_id in ids_to_delete:
            obj = self.objects.get_object_by_id(object_id)
            if obj is None:
                raise ValueError(f"Object with ID {object_id} not found")
            objects_to_delete.append(obj)

        # Collect affected objects before deletion (for core status update later)
        affected_objects_for_stats = set()
        for obj in objects_to_delete:
            affected_objects_for_stats.update(obj.neighbors)

        # Phase 1: Batch delete objects and get state information
        was_core_map, objects_removed = \
            self.objects.batch_delete_objects(ids_to_delete)

        # Phase 2: Identify all objects that lost their core property
        ex_cores = self._get_objects_that_lost_core_property_batch(
            objects_to_delete, was_core_map)

        # Phase 3: Get update seeds and non-core neighbors
        update_seeds, non_core_neighbors_of_ex_cores = \
            self._get_update_seeds_and_non_core_neighbors_of_ex_cores_batch(
                ex_cores, objects_removed)

        # Phase 4: Handle cluster splits if needed
        if update_seeds:
            # Only for update seeds belonging to the same cluster do we
            # have to consider if split is needed.

            update_seeds_by_cluster = \
                self._group_objects_by_cluster(update_seeds)

            for original_label, seeds in update_seeds_by_cluster.items():
                components = list(self._find_components_to_split_away(seeds))

                if not components:
                    continue

                # Step 1: Allocate new labels for all components
                base_label = self.clusters.get_next_cluster_label()
                new_components = []

                for i, component in enumerate(components):
                    new_label = base_label + i

                    # Assign labels internally to all objects in this component
                    for obj in component:
                        self.clusters._assign_label_internal(obj, new_label)

                    new_components.append((new_label, component))

                # Step 2: Trigger split hook to create clusters properly
                self.clusters.split_cluster(original_label, new_components)

                # Step 3: Clean up split clusters that have no core points
                # After split, some clusters may only have border points if all cores lost core status
                for new_label, members in new_components:
                    has_core = any(obj.is_core for obj in members)
                    if not has_core:
                        # Convert all members to noise since there are no core points
                        for obj in list(members):
                            self.clusters.set_label(obj, CLUSTER_LABEL_NOISE)

        # Phase 5: Update labels of border objects that were in the neighborhood
        # of objects that lost their core property. They become either borders
        # of other clusters or noise.
        self._set_each_border_object_labels_to_largest_around(
            non_core_neighbors_of_ex_cores)

        # Phase 6: Update core status for all affected objects
        # After deletions and label reassignments, ensure cluster statistics reflect current core status.
        # This is done once here (instead of before splits) to cover all cases comprehensively.
        affected_objects_for_stats.difference_update(objects_removed)
        self.clusters.update_core_status_for_objects(
            affected_objects_for_stats)

        # Phase 7: Final cleanup - remove clusters that have no core points
        # After all reassignments, some clusters may only have border points.
        # This happens when border points are assigned to clusters whose cores were all deleted.
        clusters_to_check = list(self.clusters.get_all_clusters())
        for cluster in clusters_to_check:
            has_core = any(obj.is_core for obj in cluster.members)
            if not has_core and cluster.size > 0:
                # No core points means this is not a valid cluster by DBSCAN definition
                for obj in list(cluster.members):
                    self.clusters.set_label(obj, CLUSTER_LABEL_NOISE)

        # Phase 8: Dissolve clusters that are smaller than min_cluster_size
        self.clusters.dissolve_small_clusters(self.min_cluster_size)

    def _get_objects_that_lost_core_property_batch(self, objects_deleted, was_core_map):
        """Identify all objects that lost their core property during batch deletion.

        Args:
            objects_deleted: list of objects that were deleted
            was_core_map: dict mapping object to whether it was core before deletion

        Returns:
            set of objects that lost their core property
        """
        ex_cores = set()
        affected_neighbors = set()

        # Collect all affected neighbors
        for obj in objects_deleted:
            affected_neighbors.update(obj.neighbors)

        # Check each affected neighbor to see if it lost core property
        for neighbor in affected_neighbors:
            # Skip if already removed
            if neighbor.id not in self.objects._object_id_to_node_id:
                continue

            # Check if this neighbor lost core property
            # Current state: not core, but was it core before?
            if not neighbor.is_core:
                # Calculate what the neighbor_count was before all deletions
                # We need to add back the weights from all deleted neighbors
                weight_lost = sum(
                    obj.weight
                    for obj in objects_deleted
                    if neighbor in obj.neighbors
                )
                previous_neighbor_count = neighbor.neighbor_count + weight_lost

                # Was it core before?
                if previous_neighbor_count >= self.min_pts:
                    ex_cores.add(neighbor)

        # Add deleted objects that were core
        for obj in objects_deleted:
            if was_core_map.get(obj, False):
                ex_cores.add(obj)

        return ex_cores

    def _get_update_seeds_and_non_core_neighbors_of_ex_cores_batch(
            self, ex_cores, objects_removed):
        """Get update seeds and non-core neighbors for batch deletion.

        Args:
            ex_cores: set of objects that lost their core property
            objects_removed: list of objects that were completely removed (weight <= 0)

        Returns:
            update_seeds: set of core neighbors of ex-cores
            non_core_neighbors_of_ex_cores: set of non-core neighbors of ex-cores
        """
        update_seeds = set()
        non_core_neighbors_of_ex_cores = set()

        objects_removed_set = set(objects_removed)

        for ex_core in ex_cores:
            for neighbor in ex_core.neighbors:
                if neighbor.is_core:
                    update_seeds.add(neighbor)
                else:
                    non_core_neighbors_of_ex_cores.add(neighbor)

        # Remove objects that were completely deleted
        update_seeds = update_seeds.difference(objects_removed_set)
        non_core_neighbors_of_ex_cores = \
            non_core_neighbors_of_ex_cores.difference(objects_removed_set)

        return update_seeds, non_core_neighbors_of_ex_cores

    def _group_objects_by_cluster(self, objects):
        grouped_objects = defaultdict(list)

        for obj in objects:
            label = self.clusters.get_label(obj)
            grouped_objects[label].append(obj)

        return grouped_objects

    def _find_components_to_split_away(self, seed_objects):
        if len(seed_objects) == 1:
            return []

        if self._objects_are_neighbors_of_each_other(seed_objects):
            return []

        seed_node_ids = [obj.node_id for obj in seed_objects]
        finder = BFSComponentFinder(self.objects.graph)
        rx.bfs_search(self.objects.graph, seed_node_ids, finder)

        seed_of_largest, size_of_largest = 0, 0
        for seed_id, component in finder.seed_to_component.items():
            component_size = len(component)
            if component_size > size_of_largest:
                size_of_largest = component_size
                seed_of_largest = seed_id

        for seed_id, component in finder.seed_to_component.items():
            if seed_id != seed_of_largest:
                yield component

    @staticmethod
    def _objects_are_neighbors_of_each_other(objects):
        for obj1 in objects:
            for obj2 in objects:
                if obj2 not in obj1.neighbors:
                    return False
        return True

    def _set_each_border_object_labels_to_largest_around(self, objects_to_set):
        cluster_updates = {}

        for obj in objects_to_set:
            labels = self._get_cluster_labels_in_neighborhood(obj)
            if not labels:
                labels.add(CLUSTER_LABEL_NOISE)

            cluster_updates[obj] = max(labels)

        for obj, new_cluster_label in cluster_updates.items():
            self.clusters.set_label(obj, new_cluster_label)

    def _get_cluster_labels_in_neighborhood(self, obj):
        return {self.clusters.get_label(neighbor)
                for neighbor in obj.neighbors
                if neighbor.is_core}
