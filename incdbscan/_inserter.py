from ._clusters import (
    CLUSTER_LABEL_NOISE,
    CLUSTER_LABEL_UNCLASSIFIED
)


class Inserter:
    def __init__(self, eps, min_pts, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.objects = objects
        self.clusters = objects.clusters  # Shorthand for clusters

    def batch_insert(self, object_values, weights):
        """Insert multiple objects at once and update clustering."""
        objects_inserted = self.objects.batch_insert_objects(
            object_values, weights)

        if not objects_inserted:
            # All objects already existed, no new objects to cluster
            return

        # Collect all affected objects (new objects + their neighbors)
        all_affected_objects = set(objects_inserted)
        for obj in objects_inserted:
            all_affected_objects.update(obj.neighbors)

        # Track existing neighbors for core status update
        existing_neighbors = all_affected_objects - set(objects_inserted)

        # Identify new cores and old cores among ALL affected objects
        new_core_neighbors, old_core_neighbors = \
            self._separate_core_neighbors_by_novelty_batch(
                objects_inserted, all_affected_objects)

        if not new_core_neighbors:
            # No new core objects created, just assign labels to new objects
            for obj in objects_inserted:
                # Find core neighbors of this object
                core_neighbors = [n for n in obj.neighbors
                                  if n.neighbor_count >= self.min_pts and n != obj]

                if core_neighbors:
                    # Assign to the most recent cluster
                    label_of_new_object = max([
                        self.clusters.get_label(n) for n in core_neighbors
                    ])
                else:
                    # No core neighbors, mark as noise
                    label_of_new_object = CLUSTER_LABEL_NOISE

                self.clusters.set_label(obj, label_of_new_object)
            return

        update_seeds = self._get_update_seeds(new_core_neighbors)

        connected_components_in_update_seeds = \
            self.objects.get_connected_components_within_objects(update_seeds)

        for component in connected_components_in_update_seeds:
            effective_cluster_labels = \
                self._get_effective_cluster_labels_of_objects(component)

            if not effective_cluster_labels:
                # If in a connected component of update seeds there are only
                # previously unclassified and noise objects, a new cluster is
                # created. Corresponds to case "Creation" in the paper.

                next_cluster_label = self.clusters.get_next_cluster_label()

                # Assign labels to all objects in component
                self.clusters.set_labels(component, next_cluster_label)

            else:
                # If in a connected component of update seeds there are
                # already clustered objects, all objects in the component
                # will be merged into the most recent cluster.
                # Corresponds to cases "Absorption" and "Merge" in the paper.

                max_label = max(effective_cluster_labels)

                # Use high-level set_labels for component (handles hooks properly)
                self.clusters.set_labels(component, max_label)

                # Then merge all effective clusters using change_labels
                for label in effective_cluster_labels:
                    self.clusters.change_labels(label, max_label)

        # Finally all neighbors of each new core object inherits a label from
        # its new core neighbor, thereby affecting border and noise objects,
        # and the object being inserted.

        self._set_cluster_label_around_new_core_neighbors(new_core_neighbors)

        # After processing core objects, handle any remaining UNCLASSIFIED objects
        # that were inserted but not assigned a label yet
        for obj in objects_inserted:
            if self.clusters.get_label(obj) == CLUSTER_LABEL_UNCLASSIFIED:
                # Find core neighbors
                core_neighbors = [n for n in obj.neighbors
                                  if n.neighbor_count >= self.min_pts and n != obj]

                if core_neighbors:
                    # Assign to the most recent cluster among core neighbors
                    label_of_new_object = max([
                        self.clusters.get_label(n) for n in core_neighbors
                    ])
                else:
                    # No core neighbors, mark as noise
                    label_of_new_object = CLUSTER_LABEL_NOISE

                self.clusters.set_label(obj, label_of_new_object)

        # Update core status for existing neighbors that might have changed
        self.clusters.update_core_status_for_objects(existing_neighbors)

    def _separate_core_neighbors_by_novelty_batch(self, objects_inserted, all_affected_objects):
        """Identify new cores and old cores among all affected objects during batch insert."""
        new_cores = set()
        old_cores = set()
        objects_inserted_set = set(objects_inserted)

        for obj in all_affected_objects:
            if obj.neighbor_count == self.min_pts:
                # Just became a core
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                # Was already a core, but check if it's a newly inserted object
                if obj in objects_inserted_set:
                    # Newly inserted object that is core -> treat as new core
                    new_cores.add(obj)
                else:
                    # Was core before this batch insertion
                    old_cores.add(obj)

        return new_cores, old_cores

    def _separate_core_neighbors_by_novelty(self, object_inserted):
        new_cores = set()
        old_cores = set()

        for obj in object_inserted.neighbors:
            if obj.neighbor_count == self.min_pts:
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                old_cores.add(obj)

        # If the inserted object is core, it is a new core

        if object_inserted in old_cores:
            old_cores.remove(object_inserted)
            new_cores.add(object_inserted)

        return new_cores, old_cores

    def _get_update_seeds(self, new_core_neighbors):
        seeds = set()

        for new_core_neighbor in new_core_neighbors:
            core_neighbors = [obj for obj in new_core_neighbor.neighbors
                              if obj.neighbor_count >= self.min_pts]
            seeds.update(core_neighbors)

        return seeds

    def _get_effective_cluster_labels_of_objects(self, objects):
        non_effective_cluster_labels = {CLUSTER_LABEL_UNCLASSIFIED,
                                        CLUSTER_LABEL_NOISE}
        effective_cluster_labels = set()

        for obj in objects:
            label = self.clusters.get_label(obj)
            if label not in non_effective_cluster_labels:
                effective_cluster_labels.add(label)

        return effective_cluster_labels

    def _set_cluster_label_around_new_core_neighbors(self, new_core_neighbors):
        for obj in new_core_neighbors:
            label = self.clusters.get_label(obj)
            self.clusters.set_labels(obj.neighbors, label)
