from ._clusters import (
    CLUSTER_LABEL_NOISE,
    CLUSTER_LABEL_UNCLASSIFIED
)


class Inserter:
    def __init__(self, eps, min_pts, min_cluster_size, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.objects = objects
        self.clusters = objects.clusters  # Shorthand for clusters

    def batch_insert(self, object_values, weights):
        """Insert multiple objects at once and update clustering."""
        new_objects, weight_updated_objects = self.objects.batch_insert_objects(
            object_values, weights)

        # Update cluster weight statistics for objects whose weight was updated
        if weight_updated_objects:
            self.clusters.update_weight_for_objects(weight_updated_objects)

        # Combine all objects that need clustering update
        objects_to_process = new_objects + weight_updated_objects

        if not objects_to_process:
            # Nothing to process
            return

        # Collect all affected objects (processed objects + their neighbors)
        all_affected_objects = set(objects_to_process)
        for obj in objects_to_process:
            all_affected_objects.update(obj.neighbors)

        # Track existing neighbors for core status update
        # These are neighbors that were not newly inserted or weight-updated in this batch
        existing_neighbors = all_affected_objects - set(objects_to_process)

        # Identify new cores and old cores among ALL affected objects
        new_core_neighbors, old_core_neighbors = \
            self._separate_core_neighbors_by_novelty_batch(
                objects_to_process, all_affected_objects)

        if not new_core_neighbors:
            # No new core objects created, just assign labels to processed objects
            for obj in objects_to_process:
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
        # that were processed but not assigned a label yet
        for obj in objects_to_process:
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

        # Dissolve clusters that are smaller than min_cluster_size
        self.clusters.dissolve_small_clusters(self.min_cluster_size)

    def _separate_core_neighbors_by_novelty_batch(self, objects_to_process, all_affected_objects):
        """Identify new cores and old cores among all affected objects during batch insert.

        Args:
            objects_to_process: Objects that were newly inserted or had weight updates
            all_affected_objects: All objects affected by the insertion (including neighbors)
        """
        new_cores = set()
        old_cores = set()
        objects_to_process_set = set(objects_to_process)

        for obj in all_affected_objects:
            if obj.neighbor_count == self.min_pts:
                # Just became a core (neighbor_count exactly equals min_pts)
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                # Has neighbor_count > min_pts
                if obj in objects_to_process_set:
                    # Object was newly inserted or had weight update -> treat as new core
                    new_cores.add(obj)
                else:
                    # Was core before this batch operation
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
        # Collect all candidate labels for each neighbor
        # For batch insertions, we want to assign the minimum label (sklearn behavior)
        # For incremental insertions, neighbors should switch to their core neighbors' clusters
        neighbor_candidate_labels = {}

        for obj in new_core_neighbors:
            label = self.clusters.get_label(obj)
            for neighbor in obj.neighbors:
                if neighbor not in neighbor_candidate_labels:
                    neighbor_candidate_labels[neighbor] = []
                neighbor_candidate_labels[neighbor].append(label)

        # Assign labels: use minimum for batch consistency with sklearn
        for neighbor, candidate_labels in neighbor_candidate_labels.items():
            # Get the minimum label from all new core neighbors of this neighbor
            min_label = min(candidate_labels)
            self.clusters.set_label(neighbor, min_label)
