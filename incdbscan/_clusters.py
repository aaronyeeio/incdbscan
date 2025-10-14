from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from ._cluster import Cluster, ClusterLabel

if TYPE_CHECKING:
    from ._object import Object

# Label constants
CLUSTER_LABEL_UNCLASSIFIED: ClusterLabel = -2
CLUSTER_LABEL_NOISE: ClusterLabel = -1
CLUSTER_LABEL_FIRST_CLUSTER: ClusterLabel = 0


class Clusters:
    """Manages all cluster instances, lifecycle events, and label assignments.

    This class replaces the old LabelHandler and provides both cluster tracking
    and label management functionality.
    """

    def __init__(self):
        # Cluster tracking
        self._clusters: Dict[ClusterLabel, Cluster] = {}
        self._max_cluster_label_ever_used: int = -1  # Track max label to avoid reuse

        # Label management
        self._object_to_label: Dict['Object', ClusterLabel] = {}
        self._label_to_objects = defaultdict(set)  # For noise and unclassified

    # ==================== Private Internal Methods ====================
    # These methods only modify data structures without triggering hooks

    def _assign_label_internal(self, obj: 'Object', label: ClusterLabel):
        """Internally assign a label to an object without any cluster operations.

        Args:
            obj: The object to label
            label: The cluster label
        """
        self._object_to_label[obj] = label

    def _add_to_cluster_internal(self, cluster: Cluster, obj: 'Object'):
        """Internally add object to cluster and update statistics.

        Args:
            cluster: The cluster to add to
            obj: The object to add
        """
        cluster.members.add(obj)
        cluster._update_statistics_incremental(obj, is_adding=True)

    def _register_cluster(self, label: ClusterLabel, cluster: Cluster):
        """Register a cluster and update the maximum label tracker.

        Args:
            label: The cluster label
            cluster: The cluster instance
        """
        self._clusters[label] = cluster
        if label >= CLUSTER_LABEL_FIRST_CLUSTER:
            self._max_cluster_label_ever_used = max(
                self._max_cluster_label_ever_used, label)

    # ==================== Mid-Level Cluster Operations ====================
    # These methods trigger hooks and assume labels are already correct

    def create_cluster(self, label: ClusterLabel, members: Set['Object'],
                       parent_labels: Optional[List[int]] = None) -> Cluster:
        """Create a new cluster with given members and trigger creation hook.

        Assumes all members already have their label set to 'label'.

        Args:
            label: The cluster label
            members: Objects that belong to this cluster
            parent_labels: Optional parent cluster labels (for merge/split)

        Returns:
            The newly created Cluster instance
        """
        cluster = Cluster(label, parent_labels)

        # Batch add all members
        for obj in members:
            self._add_to_cluster_internal(cluster, obj)

        # Trigger creation hook
        cluster.on_cluster_created(members)

        self._register_cluster(label, cluster)
        return cluster

    def get_cluster(self, label: ClusterLabel) -> Optional[Cluster]:
        """Get cluster by label.

        Args:
            label: The cluster label

        Returns:
            Cluster instance or None if not found
        """
        return self._clusters.get(label)

    def merge_clusters(self, target_label: ClusterLabel, source_labels: List[ClusterLabel]):
        """Merge multiple clusters into one target cluster.

        Note: This method assumes cluster.members are already correct.
        Use change_labels() if you need to relocate objects between clusters.

        Args:
            target_label: Label of the target cluster
            source_labels: Labels of clusters being merged (including target if it exists)
        """
        # Get target and source clusters
        target_cluster = self._clusters.get(target_label)
        source_clusters = [self._clusters.get(label) for label in source_labels
                           if label != target_label and label in self._clusters]

        # Collect all members from source clusters
        all_members = set()
        absorbed_members = set()

        if target_cluster:
            all_members.update(target_cluster.members)

        for source_cluster in source_clusters:
            absorbed_members.update(source_cluster.members)
            all_members.update(source_cluster.members)

        if target_cluster is None:
            # Create new target cluster with all merged members
            actual_source_labels = [
                label for label in source_labels if label != target_label]
            target_cluster = Cluster(
                target_label, parent_labels=actual_source_labels)

            for obj in all_members:
                self._add_to_cluster_internal(target_cluster, obj)

            target_cluster.on_cluster_created(all_members)
            self._register_cluster(target_label, target_cluster)
        else:
            # Target exists, merge absorbed members into it
            if absorbed_members:
                for obj in absorbed_members:
                    self._add_to_cluster_internal(target_cluster, obj)

                actual_source_labels = [
                    label for label in source_labels if label != target_label]
                target_cluster.on_cluster_merged(
                    actual_source_labels, absorbed_members)

        # Destroy source clusters
        for source_cluster in source_clusters:
            source_cluster.merged_into = target_label
            source_cluster.on_cluster_destroyed(reason="merged")
            del self._clusters[source_cluster.label]

    def split_cluster(self, original_label: ClusterLabel,
                      new_components: List[tuple[ClusterLabel, Set['Object']]]):
        """Handle cluster split into multiple components.

        Assumes all objects in new_components have already been relabeled.

        Args:
            original_label: Label of the cluster being split
            new_components: List of (new_label, members) tuples for new clusters
        """
        original_cluster = self._clusters.get(original_label)

        new_clusters = []
        for new_label, members in new_components:
            new_cluster = Cluster(new_label, parent_labels=[original_label])
            new_cluster.split_from = original_label

            # Batch add members
            for obj in members:
                self._add_to_cluster_internal(new_cluster, obj)

            # Trigger creation hook
            new_cluster.on_cluster_created(members)

            self._register_cluster(new_label, new_cluster)
            new_clusters.append(new_cluster)

        # Call split hook on original cluster if it exists
        if original_cluster:
            original_cluster.on_cluster_split(new_clusters)

    def add_object_to_cluster(self, label: ClusterLabel, obj: 'Object', event_type: str):
        """Add an object to a cluster.

        Args:
            label: The cluster label
            obj: The object to add
            event_type: Type of addition event
        """
        # Skip noise and unclassified
        if label in (CLUSTER_LABEL_NOISE, CLUSTER_LABEL_UNCLASSIFIED):
            return

        cluster = self._clusters.get(label)
        if cluster is None:
            # Auto-create cluster if it doesn't exist
            cluster = Cluster(label)
            self._register_cluster(label, cluster)

        cluster.on_object_added(obj, event_type)

    def remove_object_from_cluster(self, label: ClusterLabel, obj: 'Object', event_type: str):
        """Remove an object from a cluster.

        Args:
            label: The cluster label
            obj: The object to remove
            event_type: Type of removal event
        """
        # Skip noise and unclassified
        if label in (CLUSTER_LABEL_NOISE, CLUSTER_LABEL_UNCLASSIFIED):
            return

        cluster = self._clusters.get(label)
        if cluster:
            cluster.on_object_removed(obj, event_type)

            # Destroy cluster if empty
            if cluster.size == 0:
                cluster.on_cluster_destroyed(reason="empty")
                del self._clusters[label]

    def destroy_cluster(self, label: ClusterLabel, reason: str):
        """Destroy a cluster.

        Args:
            label: The cluster label
            reason: Reason for destruction
        """
        cluster = self._clusters.get(label)
        if cluster:
            cluster.on_cluster_destroyed(reason)
            del self._clusters[label]

    def get_all_clusters(self) -> List[Cluster]:
        """Get all active clusters.

        Returns:
            List of all Cluster instances
        """
        return list(self._clusters.values())

    def get_cluster_count(self) -> int:
        """Get number of active clusters.

        Returns:
            Number of clusters
        """
        return len(self._clusters)

    def get_statistics(self) -> dict:
        """Get overall statistics for all clusters.

        Returns:
            Dictionary with aggregated statistics
        """
        total_size = sum(c.size for c in self._clusters.values())
        total_cores = sum(c.core_count for c in self._clusters.values())
        total_borders = sum(c.border_count for c in self._clusters.values())
        total_weight = sum(c.total_weight for c in self._clusters.values())

        return {
            'cluster_count': len(self._clusters),
            'total_objects': total_size,
            'total_cores': total_cores,
            'total_borders': total_borders,
            'total_weight': total_weight,
            'clusters': {label: c.get_summary() for label, c in self._clusters.items()}
        }

    def update_core_status_for_objects(self, objects):
        """Update core status statistics for objects whose is_core may have changed.

        Args:
            objects: Iterable of objects to check and update
        """
        for obj in objects:
            label = self._object_to_label.get(obj)
            if label is not None and label >= CLUSTER_LABEL_FIRST_CLUSTER:
                cluster = self._clusters.get(label)
                if cluster:
                    cluster.update_member_core_status(obj)

    def update_weight_for_objects(self, objects):
        """Update weight statistics for objects whose weight has changed.

        Args:
            objects: Iterable of objects whose weight was updated
        """
        for obj in objects:
            label = self._object_to_label.get(obj)
            if label is not None and label >= CLUSTER_LABEL_FIRST_CLUSTER:
                cluster = self._clusters.get(label)
                if cluster:
                    cluster.update_member_weight(obj)

    # ==================== High-Level Convenience Methods ====================
    # These methods handle both label changes and cluster operations

    def set_label(self, obj: 'Object', label: ClusterLabel):
        """Set label for an object and update cluster membership.

        This is a high-level convenience method that handles both label updates
        and cluster membership changes with proper hooks.

        Args:
            obj: The object to label
            label: The cluster label
        """
        # Get previous label if exists
        previous_label = self._object_to_label.get(obj)

        # Skip if label is already correct
        if previous_label == label:
            return

        # Remove from previous location
        if previous_label is not None:
            if previous_label >= CLUSTER_LABEL_FIRST_CLUSTER:
                # Remove from cluster with hook
                cluster = self._clusters.get(previous_label)
                if cluster:
                    # Just call the hook - it will handle members and statistics
                    cluster.on_object_removed(obj, event_type="reassignment")
                    # Destroy if empty
                    if cluster.size == 0:
                        cluster.on_cluster_destroyed(reason="empty")
                        del self._clusters[previous_label]
            else:
                # Remove from noise/unclassified set
                self._label_to_objects[previous_label].discard(obj)

        # Update label mapping
        self._object_to_label[obj] = label

        # Add to new location
        if label >= CLUSTER_LABEL_FIRST_CLUSTER:
            # Add to cluster with hook
            cluster = self._clusters.get(label)
            if cluster is None:
                # Auto-create cluster if it doesn't exist
                cluster = Cluster(label)
                self._register_cluster(label, cluster)

            # Just call the hook - it will handle members and statistics
            cluster.on_object_added(obj, event_type="reassignment")
        else:
            # Add to noise/unclassified set
            self._label_to_objects[label].add(obj)

    def set_label_of_inserted_object(self, obj: 'Object'):
        """Set initial label for a newly inserted object.

        Args:
            obj: The newly inserted object
        """
        self._object_to_label[obj] = CLUSTER_LABEL_UNCLASSIFIED
        self._label_to_objects[CLUSTER_LABEL_UNCLASSIFIED].add(obj)

    def set_labels(self, objects, label: ClusterLabel):
        """Set label for multiple objects.

        Args:
            objects: Collection of objects to label
            label: The cluster label
        """
        for obj in objects:
            self.set_label(obj, label)

    def delete_label_of_deleted_object(self, obj: 'Object'):
        """Delete label of an object being removed.

        Args:
            obj: The object being deleted
        """
        label = self.get_label(obj)

        # Remove from cluster or noise/unclassified set
        if label >= CLUSTER_LABEL_FIRST_CLUSTER:
            self.remove_object_from_cluster(label, obj, event_type="deletion")
        else:
            self._label_to_objects[label].discard(obj)

        # Remove from object_to_label mapping
        del self._object_to_label[obj]

    def get_label(self, obj: 'Object') -> ClusterLabel:
        """Get the label of an object.

        Args:
            obj: The object to query

        Returns:
            The cluster label
        """
        return self._object_to_label[obj]

    def get_next_cluster_label(self) -> ClusterLabel:
        """Get the next available cluster label.

        Returns:
            Next cluster label (>= 0)
        """
        # Return next label after the maximum ever used
        return max(self._max_cluster_label_ever_used + 1, CLUSTER_LABEL_FIRST_CLUSTER)

    def dissolve_small_clusters(self, min_cluster_size: float):
        """Dissolve clusters with total weight less than min_cluster_size.

        Args:
            min_cluster_size: Minimum total weight threshold
        """
        if min_cluster_size <= 0:
            return

        clusters_to_dissolve = [
            cluster for cluster in self.get_all_clusters()
            if cluster.total_weight < min_cluster_size
        ]

        for cluster in clusters_to_dissolve:
            for obj in list(cluster.members):
                self.set_label(obj, CLUSTER_LABEL_NOISE)

    def change_labels(self, change_from: ClusterLabel, change_to: ClusterLabel):
        """Change all objects from one label to another.

        This is a high-level convenience method that handles both label changes
        and cluster operations.

        Args:
            change_from: Source label
            change_to: Target label
        """
        # Skip if labels are the same
        if change_from == change_to:
            return

        # Handle cluster-to-cluster merge
        if change_from >= CLUSTER_LABEL_FIRST_CLUSTER and change_to >= CLUSTER_LABEL_FIRST_CLUSTER:
            source_cluster = self._clusters.get(change_from)
            if not source_cluster:
                return

            # Step 1: Update object labels
            for obj in source_cluster.members:
                self._object_to_label[obj] = change_to

            # Step 2: Merge clusters (assumes labels already updated)
            self.merge_clusters(change_to, [change_from, change_to])

        # Handle other label transitions
        else:
            # Get affected objects
            if change_from >= CLUSTER_LABEL_FIRST_CLUSTER:
                source_cluster = self._clusters.get(change_from)
                if not source_cluster:
                    return
                affected_objects = source_cluster.members.copy()
                # Destroy source cluster
                source_cluster.on_cluster_destroyed(reason="reclassified")
                del self._clusters[change_from]
            else:
                # From noise/unclassified
                affected_objects = self._label_to_objects.pop(
                    change_from, set())

            # Update object labels
            for obj in affected_objects:
                self._object_to_label[obj] = change_to

            # Add to target
            if change_to >= CLUSTER_LABEL_FIRST_CLUSTER:
                # To cluster
                for obj in affected_objects:
                    self.add_object_to_cluster(
                        change_to, obj, event_type="reassignment")
            else:
                # To noise/unclassified
                self._label_to_objects[change_to].update(affected_objects)
