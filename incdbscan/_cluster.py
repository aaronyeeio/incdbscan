import time
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from ._object import Object

# Define ClusterLabel locally to avoid circular import
ClusterLabel = int


class Cluster:
    """Represents a cluster with tracking and lifecycle methods.

    This class maintains cluster state and provides methods that are called
    during cluster lifecycle events (creation, merge, split, etc.).
    """

    def __init__(self, label: ClusterLabel, parent_labels: Optional[List[int]] = None):
        # Basic attributes
        self.label: ClusterLabel = label
        self.members: Set[Object] = set()

        # Statistics
        self.size: int = 0
        self.core_count: int = 0
        self.border_count: int = 0

        # Track the core status when each object was added
        self._member_core_status: dict = {}  # obj -> was_core_when_added

        # Temporal information
        self.created_at: float = time.time()
        self.last_modified: float = self.created_at

        # Evolution history
        self.parent_labels: List[int] = parent_labels if parent_labels else []
        self.merged_into: Optional[int] = None
        self.split_from: Optional[int] = None

        # Event counters
        self.merge_count: int = 0
        self.split_count: int = 0
        self.total_objects_added: int = 0
        self.total_objects_removed: int = 0

    def _update_statistics_incremental(self, obj: 'Object', is_adding: bool):
        """Incrementally update statistics when adding/removing an object.

        Args:
            obj: The object being added or removed
            is_adding: True if adding, False if removing
        """
        delta = 1 if is_adding else -1
        self.size += delta

        if is_adding:
            # When adding, record current core status and update counts
            is_core = obj.is_core
            self._member_core_status[obj] = is_core
            if is_core:
                self.core_count += 1
            else:
                self.border_count += 1
        else:
            # When removing, use the recorded core status from when it was added
            was_core_when_added = self._member_core_status.pop(
                obj, obj.is_core)
            if was_core_when_added:
                self.core_count -= 1
            else:
                self.border_count -= 1

        self.last_modified = time.time()

    def on_cluster_created(self, trigger_objects: Set['Object']):
        """Called when cluster is first created.

        This only logs the creation. Members will be added via set_label() calls.

        Args:
            trigger_objects: The objects that triggered cluster creation
        """
        # Log creation event
        print(f"[CLUSTER CREATED] Label={self.label}, TriggerCount={len(trigger_objects)}, "
              f"Parents={self.parent_labels}")

    def on_cluster_merged(self, source_labels: List[int], absorbed_members: Set['Object']):
        """Called when other clusters are merged into this cluster.

        Members are already in the cluster via previous add operations.
        This only updates merge metadata.

        Args:
            source_labels: Labels of clusters being merged into this one
            absorbed_members: All members from merged clusters
        """
        self.parent_labels.extend(source_labels)
        self.merge_count += len(source_labels)

        # Log merge event
        print(f"[CLUSTER MERGED] Target={self.label}, Sources={source_labels}, "
              f"AbsorbedCount={len(absorbed_members)}, TotalMerges={self.merge_count}")

    def on_cluster_split(self, new_clusters: List['Cluster']):
        """Called when this cluster splits into multiple clusters.

        Args:
            new_clusters: List of new clusters created from the split
        """
        self.split_count += 1

        # Mark all new clusters as split from this one
        for cluster in new_clusters:
            cluster.split_from = self.label

        # Log split event
        new_labels = [c.label for c in new_clusters]
        new_sizes = [c.size for c in new_clusters]
        print(f"[CLUSTER SPLIT] Original={self.label} (size={self.size}), "
              f"NewClusters={new_labels}, Sizes={new_sizes}")

    def on_object_added(self, obj: 'Object', event_type: str):
        """Called when an object is added to the cluster.

        Args:
            obj: The object being added
            event_type: Type of addition - "insertion", "absorption", "border_update"
        """
        self.members.add(obj)
        self.total_objects_added += 1
        self._update_statistics_incremental(obj, is_adding=True)

        # Optional: Log individual additions (may be verbose for large batches)
        # print(f"[OBJECT ADDED] Cluster={self.label}, ObjID={obj.id}, "
        #       f"Type={event_type}, IsCore={obj.is_core}")

    def on_object_removed(self, obj: 'Object', event_type: str):
        """Called when an object is removed from the cluster.

        Args:
            obj: The object being removed
            event_type: Type of removal - "deletion", "reassignment", "became_noise"
        """
        self.members.discard(obj)
        self.total_objects_removed += 1
        self._update_statistics_incremental(obj, is_adding=False)

        # Optional: Log individual removals (may be verbose for large batches)
        # print(f"[OBJECT REMOVED] Cluster={self.label}, ObjID={obj.id}, "
        #       f"Type={event_type}, RemainingSize={self.size}")

    def on_cluster_destroyed(self, reason: str):
        """Called when cluster is about to be destroyed.

        Args:
            reason: Reason for destruction - "merged", "empty", "all_noise"
        """
        lifetime = time.time() - self.created_at

        # Log destruction event
        print(f"[CLUSTER DESTROYED] Label={self.label}, Reason={reason}, "
              f"Lifetime={lifetime:.2f}s, TotalAdded={self.total_objects_added}, "
              f"TotalRemoved={self.total_objects_removed}, "
              f"Merges={self.merge_count}, Splits={self.split_count}")

        # Clear all data
        self.members.clear()
        self._member_core_status.clear()
        self.size = 0
        self.core_count = 0
        self.border_count = 0

    def update_member_core_status(self, obj: 'Object'):
        """Update the recorded core status of a member when it changes.

        This should be called when an object's is_core property changes
        while it's already in the cluster (e.g., after neighbor count changes).

        Args:
            obj: The object whose core status changed
        """
        if obj not in self._member_core_status:
            return

        old_is_core = self._member_core_status[obj]
        new_is_core = obj.is_core

        if old_is_core != new_is_core:
            if new_is_core:
                # Border -> Core
                self.border_count -= 1
                self.core_count += 1
            else:
                # Core -> Border
                self.core_count -= 1
                self.border_count += 1

            self._member_core_status[obj] = new_is_core
            self.last_modified = time.time()

    def get_summary(self) -> dict:
        """Get a summary of cluster state.

        Returns:
            Dictionary containing cluster statistics
        """
        return {
            'label': self.label,
            'size': self.size,
            'core_count': self.core_count,
            'border_count': self.border_count,
            'created_at': self.created_at,
            'lifetime': time.time() - self.created_at,
            'parent_labels': self.parent_labels,
            'merged_into': self.merged_into,
            'split_from': self.split_from,
            'merge_count': self.merge_count,
            'split_count': self.split_count,
            'total_objects_added': self.total_objects_added,
            'total_objects_removed': self.total_objects_removed,
        }

    def __repr__(self):
        return (f"Cluster(label={self.label}, size={self.size}, "
                f"cores={self.core_count}, borders={self.border_count})")
