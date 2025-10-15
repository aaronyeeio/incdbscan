from typing import (
    Dict,
    List,
    Set,
    TYPE_CHECKING,
    Optional
)

import rustworkx as rx

from ._neighbor_searcher import NeighborSearcher
from ._object import (
    NodeId,
    Object,
    ObjectId
)

if TYPE_CHECKING:
    from ._clusters import Clusters


class Objects:
    def __init__(self, eps, min_pts, metric, p, clusters: 'Clusters', eps_soft, nearest_neighbors='torch_cuda'):
        self.clusters = clusters

        self.graph = rx.PyGraph(
            multigraph=False)  # pylint: disable=no-member
        self._object_id_to_node_id: Dict[ObjectId, NodeId] = {}

        self.neighbor_searcher = NeighborSearcher(
            metric=metric, p=p, nearest_neighbors=nearest_neighbors)
        self.min_pts = min_pts
        self.eps = eps
        self.eps_soft = eps_soft

        # Auto-increment ID generator
        self._next_id = 0

    def get_object_by_id(self, object_id: ObjectId) -> Optional[Object]:
        """Get object by its ID."""
        if object_id in self._object_id_to_node_id:
            return self._get_object_from_object_id(object_id)
        return None

    def _generate_id(self) -> ObjectId:
        """Generate a new auto-increment ID."""
        object_id = str(self._next_id)
        self._next_id += 1
        return object_id

    def batch_insert_objects(self, values, weights, ids: Optional[List[ObjectId]] = None):
        """Insert multiple objects at once, handling neighbor relationships efficiently.

        Each point is inserted as a separate object, even if multiple points have the same position.

        Args:
            values: array-like of embeddings/positions
            weights: array-like of weights for each point
            ids: optional list of object IDs. If None, auto-generated IDs are used.

        Returns:
            list of newly created Object instances
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id() for _ in range(len(values))]
        else:
            # Ensure IDs are strings
            ids = [str(id_) for id_ in ids]

        # Check for duplicate IDs in the input
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in input")

        # Check if any IDs already exist
        existing_ids = [
            id_ for id_ in ids if id_ in self._object_id_to_node_id]
        if existing_ids:
            raise ValueError(f"IDs already exist: {existing_ids}")

        # Create all new objects
        new_objects = []
        for object_id, weight in zip(ids, weights):
            new_obj = Object(object_id, self.min_pts, weight)
            self._insert_graph_metadata(new_obj)
            self.clusters.set_label_of_inserted_object(new_obj)
            new_objects.append(new_obj)

        if not new_objects:
            return []

        # Batch insert all new values into searcher (rebuilds tree once)
        self.neighbor_searcher.batch_insert(values, ids)

        # Batch query neighbors for all new objects
        all_neighbor_ids_list = self.neighbor_searcher.batch_query_neighbors(
            values, radius=self.eps)

        # Build lookup for new object indices
        new_obj_id_to_index = {obj.id: i for i, obj in enumerate(new_objects)}

        # Build a fast lookup cache for all objects we'll need to access
        all_unique_neighbor_ids = set()
        for neighbor_ids in all_neighbor_ids_list:
            all_unique_neighbor_ids.update(neighbor_ids)

        object_cache = {}
        for obj_id in all_unique_neighbor_ids:
            object_cache[obj_id] = self._get_object_from_object_id(obj_id)

        # Update neighbor relationships
        # Collect all edges to add them in batch (much faster than individual adds)
        edges_to_add = []

        for i, (new_obj, neighbor_ids) in enumerate(zip(new_objects, all_neighbor_ids_list)):
            for neighbor_id in neighbor_ids:
                neighbor_obj = object_cache[neighbor_id]

                if neighbor_id == new_obj.id:
                    # Self-neighbor: increase own count
                    new_obj.neighbor_count += new_obj.weight
                elif neighbor_id in new_obj_id_to_index and new_obj_id_to_index[neighbor_id] > i:
                    # New object not yet processed: update both sides and add edge
                    new_obj.neighbor_count += neighbor_obj.weight
                    new_obj.neighbors.add(neighbor_obj)
                    neighbor_obj.neighbor_count += new_obj.weight
                    neighbor_obj.neighbors.add(new_obj)
                    edges_to_add.append(
                        (new_obj.node_id, neighbor_obj.node_id, None))
                elif neighbor_id in new_obj_id_to_index:
                    # New object already processed: only add to set and edge
                    new_obj.neighbors.add(neighbor_obj)
                    edges_to_add.append(
                        (new_obj.node_id, neighbor_obj.node_id, None))
                else:
                    # Existing object: update both sides and add edge
                    new_obj.neighbor_count += neighbor_obj.weight
                    new_obj.neighbors.add(neighbor_obj)
                    neighbor_obj.neighbor_count += new_obj.weight
                    neighbor_obj.neighbors.add(new_obj)
                    edges_to_add.append(
                        (new_obj.node_id, neighbor_obj.node_id, None))

        # Batch add all edges at once (much faster than individual add_edge calls)
        if edges_to_add:
            self.graph.add_edges_from(edges_to_add)

        return new_objects

    def _insert_graph_metadata(self, new_object):
        node_id = self.graph.add_node(new_object)
        new_object.node_id = node_id
        object_id = new_object.id
        self._object_id_to_node_id[object_id] = node_id

    def _update_neighbors_during_insertion(self, object_inserted, new_value):
        neighbors = self._get_neighbors(new_value)
        for obj in neighbors:
            obj.neighbor_count += object_inserted.weight
            if obj.id != object_inserted.id:
                object_inserted.neighbor_count += obj.weight
                obj.neighbors.add(object_inserted)
                object_inserted.neighbors.add(obj)

    def _get_neighbors(self, query_value):
        neighbor_ids = self.neighbor_searcher.query_neighbors(query_value)

        for id_ in neighbor_ids:
            obj = self._get_object_from_object_id(id_)
            yield obj

    def _get_object_from_object_id(self, object_id):
        node_id = self._object_id_to_node_id[object_id]
        obj = self.graph[node_id]
        return obj

    def batch_delete_objects(self, ids_to_delete: List[ObjectId]):
        """Delete multiple objects by their IDs.

        Args:
            ids_to_delete: list of object IDs to delete

        Returns:
            was_core_map: dict mapping object to whether it was core before deletion
            objects_removed: list of objects that were removed
        """
        # Get objects from IDs
        objects_to_remove = []
        for object_id in ids_to_delete:
            obj = self.get_object_by_id(object_id)
            if obj is None:
                raise ValueError(f"Object with ID {object_id} not found")
            objects_to_remove.append(obj)

        # Record which objects were core before deletion
        was_core_map = {}
        for obj in objects_to_remove:
            was_core_map[obj] = obj.is_core

        # Phase 1: Update neighbor counts for all neighbors
        for obj in objects_to_remove:
            # Update neighbor counts for all neighbors
            for neighbor in obj.neighbors:
                neighbor.neighbor_count -= obj.weight

        # Phase 2: Clean up objects that need to be completely removed
        # Remove from neighbor sets
        for obj in objects_to_remove:
            for neighbor in obj.neighbors:
                if neighbor.id != obj.id:
                    neighbor.neighbors.remove(obj)

        # Batch delete from neighbor searcher (rebuild tree only once)
        if objects_to_remove:
            self.neighbor_searcher.batch_delete(ids_to_delete)

        # Remove graph metadata and labels
        for obj in objects_to_remove:
            self._delete_graph_metadata(obj)
            self.clusters.delete_label_of_deleted_object(obj)

        return was_core_map, objects_to_remove

    def _delete_graph_metadata(self, deleted_object):
        node_id = deleted_object.node_id
        self.graph.remove_node(node_id)
        del self._object_id_to_node_id[deleted_object.id]

    def get_connected_components_within_objects(
            self, objects: Set[Object]) -> List[Set[Object]]:

        if len(objects) == 1:
            return [objects]

        node_ids = [obj.node_id for obj in objects]
        subgraph = self.graph.subgraph(node_ids)
        components_as_ids: List[Set[NodeId]] = rx.connected_components(
            subgraph)  # pylint: disable=no-member

        def _get_original_object(subgraph, subgraph_node_id):
            original_node_id = subgraph[subgraph_node_id].node_id
            return self.graph[original_node_id]

        components_as_objects = []
        for component in components_as_ids:
            component_objects = {
                _get_original_object(subgraph, subgraph_node_id)
                for subgraph_node_id in component
            }
            components_as_objects.append(component_objects)

        return components_as_objects
