from typing import (
    Dict,
    List,
    Set,
    TYPE_CHECKING
)

import rustworkx as rx

from ._neighbor_searcher import NeighborSearcher
from ._object import (
    NodeId,
    Object,
    ObjectId
)
from ._utils import hash_

if TYPE_CHECKING:
    from ._clusters import Clusters


class Objects:
    def __init__(self, eps, eps_merge, min_pts, metric, p, clusters: 'Clusters'):
        self.clusters = clusters

        self.merge_graph = rx.PyGraph(
            multigraph=False)  # pylint: disable=no-member
        self._object_id_to_node_id: Dict[ObjectId, NodeId] = {}

        self.neighbor_searcher = \
            NeighborSearcher(radius=eps, metric=metric, p=p)
        self.merge_searcher = \
            NeighborSearcher(radius=eps_merge, metric=metric, p=p)
        self.min_pts = min_pts

    def get_object(self, value):
        object_id = hash_(value)
        if object_id in self._object_id_to_node_id:
            obj = self._get_object_from_object_id(object_id)
            return obj
        return None

    def batch_insert_objects(self, values, weights):
        """Insert multiple objects at once, handling neighbor relationships efficiently.

        Returns:
            tuple: (new_objects, weight_updated_objects)
                - new_objects: list of newly created Object instances
                - weight_updated_objects: list of existing Object instances whose weights were updated
        """
        object_ids = [hash_(value) for value in values]

        # Group duplicates: accumulate weights for same object_id
        id_info = {}  # object_id -> (value, total_weight)
        for i, (value, object_id, weight) in enumerate(zip(values, object_ids, weights)):
            if object_id in id_info:
                id_info[object_id] = (
                    id_info[object_id][0], id_info[object_id][1] + weight)
            else:
                id_info[object_id] = (value, weight)

        # Separate existing vs new objects
        new_objects = []
        new_values = []
        new_ids = []
        weight_updated_objects = []

        for object_id, (value, total_weight) in id_info.items():
            if object_id in self._object_id_to_node_id:
                # Existing object - update weight and neighbor counts
                obj = self._get_object_from_object_id(object_id)
                obj.weight += total_weight
                for neighbor in obj.neighbors:
                    neighbor.neighbor_count += total_weight
                weight_updated_objects.append(obj)
            else:
                # New object
                new_obj = Object(object_id, self.min_pts, total_weight)
                self._insert_graph_metadata(new_obj)
                self.clusters.set_label_of_inserted_object(new_obj)
                new_objects.append(new_obj)
                new_values.append(value)
                new_ids.append(object_id)

        if not new_objects:
            return [], weight_updated_objects

        # Batch insert all new values into both searchers (rebuilds trees once each)
        self.neighbor_searcher.batch_insert(new_values, new_ids)
        self.merge_searcher.batch_insert(new_values, new_ids)

        # Batch query neighbors for all new objects (both eps and eps_merge)
        all_neighbor_ids_list = self.neighbor_searcher.batch_query_neighbors(
            new_values)
        all_merge_neighbor_ids_list = self.merge_searcher.batch_query_neighbors(
            new_values)

        # Build lookup for new object indices
        new_obj_id_to_index = {obj.id: i for i, obj in enumerate(new_objects)}

        # Update neighbor relationships (eps neighbors)
        for i, (new_obj, neighbor_ids) in enumerate(zip(new_objects, all_neighbor_ids_list)):
            for neighbor_id in neighbor_ids:
                neighbor_obj = self._get_object_from_object_id(neighbor_id)

                if neighbor_id == new_obj.id:
                    # Self-neighbor: increase own count
                    new_obj.neighbor_count += new_obj.weight
                elif neighbor_id in new_obj_id_to_index and new_obj_id_to_index[neighbor_id] > i:
                    # New object not yet processed: update both sides
                    new_obj.neighbor_count += neighbor_obj.weight
                    new_obj.neighbors.add(neighbor_obj)
                    neighbor_obj.neighbor_count += new_obj.weight
                    neighbor_obj.neighbors.add(new_obj)
                elif neighbor_id in new_obj_id_to_index:
                    # New object already processed: only add to set
                    new_obj.neighbors.add(neighbor_obj)
                else:
                    # Existing object: update both sides
                    new_obj.neighbor_count += neighbor_obj.weight
                    new_obj.neighbors.add(neighbor_obj)
                    neighbor_obj.neighbor_count += new_obj.weight
                    neighbor_obj.neighbors.add(new_obj)

        # Update merge neighbor relationships (eps_merge neighbors)
        for i, (new_obj, merge_neighbor_ids) in enumerate(zip(new_objects, all_merge_neighbor_ids_list)):
            for neighbor_id in merge_neighbor_ids:
                neighbor_obj = self._get_object_from_object_id(neighbor_id)

                if neighbor_id == new_obj.id:
                    # Self is always a merge neighbor
                    pass
                elif neighbor_id in new_obj_id_to_index and new_obj_id_to_index[neighbor_id] > i:
                    # New object not yet processed: update both sides and add edge
                    new_obj.merge_neighbors.add(neighbor_obj)
                    neighbor_obj.merge_neighbors.add(new_obj)
                    self.merge_graph.add_edge(
                        new_obj.node_id, neighbor_obj.node_id, None)
                elif neighbor_id in new_obj_id_to_index:
                    # New object already processed: only add to set and edge
                    new_obj.merge_neighbors.add(neighbor_obj)
                    self.merge_graph.add_edge(
                        new_obj.node_id, neighbor_obj.node_id, None)
                else:
                    # Existing object: update both sides and add edge
                    new_obj.merge_neighbors.add(neighbor_obj)
                    neighbor_obj.merge_neighbors.add(new_obj)
                    self.merge_graph.add_edge(
                        new_obj.node_id, neighbor_obj.node_id, None)

        return new_objects, weight_updated_objects

    def _insert_graph_metadata(self, new_object):
        node_id = self.merge_graph.add_node(new_object)
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
        obj = self.merge_graph[node_id]
        return obj

    def batch_delete_objects(self, objects_to_delete, weights):
        """Delete multiple objects at once, handling weight reduction and cleanup efficiently.

        Args:
            objects_to_delete: list of Object instances to delete
            weights: list of weights to reduce for each object

        Returns:
            was_core_map: dict mapping object to whether it was core before deletion
            objects_to_remove: list of objects that will be completely removed (weight <= 0)
        """
        from collections import defaultdict

        # Group by object and accumulate weights for duplicates
        object_weight_map = defaultdict(float)
        for obj, weight in zip(objects_to_delete, weights):
            object_weight_map[obj] += weight

        # Record which objects were core before deletion
        was_core_map = {}
        for obj in object_weight_map:
            was_core_map[obj] = obj.is_core

        # Phase 1: Reduce weights for all objects and their neighbors
        objects_to_remove = []
        for obj, total_weight in object_weight_map.items():
            obj.weight -= total_weight

            # Update neighbor counts for all neighbors
            for neighbor in obj.neighbors:
                neighbor.neighbor_count -= total_weight

            if obj.weight <= 0:
                objects_to_remove.append(obj)

        # Phase 2: Clean up objects that need to be completely removed
        # Remove from both neighbor sets
        for obj in objects_to_remove:
            for neighbor in obj.neighbors:
                if neighbor.id != obj.id:
                    neighbor.neighbors.remove(obj)
            for neighbor in obj.merge_neighbors:
                if neighbor.id != obj.id:
                    neighbor.merge_neighbors.remove(obj)

        # Batch delete from both neighbor searchers (rebuild trees only once each)
        if objects_to_remove:
            ids_to_remove = [obj.id for obj in objects_to_remove]
            self.neighbor_searcher.batch_delete(ids_to_remove)
            self.merge_searcher.batch_delete(ids_to_remove)

        # Remove graph metadata and labels
        for obj in objects_to_remove:
            self._delete_graph_metadata(obj)
            self.clusters.delete_label_of_deleted_object(obj)

        return was_core_map, objects_to_remove

    def _delete_graph_metadata(self, deleted_object):
        node_id = deleted_object.node_id
        self.merge_graph.remove_node(node_id)
        del self._object_id_to_node_id[deleted_object.id]

    def get_connected_components_within_objects(
            self, objects: Set[Object]) -> List[Set[Object]]:

        if len(objects) == 1:
            return [objects]

        node_ids = [obj.node_id for obj in objects]
        subgraph = self.merge_graph.subgraph(node_ids)
        components_as_ids: List[Set[NodeId]] = rx.connected_components(
            subgraph)  # pylint: disable=no-member

        def _get_original_object(subgraph, subgraph_node_id):
            original_node_id = subgraph[subgraph_node_id].node_id
            return self.merge_graph[original_node_id]

        components_as_objects = []
        for component in components_as_ids:
            component_objects = {
                _get_original_object(subgraph, subgraph_node_id)
                for subgraph_node_id in component
            }
            components_as_objects.append(component_objects)

        return components_as_objects
