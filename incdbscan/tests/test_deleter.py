import numpy as np
from conftest import EPS

from testutils import (
    CLUSTER_LABEL_FIRST_CLUSTER,
    CLUSTER_LABEL_NOISE,
    assert_cluster_labels,
    assert_label_of_object_is_among_possible_ones,
    assert_split_creates_new_labels_for_new_clusters,
    insert_objects_then_assert_cluster_labels,
    reflect_horizontally
)


def test_after_deleting_enough_objects_only_noise_remain(
        incdbscan4,
        blob_in_middle):

    inserted_objects = incdbscan4.insert(blob_in_middle)
    object_ids = [obj.id for obj in inserted_objects]

    for i in range(len(blob_in_middle) - 1):
        # Delete by ID
        incdbscan4.delete([object_ids[i]])

        expected_label = (
            CLUSTER_LABEL_NOISE
            if i > incdbscan4.min_pts + 1
            else CLUSTER_LABEL_FIRST_CLUSTER
        )

        # Check remaining objects
        remaining_ids = object_ids[i+1:]
        assert_cluster_labels(incdbscan4, remaining_ids, expected_label)


def test_deleting_cores_only_makes_borders_noise(incdbscan4, point_at_origin):
    # Insert core point and get ID
    core_objects = incdbscan4.insert(point_at_origin)
    core_id = core_objects[0].id

    border = np.array([
        [EPS, 0],
        [0, EPS],
        [0, -EPS],
    ])

    # Insert border points and get IDs
    border_objects = incdbscan4.insert(border)
    border_ids = [obj.id for obj in border_objects]

    # Delete core point
    incdbscan4.delete([core_id])

    assert_cluster_labels(incdbscan4, border_ids, CLUSTER_LABEL_NOISE)


def test_objects_losing_core_property_can_keep_cluster_id(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    core_points = np.array([
        [EPS, 0],
        [0, EPS],
        [EPS, EPS],
    ])

    all_points = np.vstack([point_to_delete, core_points])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id
    core_point_ids = [obj.id for obj in inserted_objects[1:]]

    assert_cluster_labels(incdbscan3, core_point_ids,
                          CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])
    assert_cluster_labels(incdbscan3, core_point_ids,
                          CLUSTER_LABEL_FIRST_CLUSTER)


def test_border_object_can_switch_to_other_cluster(
        incdbscan4,
        point_at_origin):

    border = point_at_origin
    border_objects = incdbscan4.insert(border)
    border_id = border_objects[0].id

    cluster_1 = np.array([
        [EPS, 0],
        [EPS, EPS],
        [EPS, -EPS],
    ])
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    cluster_2 = reflect_horizontally(cluster_1)
    cluster_2_expected_label = cluster_1_expected_label + 1

    insert_objects_then_assert_cluster_labels(
        incdbscan4, cluster_1, cluster_1_expected_label)

    cluster_2_objects = incdbscan4.insert(cluster_2)
    cluster_2_point_to_delete_id = cluster_2_objects[0].id

    assert_cluster_labels(incdbscan4, [border_id], cluster_2_expected_label)

    incdbscan4.delete([cluster_2_point_to_delete_id])

    assert_cluster_labels(incdbscan4, [border_id], cluster_1_expected_label)


def test_borders_around_point_losing_core_property_can_become_noise(
        incdbscan4,
        point_at_origin):

    point_to_delete = point_at_origin

    core = np.array([[0, EPS]])

    border = np.array([
        [0, EPS * 2],
        [EPS, EPS]
    ])

    all_points = np.vstack([point_to_delete, core, border])
    all_points_but_point_to_delete = np.vstack([core, border])

    inserted_objects = incdbscan4.insert(all_points)
    point_to_delete_id = inserted_objects[0].id
    remaining_point_ids = [obj.id for obj in inserted_objects[1:]]

    assert_cluster_labels(incdbscan4, remaining_point_ids,
                          CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan4.delete([point_to_delete_id])

    assert_cluster_labels(incdbscan4, remaining_point_ids, CLUSTER_LABEL_NOISE)


def test_core_property_of_singleton_update_seed_is_kept_after_deletion(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    cores = np.array([
        [EPS, 0],
        [2 * EPS, 0],
        [2 * EPS, 0],
    ])

    lonely = np.array([[-EPS, 0]])

    all_points = np.vstack([point_to_delete, cores, lonely])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id
    core_ids = [obj.id for obj in inserted_objects[1:4]]
    lonely_id = inserted_objects[4].id

    assert_cluster_labels(incdbscan3, core_ids, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(incdbscan3, [lonely_id], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    assert_cluster_labels(incdbscan3, core_ids, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(incdbscan3, [lonely_id], CLUSTER_LABEL_NOISE)


def test_cluster_id_of_single_component_update_seeds_is_kept_after_deletion(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    cores = np.array([
        [EPS, 0],
        [EPS, 0],
        [2 * EPS, 0],
    ])

    lonely = np.array([[-EPS, 0]])

    all_points = np.vstack([point_to_delete, cores, lonely])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id
    core_ids = [obj.id for obj in inserted_objects[1:4]]
    lonely_id = inserted_objects[4].id

    assert_cluster_labels(incdbscan3, core_ids, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(incdbscan3, [lonely_id], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    assert_cluster_labels(incdbscan3, core_ids, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(incdbscan3, [lonely_id], CLUSTER_LABEL_NOISE)


def test_cluster_id_of_single_component_objects_is_kept_after_deletion(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    cores = np.array([
        [EPS, 0],
        [0, EPS],
        [EPS, EPS],
        [EPS, EPS],
    ])

    all_points = np.vstack([point_to_delete, cores])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id
    core_ids = [obj.id for obj in inserted_objects[1:]]

    assert_cluster_labels(incdbscan3, core_ids, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    assert_cluster_labels(incdbscan3, core_ids, CLUSTER_LABEL_FIRST_CLUSTER)


def test_simple_two_way_split(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left
    points_right = reflect_horizontally(points_left)

    all_points = np.vstack([point_to_delete, points_left, points_right])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id

    assert_cluster_labels(
        incdbscan3, [obj.id for obj in inserted_objects[1:]], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    # Get object IDs for the remaining points after deletion
    left_ids = [obj.id for obj in inserted_objects[1:4]]  # points_left
    right_ids = [obj.id for obj in inserted_objects[4:]]  # points_right

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3, [left_ids, right_ids], CLUSTER_LABEL_FIRST_CLUSTER)


def test_simple_two_way_split_with_noise(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left,
        three_points_on_the_top,
        three_points_at_the_bottom):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left
    points_top = three_points_on_the_top
    points_bottom = three_points_at_the_bottom[:-1]

    all_points = np.vstack([
        point_to_delete,
        points_left,
        points_top,
        points_bottom
    ])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id

    assert_cluster_labels(
        incdbscan3, [obj.id for obj in inserted_objects[1:]], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    # Get object IDs for the remaining points after deletion
    left_ids = [obj.id for obj in inserted_objects[1:4]]  # points_left
    top_ids = [obj.id for obj in inserted_objects[4:7]]  # points_top
    bottom_ids = [obj.id for obj in inserted_objects[7:]]  # points_bottom

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3, [left_ids, top_ids], CLUSTER_LABEL_FIRST_CLUSTER)

    assert_cluster_labels(incdbscan3, bottom_ids, CLUSTER_LABEL_NOISE)


def test_three_way_split(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left,
        three_points_on_the_top,
        three_points_at_the_bottom):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left
    points_top = three_points_on_the_top
    points_bottom = three_points_at_the_bottom

    all_points = np.vstack([
        point_to_delete,
        points_left,
        points_top,
        points_bottom
    ])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id

    assert_cluster_labels(
        incdbscan3, [obj.id for obj in inserted_objects[1:]], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    # Get object IDs for the remaining points after deletion
    left_ids = [obj.id for obj in inserted_objects[1:4]]  # points_left
    top_ids = [obj.id for obj in inserted_objects[4:7]]  # points_top
    bottom_ids = [obj.id for obj in inserted_objects[7:]]  # points_bottom

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3,
        [left_ids, top_ids, bottom_ids],
        CLUSTER_LABEL_FIRST_CLUSTER
    )


def test_simultaneous_split_and_non_split(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left

    points_right = np.array([
        [0, EPS],
        [0, -EPS],
        [EPS, 0],
        [EPS, EPS],
        [EPS, -EPS],
    ])

    all_points = np.vstack([point_to_delete, points_left, points_right])

    inserted_objects = incdbscan3.insert(all_points)
    point_to_delete_id = inserted_objects[0].id

    assert_cluster_labels(
        incdbscan3, [obj.id for obj in inserted_objects[1:]], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete([point_to_delete_id])

    # Get object IDs for the remaining points after deletion
    left_ids = [obj.id for obj in inserted_objects[1:4]]  # points_left
    right_ids = [obj.id for obj in inserted_objects[4:]]  # points_right

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3, [left_ids, right_ids], CLUSTER_LABEL_FIRST_CLUSTER)


def test_two_way_split_with_non_dense_bridge(incdbscan4, point_at_origin):
    point_to_delete = bridge_point = point_at_origin

    points_left = np.array([
        [0, -EPS],
        [0, -EPS * 2],
        [0, -EPS * 2],
        [0, -EPS * 3],
        [0, -EPS * 3],
    ])

    points_right = np.array([
        [0, EPS],
        [0, EPS * 2],
        [0, EPS * 2],
        [0, EPS * 3],
        [0, EPS * 3],
    ])

    all_points = np.vstack([
        bridge_point, point_to_delete, points_left, points_right
    ])

    inserted_objects = incdbscan4.insert(all_points)
    # point_to_delete is the second point
    point_to_delete_id = inserted_objects[1].id
    bridge_point_id = inserted_objects[0].id

    assert_cluster_labels(
        incdbscan4, [obj.id for obj in inserted_objects], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan4.delete([point_to_delete_id])

    # Get object IDs for the remaining points after deletion
    left_ids = [obj.id for obj in inserted_objects[2:7]]  # points_left
    right_ids = [obj.id for obj in inserted_objects[7:]]  # points_right

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan4, [left_ids, right_ids], CLUSTER_LABEL_FIRST_CLUSTER)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point_id,
        {CLUSTER_LABEL_FIRST_CLUSTER, CLUSTER_LABEL_FIRST_CLUSTER + 1}
    )


def test_simultaneous_splits_within_two_clusters(
        incdbscan4,
        point_at_origin,
        hourglass_on_the_right):

    point_to_delete = point_at_origin
    points_right = hourglass_on_the_right
    points_left = reflect_horizontally(points_right)

    point_to_delete_objects = incdbscan4.insert(point_to_delete)
    point_to_delete_id = point_to_delete_objects[0].id

    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER
    left_objects = incdbscan4.insert(points_left)
    left_ids = [obj.id for obj in left_objects]
    assert_cluster_labels(incdbscan4, left_ids, cluster_1_expected_label)

    cluster_2_expected_label = CLUSTER_LABEL_FIRST_CLUSTER + 1
    right_objects = incdbscan4.insert(points_right)
    right_ids = [obj.id for obj in right_objects]
    assert_cluster_labels(incdbscan4, right_ids, cluster_2_expected_label)

    incdbscan4.delete([point_to_delete_id])

    # Create expected clusters using object IDs
    expected_clusters = [
        left_ids[:3], left_ids[-3:], right_ids[:3], right_ids[-3:]
    ]

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan4, expected_clusters, CLUSTER_LABEL_FIRST_CLUSTER)

    expected_cluster_labels_left = {
        incdbscan4.get_cluster_labels([left_ids[2]])[0],
        incdbscan4.get_cluster_labels([left_ids[4]])[0],
    }

    assert_label_of_object_is_among_possible_ones(
        incdbscan4, left_ids[3], expected_cluster_labels_left)

    expected_cluster_labels_right = {
        incdbscan4.get_cluster_labels([right_ids[2]])[0],
        incdbscan4.get_cluster_labels([right_ids[4]])[0]
    }

    assert_label_of_object_is_among_possible_ones(
        incdbscan4, right_ids[3], expected_cluster_labels_right)


def test_two_non_dense_bridges(incdbscan4, point_at_origin):
    point_to_delete = point_at_origin

    points_left = np.array([
        [-EPS, 0],
        [-EPS, 0],
        [-EPS, -EPS],
        [-EPS, -EPS],
        [-EPS, -EPS * 2],
    ])
    points_right = reflect_horizontally(points_left)

    points_top = np.array([
        [0, EPS],
        [0, EPS],
        [0, EPS * 2],
        [0, EPS * 2],
        [0, EPS * 3],
        [0, EPS * 3],
        [0, EPS * 4],
        [0, EPS * 4],
    ])

    bottom_bridge = np.array([[0, -EPS * 2]])

    all_points = np.vstack([
        point_to_delete, points_left, points_right, points_top, bottom_bridge
    ])

    inserted_objects = incdbscan4.insert(all_points)
    point_to_delete_id = inserted_objects[0].id

    assert_cluster_labels(
        incdbscan4, [obj.id for obj in inserted_objects], CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan4.delete([point_to_delete_id])

    # Get object IDs for the remaining points after deletion
    left_ids = [obj.id for obj in inserted_objects[1:6]]  # points_left
    right_ids = [obj.id for obj in inserted_objects[6:11]]  # points_right
    top_ids = [obj.id for obj in inserted_objects[11:19]]  # points_top

    expected_clusters = [left_ids, right_ids, top_ids]

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan4, expected_clusters, CLUSTER_LABEL_FIRST_CLUSTER)


def test_point_whose_neighbor_changes_its_core_property(incdbscan3):
    some_points_in_the_top = np.array([
        [0, EPS * 1],  # this is the point that is core, then non core, then core
        [0, EPS * 2],  # this is the neighbor of the above point
    ])

    another_point_in_the_top = np.array([
        [0, EPS * 3],
    ])

    all_points_in_the_top = np.vstack([
        some_points_in_the_top,
        another_point_in_the_top
    ])

    point_to_delete = np.array([
        [0, 0],
    ])

    points_right = np.array([
        [EPS * 1, 0],
        [EPS * 2, 0],
        [EPS * 3, 0],
    ])

    # Step 1
    some_points_objects = incdbscan3.insert(some_points_in_the_top)
    some_points_ids = [obj.id for obj in some_points_objects]
    point_to_delete_objects = incdbscan3.insert(point_to_delete)
    point_to_delete_id = point_to_delete_objects[0].id

    assert_cluster_labels(incdbscan3, some_points_ids,
                          CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(
        incdbscan3, [point_to_delete_id], CLUSTER_LABEL_FIRST_CLUSTER)

    # Step 2
    incdbscan3.delete([point_to_delete_id])
    assert_cluster_labels(incdbscan3, some_points_ids, CLUSTER_LABEL_NOISE)

    # Step 3
    cluster_label_second_cluster = CLUSTER_LABEL_FIRST_CLUSTER + 1
    another_point_objects = incdbscan3.insert(another_point_in_the_top)
    another_point_id = another_point_objects[0].id
    point_to_delete_objects = incdbscan3.insert(point_to_delete)
    point_to_delete_id = point_to_delete_objects[0].id
    points_right_objects = incdbscan3.insert(points_right)
    points_right_ids = [obj.id for obj in points_right_objects]

    all_points_in_the_top_ids = some_points_ids + [another_point_id]
    assert_cluster_labels(
        incdbscan3, all_points_in_the_top_ids, cluster_label_second_cluster)
    assert_cluster_labels(
        incdbscan3, [point_to_delete_id], cluster_label_second_cluster)
    assert_cluster_labels(incdbscan3, points_right_ids,
                          cluster_label_second_cluster)

    # Step 4
    expected_clusters = [all_points_in_the_top_ids, points_right_ids]
    incdbscan3.delete([point_to_delete_id])
    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3,
        expected_clusters,
        cluster_label_second_cluster
    )
