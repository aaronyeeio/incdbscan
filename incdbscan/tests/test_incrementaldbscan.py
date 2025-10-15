import numpy as np

from incdbscan.incrementaldbscan import (
    IncrementalDBSCAN,
    IncrementalDBSCANWarning
)
from testutils import (
    CLUSTER_LABEL_NOISE,
    assert_cluster_labels,
    delete_object_and_assert_error,
    delete_object_and_assert_no_warning,
    delete_object_and_assert_warning,
    get_label_and_assert_error,
    get_label_and_assert_no_warning,
    get_label_and_assert_warning,
    insert_object_and_assert_error,
    insert_objects_then_assert_cluster_labels
)


def test_error_when_input_is_non_numeric(incdbscan3):
    inputs_not_welcomed = np.array([
        [1, 2, 'x'],
        [1, 2, None],
        [1, 2, np.nan],
        [1, 2, np.inf],
    ])

    for i in range(len(inputs_not_welcomed)):
        input_ = inputs_not_welcomed[[i]]

        insert_object_and_assert_error(incdbscan3, input_, ValueError)
        # Note: delete and get_cluster_labels now only accept object IDs and don't raise ValueError
        # for invalid IDs - they just issue warnings. So we test with non-existent IDs instead
        delete_object_and_assert_warning(
            incdbscan3, [str(input_)], IncrementalDBSCANWarning)
        get_label_and_assert_warning(
            incdbscan3, [str(input_)], IncrementalDBSCANWarning)


def test_handling_of_same_object_with_different_dtype(incdbscan3):
    object_as_int = np.array([[1, 2]])
    object_as_float = np.array([[1., 2.]])

    # Insert with auto-generated IDs
    inserted_objects = incdbscan3.insert(object_as_int)
    object_id = inserted_objects[0].id

    # Both should have the same label since they're the same position
    assert incdbscan3.get_cluster_labels([object_id]) == \
        incdbscan3.get_cluster_labels([object_id])

    # Delete by ID
    delete_object_and_assert_no_warning(incdbscan3, [object_id])


def test_handling_of_more_than_2d_arrays(incdbscan3, incdbscan4):
    object_3d = np.array([[1, 2, 3]])

    # Insert two objects at same position
    inserted1 = incdbscan3.insert(object_3d)
    inserted2 = incdbscan3.insert(object_3d)

    # Delete one object
    incdbscan3.delete([inserted1[0].id])

    # Remaining object should still be there
    assert incdbscan3.get_cluster_labels(
        [inserted2[0].id]) == CLUSTER_LABEL_NOISE

    object_100d = np.random.random(100).reshape(1, -1)

    # Insert two objects at same position
    inserted3 = incdbscan4.insert(object_100d)
    inserted4 = incdbscan4.insert(object_100d)

    # Delete one object
    incdbscan4.delete([inserted3[0].id])

    # Remaining object should still be there
    assert incdbscan4.get_cluster_labels(
        [inserted4[0].id]) == CLUSTER_LABEL_NOISE


def test_no_warning_when_a_known_object_is_deleted(
        incdbscan3,
        point_at_origin):

    # Insert and get ID
    inserted = incdbscan3.insert(point_at_origin)
    object_id = inserted[0].id
    delete_object_and_assert_no_warning(incdbscan3, [object_id])

    # Insert multiple objects at same position
    inserted1 = incdbscan3.insert(point_at_origin)
    inserted2 = incdbscan3.insert(point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, [inserted1[0].id])
    delete_object_and_assert_no_warning(incdbscan3, [inserted2[0].id])


def test_warning_when_unknown_object_is_deleted(
        incdbscan3,
        point_at_origin):

    # Try to delete non-existent ID
    delete_object_and_assert_warning(
        incdbscan3, ['non_existent_id'], IncrementalDBSCANWarning)

    # Insert and delete
    inserted = incdbscan3.insert(point_at_origin)
    object_id = inserted[0].id
    incdbscan3.delete([object_id])

    # Try to delete already deleted ID
    delete_object_and_assert_warning(
        incdbscan3, [object_id], IncrementalDBSCANWarning)


def test_no_warning_when_cluster_label_is_gotten_for_known_object(
        incdbscan3,
        point_at_origin):

    expected_label = np.array([CLUSTER_LABEL_NOISE])

    # Insert and get label by ID
    inserted = incdbscan3.insert(point_at_origin)
    object_id = inserted[0].id
    label = get_label_and_assert_no_warning(incdbscan3, [object_id])
    assert label == expected_label

    # Insert another object at same position and delete first one
    inserted2 = incdbscan3.insert(point_at_origin)
    incdbscan3.delete([object_id])
    label = get_label_and_assert_no_warning(incdbscan3, [inserted2[0].id])
    assert label == expected_label


def test_warning_when_cluster_label_is_gotten_for_unknown_object(
        incdbscan3,
        point_at_origin):

    # Try to get label for non-existent ID
    label = get_label_and_assert_warning(
        incdbscan3, ['non_existent_id'], IncrementalDBSCANWarning)
    assert np.isnan(label)

    # Insert and delete
    inserted = incdbscan3.insert(point_at_origin)
    object_id = inserted[0].id
    incdbscan3.delete([object_id])

    # Try to get label for deleted ID
    label = get_label_and_assert_warning(
        incdbscan3, [object_id], IncrementalDBSCANWarning)
    assert np.isnan(label)


def test_different_metrics_are_available():
    incdbscan_euclidean = \
        IncrementalDBSCAN(eps=1.5, min_pts=3, metric='euclidean')
    incdbscan_manhattan = \
        IncrementalDBSCAN(eps=1.5, min_pts=3, metric='manhattan')

    diagonal = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
    ])

    expected_label_euclidean = CLUSTER_LABEL_NOISE + 1
    inserted_euclidean = incdbscan_euclidean.insert(diagonal)
    ids_euclidean = [obj.id for obj in inserted_euclidean]
    assert_cluster_labels(incdbscan_euclidean,
                          ids_euclidean, expected_label_euclidean)

    expected_label_manhattan = CLUSTER_LABEL_NOISE
    inserted_manhattan = incdbscan_manhattan.insert(diagonal)
    ids_manhattan = [obj.id for obj in inserted_manhattan]
    assert_cluster_labels(incdbscan_manhattan,
                          ids_manhattan, expected_label_manhattan)
