"""Two-view colour fusion and relation-reevaluation regression."""

from fixtures import make_candidate
from fixtures import make_observation
from fixtures import support_geometry

import numpy as np

from qmapnav.mapping.object_map import ObjectMap
from qmapnav.reasoning.colour_classifier import classify_colour
from qmapnav.reasoning.colour_features import extract_colour_features
from qmapnav.reasoning.colour_pixel_filter import filter_reliable_pixels
from qmapnav.reasoning.colour_pixel_filter import select_object_pixels
from qmapnav.reasoning.colour_prototypes import fit_colour_prototypes_from_rgb
from qmapnav.reasoning.relation_graph import RelationGraph
from qmapnav.reasoning.support_relations import on_evidence


def _prototypes():
    rgb = {
        'black': [0, 0, 0], 'blue': [30, 80, 210],
        'brown': [100, 60, 20], 'green': [20, 150, 60],
        'grey': [110, 110, 110], 'orange': [240, 120, 20],
        'pink': [240, 140, 170], 'purple': [110, 40, 160],
        'red': [210, 30, 30], 'white': [245, 245, 245],
        'yellow': [230, 210, 30],
    }
    return fit_colour_prototypes_from_rgb({
        name: np.tile(value, (3, 1)) for name, value in rgb.items()
    })


def _classify(crop, viewpoint):
    mask = np.ones(crop.shape[:2], dtype=np.bool_)
    selection = select_object_pixels(crop, segmentation_mask=mask)
    pixels = filter_reliable_pixels(selection)
    features = extract_colour_features(pixels)
    return classify_colour(
        features, pixels, _prototypes(), source_viewpoint_id=viewpoint,
        source_detection_id=f'chair_{viewpoint}',
    )


def test_two_view_colour_stability_and_relation_geometry_update() -> None:
    object_map = ObjectMap()
    chair_a = make_candidate(
        'chair_a', (1.0, 2.0, 0.5),
        class_name='chair', dimensions=(0.6, 0.5, 1.0), timestamp_ns=1,
    )
    chair_id = object_map.add_or_update(
        chair_a, make_observation(chair_a, 'view_a', timestamp_ns=1)
    )
    blue_crop = np.full((30, 30, 3), [30, 80, 210], dtype=np.uint8)
    first = _classify(blue_crop, 'view_a')
    object_map.update_colour(chair_id, first)

    chair_b = make_candidate(
        'chair_b', (1.02, 2.01, 0.5),
        class_name='chair', dimensions=(0.6, 0.5, 1.0), timestamp_ns=2,
    )
    assert object_map.add_or_update(
        chair_b, make_observation(chair_b, 'view_b', timestamp_ns=2)
    ) == chair_id
    contaminated = blue_crop.copy()
    contaminated[:, 15:] = [230, 30, 30]
    second = _classify(contaminated, 'view_b')
    object_map.update_colour(
        chair_id, second, crop_quality=0.2, mask_quality=0.2,
        geometry_support=0.3,
    )

    table = support_geometry('table', 'table', (0.0, 0.0, 0.4), (1.5, 1.0, 0.8))
    book = support_geometry('book', 'book', (0.0, 0.0, 0.87), (0.2, 0.2, 0.1))
    picture = support_geometry(
        'picture', 'picture', (0.0, 0.2, 2.0), (0.8, 0.1, 0.8)
    )
    graph = RelationGraph()
    graph.recompute([book, picture, table])
    initial_keys = {(item.relation, item.subject_id, item.anchor_id)
                    for item in graph.edges}
    raised_book = support_geometry(
        'book', 'book', (0.0, 0.0, 1.4), (0.2, 0.2, 0.1)
    )
    graph.recompute([raised_book, picture, table])
    updated_keys = {(item.relation, item.subject_id, item.anchor_id)
                    for item in graph.edges}

    record = object_map.record(chair_id)
    assert record.instance.colour_scores['blue'] > 0.8
    assert record.instance.observation_count == 2
    assert on_evidence(book, table).accepted
    assert ('on', 'book', 'table') in initial_keys
    assert ('above', 'picture', 'table') in initial_keys
    assert ('on', 'picture', 'table') not in initial_keys
    assert ('on', 'book', 'table') not in updated_keys
    assert graph.revision == 2
