"""Saved top-down object and structural map diagnostics."""

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.map_visualisation import draw_persistent_map_top_down
from qmapnav.mapping.object_map import PersistentObjectRecord
from qmapnav.mapping.structural_map import StructuralAnchor


def test_top_down_map_draws_path_object_wall_and_anchor() -> None:
    instance = ObjectInstance(
        0,
        {'chair': 1.0},
        {},
        np.array([0.0, 1.0, 0.5]),
        np.array([-0.3, 0.75, 0.0]),
        np.array([0.3, 1.25, 1.0]),
        np.array([0.6, 0.5, 1.0]),
        0.1,
        0.8,
        2,
        0.8,
    )
    record = PersistentObjectRecord(
        instance,
        'chair',
        0.8,
        1,
        2,
        ('a', 'b'),
        ('one', 'two'),
        None,
        0.0,
        'active',
        np.array([[0.0, 1.0, 0.5]]),
        (),
        'candidate',
    )
    wall = StructuralAnchor(
        'wall_0000',
        'wall',
        'wall',
        np.array([0.0, 2.0, 1.0]),
        np.array([[-2.0, 2.0], [2.0, 2.0]]),
        None,
        np.array([0.0, 1.0, 0.0, -2.0]),
        np.array([4.0, 0.05, 2.0]),
        0.0,
        None,
        0.9,
        1,
        2,
        ('scan',),
        (),
    )
    window = StructuralAnchor(
        'anchor_0000',
        'window',
        'window',
        np.array([0.0, 2.0, 1.0]),
        None,
        None,
        wall.plane_parameters,
        np.array([1.0, 0.05, 1.0]),
        0.0,
        wall.anchor_id,
        0.8,
        1,
        2,
        ('a', 'b'),
        ('window',),
    )

    image = draw_persistent_map_top_down(
        [record],
        [wall],
        [window],
        np.array([[0.0, 0.0], [0.5, 0.5]]),
        size_px=300,
    )

    assert image.shape == (300, 300, 3)
    assert image.dtype == np.uint8
    assert np.any(image != 245)
