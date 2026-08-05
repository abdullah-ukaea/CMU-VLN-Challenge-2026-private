"""
Small-object observation mode and the class-to-support search prior.

The lookup below is deliberately small and grounded in the released question
vocabulary. It is a search hint that decides *where to look first*, never a
reasoning shortcut: a target is still only accepted once perception and the
persistent maps actually observe it.
"""

from dataclasses import dataclass
from math import isfinite


#: Released-corpus classes whose instances are small enough that a normal
#: stand-off view rarely yields usable geometry.
SMALL_OBJECT_CLASSES = frozenset(
    {
        'book',
        'bottle',
        'bowl',
        'crystal_ball_decoration',
        'cup',
        'elephant_figurine',
        'figurine',
        'folder',
        'horse_figurine',
        'jar',
        'kettle',
        'magazine',
        'mug',
        'paper_cup',
        'remote_control',
        'soccer_ball',
        'sphere_decoration',
        'tray',
        'vase',
    }
)

#: Ordered support classes searched when a small target has not been seen.
LIKELY_SUPPORTS: dict[str, tuple[str, ...]] = {
    'book': ('shelf', 'table', 'desk', 'stool', 'cabinet'),
    'bottle': ('table', 'desk', 'counter', 'shelf'),
    'bowl': ('table', 'counter', 'desk'),
    'crystal_ball_decoration': ('shelf', 'cabinet', 'table'),
    'cup': ('table', 'desk', 'counter'),
    'figurine': ('shelf', 'table', 'cabinet'),
    'folder': ('cabinet', 'desk', 'shelf', 'table'),
    'jar': ('table', 'shelf', 'counter'),
    'kettle': ('table', 'counter', 'dining_table', 'coffee_table'),
    'magazine': ('ottoman', 'table', 'coffee_table', 'shelf'),
    'monitor': ('desk', 'table'),
    'mug': ('table', 'desk', 'counter'),
    'paper_cup': ('table', 'desk', 'counter'),
    'remote_control': ('table', 'coffee_table', 'sofa'),
    'sphere_decoration': ('cabinet', 'shelf', 'table'),
    'tray': ('table', 'counter', 'cabinet'),
    'vase': ('table', 'cabinet', 'shelf', 'display_ledge'),
}

#: Classes that can physically support a small object.
SUPPORT_SURFACE_CLASSES = frozenset(
    {
        'bookcase',
        'cabinet',
        'coffee_table',
        'counter',
        'desk',
        'dining_table',
        'display_ledge',
        'dressing_table',
        'kitchen_counter',
        'nightstand',
        'ottoman',
        'shelf',
        'small_table',
        'stool',
        'table',
        'tea_table',
    }
)

#: Support classes observed best from a shelf-style shorter stand-off.
_SHELF_LIKE = frozenset(
    {'bookcase', 'cabinet', 'display_ledge', 'shelf'}
)


def likely_supports(class_name: str) -> tuple[str, ...]:
    """Return the ordered support classes to search for a target class."""
    return LIKELY_SUPPORTS.get(class_name, ())


def is_support_surface(class_name: str) -> bool:
    """Return whether a class can hold a small object."""
    return class_name in SUPPORT_SURFACE_CLASSES


def is_shelf_like(class_name: str) -> bool:
    """Return whether a support is viewed best from a shorter stand-off."""
    return class_name in _SHELF_LIKE


@dataclass(frozen=True)
class SmallObjectObservationMode:
    """The observation policy applied while hunting a small target."""

    active: bool
    target_classes: tuple[str, ...] = ()
    support_instance_ids: tuple[str, ...] = ()
    preferred_distance_m: float = 1.6
    pause_duration_sec: float = 2.5
    detector_threshold_override: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise TypeError('active must be bool')
        object.__setattr__(
            self, 'target_classes', tuple(self.target_classes)
        )
        object.__setattr__(
            self, 'support_instance_ids', tuple(self.support_instance_ids)
        )
        for name in ('preferred_distance_m', 'pause_duration_sec'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        override = self.detector_threshold_override
        if override is not None and (
            not isfinite(override) or not 0.0 < override < 1.0
        ):
            raise ValueError('detector_threshold_override must be in (0, 1)')

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'active': self.active,
            'target_classes': list(self.target_classes),
            'support_instance_ids': list(self.support_instance_ids),
            'preferred_distance_m': self.preferred_distance_m,
            'pause_duration_sec': self.pause_duration_sec,
            'detector_threshold_override': self.detector_threshold_override,
        }


@dataclass(frozen=True)
class SmallObjectTriggerConfig:
    """Thresholds deciding when small-object mode switches on."""

    max_box_area_ratio: float = 0.01
    min_projected_points: int = 40
    min_detection_confidence: float = 0.35
    detector_threshold_override: float | None = 0.15
    tabletop_distance_m: float = 1.6
    shelf_distance_m: float = 1.4
    pause_duration_sec: float = 2.5

    def __post_init__(self) -> None:
        if not isfinite(self.max_box_area_ratio) or not (
            0.0 < self.max_box_area_ratio < 1.0
        ):
            raise ValueError('max_box_area_ratio must lie in (0, 1)')
        if isinstance(self.min_projected_points, bool) or (
            self.min_projected_points < 0
        ):
            raise ValueError('min_projected_points must be non-negative')
        if not isfinite(self.min_detection_confidence) or not (
            0.0 <= self.min_detection_confidence <= 1.0
        ):
            raise ValueError('min_detection_confidence must lie in [0, 1]')


def decide_small_object_mode(
    target_class: str,
    *,
    config: SmallObjectTriggerConfig | None = None,
    detected: bool = False,
    box_area_ratio: float | None = None,
    projected_points: int | None = None,
    detection_confidence: float | None = None,
    support_instance_ids: tuple[str, ...] = (),
    support_class: str | None = None,
) -> SmallObjectObservationMode:
    """
    Activate small-object mode from class priors and observed evidence.

    Activation deliberately requires either a known-small class or measured
    evidence of a tiny/sparse detection, so an ordinary chair never enters
    small-object mode merely for being smaller than a table.
    """
    policy = config or SmallObjectTriggerConfig()
    reasons_active = False
    if target_class in SMALL_OBJECT_CLASSES:
        reasons_active = True
    if box_area_ratio is not None and (
        box_area_ratio <= policy.max_box_area_ratio
    ):
        reasons_active = True
    if projected_points is not None and (
        projected_points < policy.min_projected_points
    ):
        reasons_active = True
    if detected and detection_confidence is not None and (
        detection_confidence < policy.min_detection_confidence
    ):
        reasons_active = True
    if not detected and likely_supports(target_class):
        reasons_active = True

    if not reasons_active:
        return SmallObjectObservationMode(active=False)
    distance = (
        policy.shelf_distance_m
        if support_class is not None and is_shelf_like(support_class)
        else policy.tabletop_distance_m
    )
    return SmallObjectObservationMode(
        active=True,
        target_classes=(target_class,),
        support_instance_ids=tuple(support_instance_ids),
        preferred_distance_m=distance,
        pause_duration_sec=policy.pause_duration_sec,
        detector_threshold_override=policy.detector_threshold_override,
    )


__all__ = [
    'LIKELY_SUPPORTS',
    'SMALL_OBJECT_CLASSES',
    'SUPPORT_SURFACE_CLASSES',
    'SmallObjectObservationMode',
    'SmallObjectTriggerConfig',
    'decide_small_object_mode',
    'is_shelf_like',
    'is_support_surface',
    'likely_supports',
]
