"""Build a bounded detector vocabulary from a parsed task specification."""

from collections.abc import Mapping
from collections.abc import Sequence

from qmapnav.common import TaskSpecification
from qmapnav.perception.contracts import DetectorClass


def detector_classes_from_task_specification(
    task: TaskSpecification,
    class_aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[DetectorClass, ...]:
    """Return unique query-conditioned classes in first-mention order."""
    if not isinstance(task, TaskSpecification):
        raise TypeError('task must be a TaskSpecification')
    aliases = {} if class_aliases is None else dict(class_aliases)
    classes = []
    seen = set()
    for entity in task.entities:
        canonical_name = entity.class_name
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        base_prompt = canonical_name.replace('_', ' ')
        prompts = [base_prompt]
        for alias in aliases.get(canonical_name, ()):
            if alias.casefold() not in {item.casefold() for item in prompts}:
                prompts.append(alias)
        classes.append(DetectorClass(canonical_name, tuple(prompts)))
    if not classes:
        raise ValueError('task must contain at least one detector entity')
    return tuple(classes)
