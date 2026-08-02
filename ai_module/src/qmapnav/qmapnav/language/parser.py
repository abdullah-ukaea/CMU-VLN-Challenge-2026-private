"""Full and degraded deterministic parsers for challenge questions."""

from dataclasses import dataclass
import re

from qmapnav.common import EntityReference
from qmapnav.common import RelationConstraint
from qmapnav.common import RouteConstraint
from qmapnav.common import RouteStep
from qmapnav.common import TaskSpecification
from qmapnav.language.classification import classify_task_type
from qmapnav.language.classification import INSTRUCTION_FOLLOWING
from qmapnav.language.classification import NUMERICAL
from qmapnav.language.classification import OBJECT_REFERENCE
from qmapnav.language.extraction import AvoidancePhrase
from qmapnav.language.extraction import extract_language_features
from qmapnav.language.extraction import LanguageExtraction
from qmapnav.language.extraction import Mention


_ROUTE_ACTION_PATTERNS = (
    (re.compile(r'\btake\s+the\s+path\s+between\s+', re.IGNORECASE),
     'pass_between'),
    (re.compile(r'\btake\s+the\s+path\s+near\s+', re.IGNORECASE),
     'pass_near'),
    (re.compile(r'\bpass\s+between\s+', re.IGNORECASE), 'pass_between'),
    (re.compile(r'\bpass\s+by\s+', re.IGNORECASE), 'pass_by'),
    (re.compile(r'\bgo\s+between\s+', re.IGNORECASE), 'pass_between'),
    (re.compile(r'\bgo\s+near\s+', re.IGNORECASE), 'go_near'),
    (re.compile(r'\bgo\s+to\s+', re.IGNORECASE), 'go_to'),
    (re.compile(r'\bstop\s+(?:at|by)\s+', re.IGNORECASE), 'stop_at'),
    (re.compile(r'\bstop\s+near\s+', re.IGNORECASE), 'stop_near'),
    (re.compile(r'\bfinish\s+at\s+', re.IGNORECASE), 'stop_at'),
    (
        re.compile(r'\bfinish\s+(?:near|beside)\s+', re.IGNORECASE),
        'stop_near',
    ),
)
_DESTINATION_ACTIONS = frozenset({'go_near', 'go_to', 'stop_at', 'stop_near'})
_DEGRADED_INSTRUCTION_PATTERN = re.compile(
    r'\b(?:avoid|finish|go|head|move|navigate|pass|proceed|stay|stop|walk)\b',
    re.IGNORECASE,
)
_DEGRADED_NUMERICAL_PATTERN = re.compile(
    r'\b(?:count|how\s+many|number\s+of)\b',
    re.IGNORECASE,
)


class FullParseError(ValueError):
    """Indicate that a question is outside the supported full grammar."""


@dataclass(frozen=True)
class _BoundEntity:
    """Associate an extracted source mention with its shared entity contract."""

    mention: Mention
    reference: EntityReference


@dataclass(frozen=True)
class _ActionCue:
    """One recognized route action and its source span."""

    start: int
    end: int
    action: str


def _validate_question(question: str) -> None:
    if not isinstance(question, str):
        raise TypeError('question must be a string')
    if not question.strip():
        raise ValueError('question must not be empty')


def _entity_mentions(extraction: LanguageExtraction) -> tuple[Mention, ...]:
    return tuple(
        sorted(
            extraction.object_nouns + extraction.structural_entities,
            key=lambda mention: (mention.start, mention.end),
        )
    )


def _feature_before_entity(
    features: tuple[Mention, ...],
    entity: Mention,
    previous_entity_end: int,
) -> Mention | None:
    candidates = [
        feature
        for feature in features
        if previous_entity_end <= feature.start and feature.end <= entity.start
    ]
    return candidates[-1] if candidates else None


def _cardinality_before_entity(
    extraction: LanguageExtraction,
    entity: Mention,
    previous_entity_end: int,
) -> int | None:
    candidates = [
        cardinality
        for cardinality in extraction.cardinalities
        if (
            previous_entity_end <= cardinality.start
            and cardinality.end <= entity.start
        )
    ]
    return candidates[-1].value if candidates else None


def _build_entities(
    extraction: LanguageExtraction,
) -> tuple[_BoundEntity, ...]:
    mentions = _entity_mentions(extraction)
    occurrence_counts: dict[str, int] = {}
    entities = []
    previous_entity_end = 0

    for mention in mentions:
        occurrence = occurrence_counts.get(mention.normalized, 0) + 1
        occurrence_counts[mention.normalized] = occurrence
        colour = _feature_before_entity(
            extraction.colours,
            mention,
            previous_entity_end,
        )
        attributes = {'colour': colour.normalized} if colour else {}
        for attribute in extraction.attributes:
            if (
                previous_entity_end <= attribute.start
                and attribute.end <= mention.start
            ):
                attributes[attribute.attribute_name] = attribute.normalized
        reference = EntityReference(
            entity_id=f'{mention.normalized}_{occurrence}',
            class_name=mention.normalized,
            attributes=attributes,
            cardinality=_cardinality_before_entity(
                extraction,
                mention,
                previous_entity_end,
            ),
            source_text=mention.source_text,
        )
        entities.append(_BoundEntity(mention, reference))
        previous_entity_end = mention.end

    return tuple(entities)


def _is_route_relation(question: str, relation: Mention) -> bool:
    prefix = question[max(0, relation.start - 32):relation.start].lower()
    if relation.normalized == 'between':
        return bool(re.search(r'\b(?:go|pass|path)\s*$', prefix))
    if relation.normalized == 'near':
        return bool(re.search(r'\b(?:finish|go|path|stop)\s*$', prefix))
    if relation.normalized == 'beside':
        return bool(re.search(r'\bfinish\s*$', prefix))
    return False


def _previous_entity(
    entities: tuple[_BoundEntity, ...],
    position: int,
) -> _BoundEntity | None:
    candidates = [entity for entity in entities if entity.mention.end <= position]
    return candidates[-1] if candidates else None


def _next_entities(
    entities: tuple[_BoundEntity, ...],
    position: int,
    end: int | None = None,
) -> list[_BoundEntity]:
    return [
        entity
        for entity in entities
        if entity.mention.start >= position
        and (end is None or entity.mention.start < end)
    ]


def _pronoun_anchor(
    question: str,
    relation: Mention,
    entities: tuple[_BoundEntity, ...],
    subject: _BoundEntity,
) -> _BoundEntity | None:
    tail = question[relation.end:relation.end + 12]
    if not re.match(r'\s+(?:it|them)\b', tail, re.IGNORECASE):
        return None
    subject_index = entities.index(subject)
    if subject_index < 1:
        return None
    return entities[subject_index - 1]


def _between_anchors(
    candidates: list[_BoundEntity],
) -> list[_BoundEntity]:
    if not candidates:
        return []
    if candidates[0].reference.cardinality == 2:
        return candidates[:1]
    return candidates[:2]


def _build_relations(
    question: str,
    extraction: LanguageExtraction,
    entities: tuple[_BoundEntity, ...],
    *,
    strict: bool,
) -> list[RelationConstraint]:
    constraints = []
    for relation in extraction.spatial_relations:
        if _is_route_relation(question, relation):
            continue

        subject = _previous_entity(entities, relation.start)
        pronoun = None
        if subject is not None:
            pronoun = _pronoun_anchor(question, relation, entities, subject)
        if pronoun is not None:
            anchors = [pronoun]
        else:
            candidates = _next_entities(entities, relation.end)
            if relation.normalized == 'between':
                anchors = _between_anchors(candidates)
            else:
                anchors = candidates[:1]

        if subject is None or not anchors:
            if strict:
                raise FullParseError(
                    f'could not bind relation {relation.source_text!r}'
                )
            continue

        constraints.append(
            RelationConstraint(
                relation.normalized,
                subject.reference.entity_id,
                [anchor.reference.entity_id for anchor in anchors],
            )
        )
    return constraints


def _inside_avoidance(
    start: int,
    avoidance_phrases: tuple[AvoidancePhrase, ...],
) -> bool:
    return any(
        phrase.start <= start < phrase.end for phrase in avoidance_phrases
    )


def _action_cues(
    question: str,
    avoidance_phrases: tuple[AvoidancePhrase, ...],
) -> tuple[_ActionCue, ...]:
    candidates = []
    for pattern, action in _ROUTE_ACTION_PATTERNS:
        for match in pattern.finditer(question):
            if not _inside_avoidance(match.start(), avoidance_phrases):
                candidates.append(
                    _ActionCue(match.start(), match.end(), action)
                )

    candidates.sort(key=lambda cue: (cue.start, -(cue.end - cue.start)))
    selected = []
    for cue in candidates:
        if any(
            cue.start < existing.end and existing.start < cue.end
            for existing in selected
        ):
            continue
        selected.append(cue)
    return tuple(selected)


def _entity_for_mention(
    entities: tuple[_BoundEntity, ...],
    mention: Mention,
) -> _BoundEntity | None:
    return next(
        (
            entity
            for entity in entities
            if (
                entity.mention.start == mention.start
                and entity.mention.end == mention.end
                and entity.mention.normalized == mention.normalized
            )
        ),
        None,
    )


def _build_route_steps(
    question: str,
    extraction: LanguageExtraction,
    entities: tuple[_BoundEntity, ...],
    terminal: _BoundEntity | None,
    *,
    strict: bool,
) -> list[RouteStep]:
    cues = _action_cues(question, extraction.avoidance_phrases)
    steps = []
    for cue_index, cue in enumerate(cues):
        next_cue_start = (
            cues[cue_index + 1].start
            if cue_index + 1 < len(cues)
            else None
        )
        candidates = _next_entities(entities, cue.end, next_cue_start)
        if cue.action == 'pass_between':
            targets = _between_anchors(candidates)
        else:
            targets = candidates[:1]

        if not targets:
            if strict:
                raise FullParseError(
                    f'route action {cue.action!r} has no entity target'
                )
            continue
        steps.append(
            RouteStep(
                len(steps),
                cue.action,
                [target.reference.entity_id for target in targets],
            )
        )

    terminal_already_targeted = terminal is not None and any(
        step.action in _DESTINATION_ACTIONS
        and step.entity_ids[0] == terminal.reference.entity_id
        for step in steps
    )
    if terminal is not None and not terminal_already_targeted:
        steps.append(
            RouteStep(len(steps), 'go_to', [terminal.reference.entity_id])
        )
    return steps


def _entities_in_phrase(
    entities: tuple[_BoundEntity, ...],
    phrase: AvoidancePhrase,
) -> list[_BoundEntity]:
    return [
        entity
        for entity in entities
        if phrase.start <= entity.mention.start < phrase.end
    ]


def _build_forbidden_constraints(
    extraction: LanguageExtraction,
    entities: tuple[_BoundEntity, ...],
    *,
    strict: bool,
) -> list[RouteConstraint]:
    constraints = []
    for phrase in extraction.avoidance_phrases:
        candidates = _entities_in_phrase(entities, phrase)
        if phrase.constraint_type == 'avoid_between':
            targets = _between_anchors(candidates)
        else:
            targets = candidates[:1]
        if not targets:
            if strict:
                raise FullParseError(
                    f'avoidance phrase {phrase.source_text!r} has no entity'
                )
            continue
        constraints.append(
            RouteConstraint(
                phrase.constraint_type,
                [target.reference.entity_id for target in targets],
            )
        )
    return constraints


def _terminal_entity(
    extraction: LanguageExtraction,
    entities: tuple[_BoundEntity, ...],
) -> _BoundEntity | None:
    if extraction.terminal_target is None:
        return None
    return _entity_for_mention(entities, extraction.terminal_target.target)


def _full_task(question: str) -> TaskSpecification:
    try:
        task_type = classify_task_type(question)
        extraction = extract_language_features(question)
    except ValueError as error:
        raise FullParseError(str(error)) from error

    entities = _build_entities(extraction)
    if not entities:
        raise FullParseError('no entities were extracted')

    relations = _build_relations(
        question,
        extraction,
        entities,
        strict=True,
    )
    route_steps = []
    forbidden_constraints = []
    terminal = None
    if task_type == INSTRUCTION_FOLLOWING:
        terminal = _terminal_entity(extraction, entities)
        if terminal is None:
            raise FullParseError('instruction has no terminal target')
        route_steps = _build_route_steps(
            question,
            extraction,
            entities,
            terminal,
            strict=True,
        )
        forbidden_constraints = _build_forbidden_constraints(
            extraction,
            entities,
            strict=True,
        )
        if not route_steps:
            raise FullParseError('instruction has no route steps')

    return TaskSpecification(
        task_type=task_type,
        entities=[entity.reference for entity in entities],
        relations=relations,
        ordered_route_steps=route_steps,
        forbidden_constraints=forbidden_constraints,
        terminal_target=terminal.reference if terminal else None,
        parse_confidence=1.0,
        parse_mode='full',
    )


def parse_question_full(question: str) -> TaskSpecification:
    """
    Parse a question using only the explicitly supported deterministic grammar.

    A ``FullParseError`` signals that callers should invoke the degraded parser.
    """
    _validate_question(question)
    return _full_task(question)


def _infer_degraded_task_type(question: str) -> tuple[str, bool]:
    try:
        return classify_task_type(question), True
    except ValueError:
        if _DEGRADED_NUMERICAL_PATTERN.search(question):
            return NUMERICAL, False
        if _DEGRADED_INSTRUCTION_PATTERN.search(question):
            return INSTRUCTION_FOLLOWING, False
        return OBJECT_REFERENCE, False


def _degraded_confidence(
    *,
    classified: bool,
    entity_count: int,
    relation_count: int,
    route_step_count: int,
    has_terminal: bool,
) -> float:
    confidence = 0.10
    confidence += 0.15 if classified else 0.0
    confidence += 0.20 if entity_count else 0.0
    confidence += 0.10 if relation_count else 0.0
    confidence += 0.10 if route_step_count else 0.0
    confidence += 0.10 if has_terminal else 0.0
    return min(confidence, 0.65)


def parse_question_degraded(question: str) -> TaskSpecification:
    """
    Recover a deterministic partial task from any non-empty question.

    Ungroundable relations and constraints are omitted. For an instruction, the
    last recognized entity becomes the fallback terminal target so later stages
    can still attempt partial credit.
    """
    _validate_question(question)
    task_type, classified = _infer_degraded_task_type(question)
    extraction = extract_language_features(question)
    entities = _build_entities(extraction)
    relations = _build_relations(
        question,
        extraction,
        entities,
        strict=False,
    )

    route_steps = []
    forbidden_constraints = []
    terminal = None
    if task_type == INSTRUCTION_FOLLOWING:
        terminal = _terminal_entity(extraction, entities)
        if terminal is None and entities:
            terminal = entities[-1]
        route_steps = _build_route_steps(
            question,
            extraction,
            entities,
            terminal,
            strict=False,
        )
        forbidden_constraints = _build_forbidden_constraints(
            extraction,
            entities,
            strict=False,
        )
        if terminal is not None and not route_steps:
            route_steps = [
                RouteStep(0, 'go_near', [terminal.reference.entity_id])
            ]

    confidence = _degraded_confidence(
        classified=classified,
        entity_count=len(entities),
        relation_count=len(relations),
        route_step_count=len(route_steps),
        has_terminal=terminal is not None,
    )
    return TaskSpecification(
        task_type=task_type,
        entities=[entity.reference for entity in entities],
        relations=relations,
        ordered_route_steps=route_steps,
        forbidden_constraints=forbidden_constraints,
        terminal_target=terminal.reference if terminal else None,
        parse_confidence=confidence,
        parse_mode='degraded',
    )


def parse_question(question: str) -> TaskSpecification:
    """Parse with the full grammar and deterministically degrade on failure."""
    _validate_question(question)
    try:
        return parse_question_full(question)
    except FullParseError:
        return parse_question_degraded(question)


__all__ = [
    'FullParseError',
    'parse_question',
    'parse_question_degraded',
    'parse_question_full',
]
