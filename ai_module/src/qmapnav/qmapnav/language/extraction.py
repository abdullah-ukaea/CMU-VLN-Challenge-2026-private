"""Deterministic lexical feature extraction for challenge questions."""

from dataclasses import dataclass
import re
from typing import Iterable

from qmapnav.common.colour_vocabulary import COLOUR_ALIASES
from qmapnav.common.colour_vocabulary import COLOUR_CLASSES
from qmapnav.common.colour_vocabulary import normalize_colour_name
from qmapnav.language.classification import classify_task_type
from qmapnav.language.classification import INSTRUCTION_FOLLOWING


@dataclass(frozen=True)
class Mention:
    """One normalized language feature and its half-open source span."""

    source_text: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class CardinalityMention:
    """One explicit positive cardinality and its half-open source span."""

    source_text: str
    value: int
    start: int
    end: int


@dataclass(frozen=True)
class AttributeMention:
    """One normalized non-colour entity attribute and source span."""

    source_text: str
    attribute_name: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class AvoidancePhrase:
    """One forbidden-path phrase found in an instruction."""

    source_text: str
    constraint_type: str
    start: int
    end: int


@dataclass(frozen=True)
class TerminalTarget:
    """The final destination head noun and the action that introduced it."""

    target: Mention
    action: str


@dataclass(frozen=True)
class LanguageExtraction:
    """Lexical features consumed later by the full and degraded parsers."""

    object_nouns: tuple[Mention, ...]
    colours: tuple[Mention, ...]
    attributes: tuple[AttributeMention, ...]
    cardinalities: tuple[CardinalityMention, ...]
    structural_entities: tuple[Mention, ...]
    spatial_relations: tuple[Mention, ...]
    ordering_terms: tuple[Mention, ...]
    avoidance_phrases: tuple[AvoidancePhrase, ...]
    terminal_target: TerminalTarget | None


# Longer variants win over shorter overlapping variants. The vocabulary covers
# every released class mention while retaining common singular/plural variants
# likely to occur in held-out questions.
_OBJECT_NOUNS = {
    'bed': 'bed',
    'beds': 'bed',
    'beer bottle': 'beer_bottle',
    'beer bottles': 'beer_bottle',
    'bench': 'bench',
    'benches': 'bench',
    'bedside table': 'bedside_table',
    'bedside tables': 'bedside_table',
    'big picture': 'picture',
    'book': 'book',
    'books': 'book',
    'bowl': 'bowl',
    'bowls': 'bowl',
    'box': 'box',
    'boxes': 'box',
    'cabinet': 'cabinet',
    'cabinets': 'cabinet',
    'calligraphy painting': 'calligraphy_painting',
    'calligraphy paintings': 'calligraphy_painting',
    'chair': 'chair',
    'chairs': 'chair',
    'clock': 'clock',
    'clocks': 'clock',
    'coffee table': 'coffee_table',
    'coffee tables': 'coffee_table',
    'computer monitor': 'computer_monitor',
    'computer monitors': 'computer_monitor',
    'couch': 'couch',
    'couches': 'couch',
    'crystal ball decoration': 'crystal_ball_decoration',
    'crystal ball decorations': 'crystal_ball_decoration',
    'cup': 'cup',
    'cup of coffee': 'cup',
    'cups': 'cup',
    'dining table': 'dining_table',
    'dining tables': 'dining_table',
    'dressing table': 'dressing_table',
    'dressing tables': 'dressing_table',
    'easel': 'easel',
    'easels': 'easel',
    'elephant figurine': 'elephant_figurine',
    'elephant figurines': 'elephant_figurine',
    'fan decoration': 'fan_decoration',
    'fan decorations': 'fan_decoration',
    'file cabinet': 'file_cabinet',
    'file cabinets': 'file_cabinet',
    'flowers': 'flower',
    'folder': 'folder',
    'folders': 'folder',
    'fossil decoration': 'fossil_decoration',
    'fossil decorations': 'fossil_decoration',
    'framed record': 'framed_record',
    'framed records': 'framed_record',
    'guitar': 'guitar',
    'guitars': 'guitar',
    'hookah': 'hookah',
    'hookahs': 'hookah',
    'horse figurine': 'horse_figurine',
    'horse figurines': 'horse_figurine',
    'jar': 'jar',
    'jars': 'jar',
    'kettle': 'kettle',
    'kettles': 'kettle',
    'knife rack': 'knife_rack',
    'knife racks': 'knife_rack',
    'kitchen island': 'kitchen_island',
    'kitchen islands': 'kitchen_island',
    'lamp': 'lamp',
    'lamps': 'lamp',
    'lantern': 'lantern',
    'lanterns': 'lantern',
    'magazine': 'magazine',
    'magazines': 'magazine',
    'microwave': 'microwave',
    'microwaves': 'microwave',
    'mirror': 'mirror',
    'mirrors': 'mirror',
    'nightstand': 'nightstand',
    'nightstands': 'nightstand',
    'night stand': 'nightstand',
    'night stands': 'nightstand',
    'ottoman': 'ottoman',
    'ottomans': 'ottoman',
    'painting': 'painting',
    'paintings': 'painting',
    'paper cup': 'paper_cup',
    'paper cups': 'paper_cup',
    'phone': 'phone',
    'phones': 'phone',
    'photo': 'photo',
    'photos': 'photo',
    'picture': 'picture',
    'pictures': 'picture',
    'pillow': 'pillow',
    'pillows': 'pillow',
    'potted plant': 'potted_plant',
    'potted plants': 'potted_plant',
    'plant': 'plant',
    'plants': 'plant',
    'pyramid candle holder': 'pyramid_candle_holder',
    'pyramid candle holders': 'pyramid_candle_holder',
    'refrigerator': 'refrigerator',
    'refrigerators': 'refrigerator',
    'refridgerator': 'refrigerator',
    'fridge': 'refrigerator',
    'fridges': 'refrigerator',
    'remote': 'remote',
    'remotes': 'remote',
    'round table': 'round_table',
    'round tables': 'round_table',
    'small table': 'small_table',
    'small tables': 'small_table',
    'soccer ball': 'soccer_ball',
    'soccer balls': 'soccer_ball',
    'sofa': 'sofa',
    'sofas': 'sofa',
    'speaker': 'speaker',
    'speakers': 'speaker',
    'sphere decoration': 'sphere_decoration',
    'sphere decorations': 'sphere_decoration',
    'sink': 'sink',
    'sinks': 'sink',
    'stone decoration': 'stone_decoration',
    'stone decorations': 'stone_decoration',
    'stool': 'stool',
    'stools': 'stool',
    'suitcase': 'suitcase',
    'suitcases': 'suitcase',
    'sushi': 'sushi',
    'table': 'table',
    'tables': 'table',
    'tea table': 'tea_table',
    'tea tables': 'tea_table',
    'television': 'tv',
    'televisions': 'tv',
    'tray': 'tray',
    'trays': 'tray',
    'trash can': 'trash_can',
    'trash cans': 'trash_can',
    'tv': 'tv',
    'tv cabinet': 'tv_cabinet',
    'tv cabinets': 'tv_cabinet',
    'tv remote': 'tv_remote',
    'tv remotes': 'tv_remote',
    'vase': 'vase',
    'vases': 'vase',
    'wall lamp': 'wall_lamp',
    'wall lamps': 'wall_lamp',
    'water cooler': 'water_cooler',
    'water coolers': 'water_cooler',
}

_STRUCTURAL_ENTITIES = {
    'bookcase': 'bookcase',
    'bookcases': 'bookcase',
    'column': 'column',
    'columns': 'column',
    'curtain': 'curtain',
    'curtains': 'curtain',
    'display ledge': 'display_ledge',
    'display ledges': 'display_ledge',
    'door': 'door',
    'door frame': 'door_frame',
    'door frames': 'door_frame',
    'doorway': 'doorway',
    'doorways': 'doorway',
    'doors': 'door',
    'exit sign': 'exit_sign',
    'exit signs': 'exit_sign',
    'fireplace': 'fireplace',
    'fireplaces': 'fireplace',
    'floor': 'floor',
    'floors': 'floor',
    'folding screen': 'folding_screen',
    'folding screens': 'folding_screen',
    'kitchen counter': 'kitchen_counter',
    'kitchen counters': 'kitchen_counter',
    'map wall decal': 'map_wall_decal',
    'map wall decals': 'map_wall_decal',
    'projector screen': 'projector_screen',
    'projector screens': 'projector_screen',
    'room': 'room',
    'rooms': 'room',
    'shelf': 'shelf',
    'shelves': 'shelf',
    'stairs': 'stairs',
    'wall': 'wall',
    'walls': 'wall',
    'corner': 'corner',
    'corners': 'corner',
    'whiteboard': 'whiteboard',
    'whiteboards': 'whiteboard',
    'window': 'window',
    'windows': 'window',
    'wardrobe door': 'wardrobe_door',
    'wardrobe doors': 'wardrobe_door',
}

_COLOURS = {
    term.replace('_', ' '): normalize_colour_name(term)
    for term in (*COLOUR_CLASSES, *COLOUR_ALIASES)
}

# These conservative lexical attributes complement colour without attempting a
# general-purpose adjective parser. Values are normalized for later grounding.
_ATTRIBUTES = {
    'big': ('size', 'large'),
    'large': ('size', 'large'),
    'little': ('size', 'small'),
    'small': ('size', 'small'),
    'tiny': ('size', 'small'),
    'short': ('size', 'short'),
    'tall': ('size', 'tall'),
    'circular': ('shape', 'round'),
    'oval': ('shape', 'oval'),
    'rectangular': ('shape', 'rectangular'),
    'round': ('shape', 'round'),
    'square': ('shape', 'square'),
    'ceramic': ('material', 'ceramic'),
    'fabric': ('material', 'fabric'),
    'glass': ('material', 'glass'),
    'leather': ('material', 'leather'),
    'metal': ('material', 'metal'),
    'metallic': ('material', 'metal'),
    'plastic': ('material', 'plastic'),
    'stone': ('material', 'stone'),
    'wood': ('material', 'wood'),
    'wooden': ('material', 'wood'),
}

_SPATIAL_RELATIONS = {
    'above': 'above',
    'behind': 'behind',
    'below': 'below',
    'beside': 'beside',
    'between': 'between',
    'closest to': 'closest_to',
    'far from': 'far_from',
    'farthest from': 'farthest_from',
    'furthest from': 'farthest_from',
    'in front of': 'in_front_of',
    'inside': 'inside',
    'near': 'near',
    'next to': 'beside',
    'on': 'on',
    'under': 'below',
}

_ORDERING_TERMS = {
    'after that': 'then',
    'and finally': 'finally',
    'and then': 'then',
    'after': 'after',
    'before': 'before',
    'finally': 'finally',
    'first': 'first',
    'next': 'next',
    'then': 'then',
}

_NUMBER_WORDS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
}

_AVOIDANCE_PATTERN = re.compile(
    r'\b(?:avoid(?:ing)?|do\s+not\s+(?:go\s+near|pass\s+between|'
    r'take\s+the\s+path)|without\s+passing\s+between|'
    r'stay\s+away\s+from)\b[^,.!?;]*',
    re.IGNORECASE,
)
_EXPLICIT_DESTINATION_PATTERN = re.compile(
    r'\b(?P<action>stop\s+at|stop\s+by|stop\s+near|go\s+to|go\s+near|'
    r'finish\s+at|finish\s+near|finish\s+beside)\s+',
    re.IGNORECASE,
)
_ELLIPTICAL_DESTINATION_PATTERN = re.compile(
    r'\b(?P<action>then|finally)\s*,?\s+to\s+',
    re.IGNORECASE,
)
_TO_PATTERN = re.compile(r'\bto\s+', re.IGNORECASE)


@dataclass(frozen=True)
class _EntityCandidate:
    """Internal entity match retaining its object/structure category."""

    mention: Mention
    category: str


@dataclass(frozen=True)
class _DestinationCue:
    """Internal route cue ending immediately before a destination phrase."""

    target_start: int
    action: str


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped_words = (re.escape(word) for word in phrase.split())
    return re.compile(r'\b' + r'\s+'.join(escaped_words) + r'\b', re.IGNORECASE)


def _candidate_mentions(
    text: str,
    vocabulary: dict[str, str],
) -> list[Mention]:
    candidates = []
    for phrase, normalized in vocabulary.items():
        for match in _phrase_pattern(phrase).finditer(text):
            candidates.append(
                Mention(match.group(), normalized, match.start(), match.end())
            )
    return candidates


def _select_non_overlapping(mentions: Iterable[Mention]) -> tuple[Mention, ...]:
    candidates = sorted(
        mentions,
        key=lambda item: (-(item.end - item.start), item.start, item.normalized),
    )
    selected = []
    for candidate in candidates:
        overlaps = any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        )
        if not overlaps:
            selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.start, item.end)))


def _extract_entities(
    text: str,
) -> tuple[tuple[Mention, ...], tuple[Mention, ...], tuple[Mention, ...]]:
    candidates = [
        _EntityCandidate(mention, 'object')
        for mention in _candidate_mentions(text, _OBJECT_NOUNS)
    ]
    candidates.extend(
        _EntityCandidate(mention, 'structure')
        for mention in _candidate_mentions(text, _STRUCTURAL_ENTITIES)
    )

    selected_mentions = _select_non_overlapping(
        candidate.mention for candidate in candidates
    )
    selected_keys = {
        (mention.start, mention.end, mention.normalized) for mention in selected_mentions
    }
    selected_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate.mention.start,
            candidate.mention.end,
            candidate.mention.normalized,
        )
        in selected_keys
    ]
    selected_candidates.sort(
        key=lambda candidate: (candidate.mention.start, candidate.mention.end)
    )

    objects = tuple(
        candidate.mention
        for candidate in selected_candidates
        if candidate.category == 'object'
    )
    structures = tuple(
        candidate.mention
        for candidate in selected_candidates
        if candidate.category == 'structure'
    )
    all_entities = tuple(candidate.mention for candidate in selected_candidates)
    return objects, structures, all_entities


def _extract_cardinalities(text: str) -> tuple[CardinalityMention, ...]:
    mentions = []
    word_pattern = re.compile(
        r'\b(?:' + '|'.join(_NUMBER_WORDS) + r')\b',
        re.IGNORECASE,
    )
    for match in word_pattern.finditer(text):
        mentions.append(
            CardinalityMention(
                match.group(),
                _NUMBER_WORDS[match.group().lower()],
                match.start(),
                match.end(),
            )
        )
    for match in re.finditer(r'\b[1-9][0-9]*\b', text):
        mentions.append(
            CardinalityMention(
                match.group(),
                int(match.group()),
                match.start(),
                match.end(),
            )
        )
    return tuple(sorted(mentions, key=lambda item: (item.start, item.end)))


def _extract_attributes(text: str) -> tuple[AttributeMention, ...]:
    mentions = []
    for source, (attribute_name, normalized) in _ATTRIBUTES.items():
        pattern = re.compile(r'(?<!\w)' + re.escape(source) + r'(?!\w)', re.I)
        for match in pattern.finditer(text):
            mentions.append(
                AttributeMention(
                    source_text=match.group(),
                    attribute_name=attribute_name,
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tuple(sorted(mentions, key=lambda item: (item.start, item.end)))


def _extract_avoidance(text: str) -> tuple[AvoidancePhrase, ...]:
    phrases = []
    for match in _AVOIDANCE_PATTERN.finditer(text):
        source_text = re.split(
            r'\s+and\s+(?=(?:go|stop|finish|take|pass)\b)',
            match.group(),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        lowered = source_text.lower()
        if re.search(r'\bbetween\b', lowered):
            constraint_type = 'avoid_between'
        elif re.search(r'\b(?:near|away\s+from)\b', lowered):
            constraint_type = 'avoid_near'
        elif not re.search(r'\bpath\b', lowered):
            constraint_type = 'avoid_near'
        else:
            constraint_type = 'avoid_path'
        phrases.append(
            AvoidancePhrase(
                source_text,
                constraint_type,
                match.start(),
                match.start() + len(source_text),
            )
        )
    return tuple(phrases)


def _normalize_action(source_action: str) -> str:
    return source_action.lower().replace(' ', '_')


def _destination_cues(text: str) -> tuple[_DestinationCue, ...]:
    cues = []
    for pattern in (
        _EXPLICIT_DESTINATION_PATTERN,
        _ELLIPTICAL_DESTINATION_PATTERN,
    ):
        for match in pattern.finditer(text):
            cues.append(
                _DestinationCue(
                    match.end(),
                    _normalize_action(match.group('action')),
                )
            )

    for match in _TO_PATTERN.finditer(text):
        prefix = text[: match.start()]
        previous_words = re.findall(r'[A-Za-z]+', prefix.lower())[-2:]
        previous_word = previous_words[-1] if previous_words else ''
        if previous_word in {'closest', 'go'}:
            continue
        if previous_word in {'then', 'finally'}:
            continue
        cues.append(_DestinationCue(match.end(), 'path_to'))

    unique = {(cue.target_start, cue.action): cue for cue in cues}
    return tuple(sorted(unique.values(), key=lambda cue: cue.target_start))


def _extract_terminal_target(
    text: str,
    all_entities: tuple[Mention, ...],
    avoidance_phrases: tuple[AvoidancePhrase, ...],
) -> TerminalTarget | None:
    try:
        task_type = classify_task_type(text)
    except ValueError:
        return None
    if task_type != INSTRUCTION_FOLLOWING:
        return None

    def is_inside_avoidance(position: int) -> bool:
        return any(
            phrase.start <= position < phrase.end
            for phrase in avoidance_phrases
        )

    cues = [
        cue
        for cue in _destination_cues(text)
        if not is_inside_avoidance(cue.target_start)
    ]
    for cue in reversed(cues):
        target = next(
            (
                entity
                for entity in all_entities
                if cue.target_start <= entity.start
            ),
            None,
        )
        if target is not None:
            return TerminalTarget(target, cue.action)
    return None


def extract_language_features(question: str) -> LanguageExtraction:
    """
    Extract the requested lexical features without constructing a task.

    This function is intentionally not the full or degraded deterministic
    parser. It recognizes normalized, span-aware evidence that those parsers can
    later assemble into the frozen shared contracts.
    """
    if not isinstance(question, str):
        raise TypeError('question must be a string')
    if not question.strip():
        raise ValueError('question must not be empty')

    object_nouns, structural_entities, all_entities = _extract_entities(question)
    colours = _select_non_overlapping(_candidate_mentions(question, _COLOURS))
    spatial_relations = _select_non_overlapping(
        _candidate_mentions(question, _SPATIAL_RELATIONS)
    )
    ordering_terms = _select_non_overlapping(
        _candidate_mentions(question, _ORDERING_TERMS)
    )
    avoidance_phrases = _extract_avoidance(question)
    terminal_target = _extract_terminal_target(
        question,
        all_entities,
        avoidance_phrases,
    )

    return LanguageExtraction(
        object_nouns=object_nouns,
        colours=colours,
        attributes=_extract_attributes(question),
        cardinalities=_extract_cardinalities(question),
        structural_entities=structural_entities,
        spatial_relations=spatial_relations,
        ordering_terms=ordering_terms,
        avoidance_phrases=avoidance_phrases,
        terminal_target=terminal_target,
    )


__all__ = [
    'AttributeMention',
    'AvoidancePhrase',
    'CardinalityMention',
    'LanguageExtraction',
    'Mention',
    'TerminalTarget',
    'extract_language_features',
]
