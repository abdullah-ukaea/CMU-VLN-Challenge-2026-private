"""Adapters for released challenge and VLA-3D development data."""

import csv
import io
import json
from math import isfinite, pi
from pathlib import Path
import re
import shlex
from typing import Any, TextIO
from zipfile import BadZipFile, ZipFile

from qmapnav.evaluation.ground_truth import ColourAttribute
from qmapnav.evaluation.ground_truth import OracleObject
from qmapnav.evaluation.ground_truth import OracleRegion
from qmapnav.evaluation.ground_truth import OracleRelation
from qmapnav.evaluation.ground_truth import OracleScene
from qmapnav.evaluation.ground_truth import OracleTrajectory
from qmapnav.evaluation.ground_truth import QuestionRecord


QUESTION_TYPE_ORDER = (
    'numerical',
    'object_reference',
    'instruction_following',
)
EXPECTED_QUESTION_COUNTS = {
    'instruction_following': 2,
    'numerical': 1,
    'object_reference': 2,
}
CLASS_ALIASES = {
    'couch': 'sofa',
    'garbage_bin': 'trash_can',
    'garbage_can': 'trash_can',
    'nightstand': 'night_stand',
    'plant_pot': 'potted_plant',
    'refridgerator': 'refrigerator',
    'television': 'tv',
    'trash_bin': 'trash_can',
}
COLOUR_ALIASES = {
    'gray': 'grey',
}
RELATION_ALIASES = {
    'above': 'above',
    'below': 'below',
    'beside': 'near',
    'between': 'between',
    'closest': 'closest_to',
    'closest_to': 'closest_to',
    'farthest': 'farthest_from',
    'farthest_from': 'farthest_from',
    'hanging_on': 'hanging_on',
    'in': 'inside',
    'inside': 'inside',
    'near': 'near',
    'next_to': 'near',
    'on': 'on',
    'on_top_of': 'on',
    'under': 'below',
}
MISSING_TOKENS = frozenset({'', '_', '-1', 'n/a', 'none', 'null'})


class DatasetLoadError(ValueError):
    """Raised when released development data violates its expected schema."""


def normalize_token(value: str) -> str:
    """Normalize a raw class, colour, or relation token to snake case."""
    if not isinstance(value, str) or not value.strip():
        raise DatasetLoadError('cannot normalize an empty token')
    token = re.sub(r'[^a-z0-9]+', '_', value.strip().lower()).strip('_')
    if not token:
        raise DatasetLoadError(f'token has no alphanumeric content: {value!r}')
    return token


def normalize_class_name(value: str) -> str:
    """Normalize a raw object class and apply challenge aliases."""
    token = normalize_token(value)
    return CLASS_ALIASES.get(token, token)


def normalize_colour(value: str) -> str:
    """Normalize a VLA-3D colour label to Q-MapNav vocabulary."""
    token = normalize_token(value)
    return COLOUR_ALIASES.get(token, token)


def normalize_relation(value: str) -> str:
    """Normalize a VLA-3D relation name or reject an unsupported relation."""
    token = normalize_token(value)
    try:
        return RELATION_ALIASES[token]
    except KeyError as exc:
        raise DatasetLoadError(f'unsupported relation: {value!r}') from exc


def _load_json(path: Path) -> Any:
    try:
        with Path(path).open(encoding='utf-8') as stream:
            return json.load(stream)
    except OSError as exc:
        raise DatasetLoadError(f'cannot read JSON file {path}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(f'invalid JSON in {path}: {exc}') from exc


def _non_empty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetLoadError(f'{context} must be a non-empty string')
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def _float_value(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise DatasetLoadError(f'{context} must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(f'{context} must be a finite number') from exc
    if not isfinite(result):
        raise DatasetLoadError(f'{context} must be finite')
    return result


def _int_value(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise DatasetLoadError(f'{context} must be an integer')
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(f'{context} must be an integer') from exc
    if str(result) != str(value).strip():
        raise DatasetLoadError(f'{context} must be an integer')
    return result


def _normalize_yaw(value: float) -> float:
    yaw = (value + pi) % (2.0 * pi) - pi
    if yaw == -pi and value > 0.0:
        return pi
    return yaw


def _answer_mapping(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise DatasetLoadError('answer mapping must be a JSON object')
    answers = payload.get('answers', payload)
    if not isinstance(answers, dict):
        raise DatasetLoadError('answer mapping "answers" must be an object')
    normalized: dict[str, dict[str, Any]] = {}
    for question_id, answer in answers.items():
        _non_empty_text(question_id, 'answer question ID')
        if not isinstance(answer, dict):
            raise DatasetLoadError(
                f'answer for {question_id!r} must be a JSON object'
            )
        unknown = set(answer) - {'expected_count', 'expected_object_id'}
        if unknown:
            raise DatasetLoadError(
                f'answer for {question_id!r} has unknown fields: {sorted(unknown)}'
            )
        if len(answer) != 1:
            raise DatasetLoadError(
                f'answer for {question_id!r} must contain exactly one value'
            )
        normalized[question_id] = dict(answer)
    return normalized


def load_questions(
    questions_path: Path,
    *,
    answers_path: Path | None = None,
    require_released_distribution: bool = True,
) -> tuple[QuestionRecord, ...]:
    """Load released questions with stable IDs and answer/trajectory links."""
    questions_path = Path(questions_path)
    payload = _load_json(questions_path)
    if not isinstance(payload, list) or not payload:
        raise DatasetLoadError('questions.json must contain a non-empty list')

    answers = _answer_mapping(answers_path)
    records: list[QuestionRecord] = []
    scene_ids: set[str] = set()
    for scene_position, scene_payload in enumerate(payload, start=1):
        if not isinstance(scene_payload, dict):
            raise DatasetLoadError(f'scene record {scene_position} must be an object')
        if set(scene_payload) != {'questions', 'scene'}:
            raise DatasetLoadError(
                f'scene record {scene_position} must contain scene and questions'
            )
        scene_id = normalize_token(
            _non_empty_text(scene_payload['scene'], 'scene name')
        )
        if scene_id in scene_ids:
            raise DatasetLoadError(f'duplicate scene ID: {scene_id!r}')
        scene_ids.add(scene_id)

        categories = scene_payload['questions']
        if not isinstance(categories, dict):
            raise DatasetLoadError(f'{scene_id} questions must be an object')
        if set(categories) != set(QUESTION_TYPE_ORDER):
            raise DatasetLoadError(
                f'{scene_id} must contain exactly {QUESTION_TYPE_ORDER}'
            )

        scene_directory = questions_path.parent / scene_id
        answer_document = scene_directory / 'questions.pdf'
        answer_document = answer_document if answer_document.is_file() else None
        question_number = 0
        for task_type in QUESTION_TYPE_ORDER:
            task_questions = categories[task_type]
            if not isinstance(task_questions, list):
                raise DatasetLoadError(
                    f'{scene_id} {task_type} questions must be a list'
                )
            if (
                require_released_distribution
                and len(task_questions) != EXPECTED_QUESTION_COUNTS[task_type]
            ):
                expected = EXPECTED_QUESTION_COUNTS[task_type]
                raise DatasetLoadError(
                    f'{scene_id} {task_type} has {len(task_questions)} questions; '
                    f'expected {expected}'
                )
            for task_index, question in enumerate(task_questions, start=1):
                question_number += 1
                text = _non_empty_text(
                    question,
                    f'{scene_id} {task_type} question {task_index}',
                )
                question_id = f'{scene_id}_{task_type}_{task_index:02d}'
                trajectory_path = None
                if task_type == 'instruction_following':
                    candidate = scene_directory / f'trajectory_q{question_number}.ply'
                    if candidate.is_file():
                        trajectory_path = candidate

                structured_answer = answers.get(question_id, {})
                expected_count = structured_answer.get('expected_count')
                expected_object_id = structured_answer.get('expected_object_id')
                if expected_count is not None:
                    expected_count = _int_value(
                        expected_count,
                        f'{question_id} expected_count',
                    )
                if expected_object_id is not None:
                    expected_object_id = _non_empty_text(
                        expected_object_id,
                        f'{question_id} expected_object_id',
                    )

                if structured_answer:
                    provenance = 'machine_readable'
                elif answer_document is not None:
                    provenance = 'visualization_only'
                else:
                    provenance = 'unavailable'

                try:
                    record = QuestionRecord(
                        question_id=question_id,
                        scene_id=scene_id,
                        question_text=text,
                        task_type=task_type,
                        question_number=question_number,
                        task_index=task_index,
                        trajectory_path=trajectory_path,
                        answer_document_path=answer_document,
                        answer_visualization_index=(
                            question_number if answer_document is not None else None
                        ),
                        expected_count=expected_count,
                        expected_object_id=expected_object_id,
                        answer_provenance=provenance,
                    )
                except (TypeError, ValueError) as exc:
                    raise DatasetLoadError(
                        f'invalid question record {question_id}: {exc}'
                    ) from exc
                records.append(record)

    known_question_ids = {record.question_id for record in records}
    unknown_answers = set(answers) - known_question_ids
    if unknown_answers:
        raise DatasetLoadError(
            f'answer mapping contains unknown questions: {sorted(unknown_answers)}'
        )
    if len(known_question_ids) != len(records):
        raise DatasetLoadError('generated question IDs are not unique')
    if require_released_distribution:
        if len(scene_ids) != 15 or len(records) != 75:
            raise DatasetLoadError(
                f'released corpus must contain 15 scenes and 75 questions; '
                f'found {len(scene_ids)} scenes and {len(records)} questions'
            )
    return tuple(records)


def _unity_objects_from_stream(stream: TextIO, source: str) -> tuple[OracleObject, ...]:
    objects: list[OracleObject] = []
    object_ids: set[str] = set()
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            fields = shlex.split(line, posix=True)
        except ValueError as exc:
            raise DatasetLoadError(
                f'{source}:{line_number}: invalid quoted object row: {exc}'
            ) from exc
        if len(fields) != 9:
            raise DatasetLoadError(
                f'{source}:{line_number}: expected 9 fields, found {len(fields)}'
            )
        object_id = _non_empty_text(fields[0], 'Unity object ID')
        if object_id in object_ids:
            raise DatasetLoadError(f'{source}: duplicate object ID {object_id!r}')
        object_ids.add(object_id)
        raw_label = _non_empty_text(fields[8], 'Unity object label')
        centre = tuple(
            _float_value(fields[index], f'{source}:{line_number} centre')
            for index in range(1, 4)
        )
        dimensions = tuple(
            _float_value(fields[index], f'{source}:{line_number} dimensions')
            for index in range(4, 7)
        )
        yaw = _normalize_yaw(
            _float_value(fields[7], f'{source}:{line_number} yaw')
        )
        try:
            objects.append(
                OracleObject(
                    object_id=object_id,
                    class_name=normalize_class_name(raw_label),
                    centre_xyz=centre,
                    dimensions_xyz=dimensions,
                    yaw=yaw,
                    raw_class_name=raw_label,
                    sources=('unity_object_list',),
                )
            )
        except (TypeError, ValueError) as exc:
            raise DatasetLoadError(
                f'{source}:{line_number}: invalid object: {exc}'
            ) from exc
    if not objects:
        raise DatasetLoadError(f'{source}: object list is empty')
    return tuple(objects)


def load_unity_object_list(path: Path) -> tuple[OracleObject, ...]:
    """Load a released Unity ``object_list.txt`` file."""
    path = Path(path)
    try:
        with path.open(encoding='utf-8') as stream:
            return _unity_objects_from_stream(stream, str(path))
    except OSError as exc:
        raise DatasetLoadError(f'cannot read Unity object list {path}: {exc}') from exc


def load_unity_scene_objects(
    simulation_root: Path,
    scene_id: str,
) -> tuple[tuple[OracleObject, ...], Path | None]:
    """Load scene objects from an extracted directory or simulator ZIP."""
    simulation_root = Path(simulation_root)
    normalized_scene = normalize_token(scene_id)
    extracted = simulation_root / normalized_scene / 'object_list.txt'
    if extracted.is_file():
        return load_unity_object_list(extracted), None

    archive = simulation_root / f'{normalized_scene}.zip'
    if not archive.is_file():
        raise DatasetLoadError(
            f'no extracted scene or ZIP found for {normalized_scene!r} '
            f'under {simulation_root}'
        )
    expected_suffix = f'/{normalized_scene}/object_list.txt'
    try:
        with ZipFile(archive) as zip_file:
            members = [
                item
                for item in zip_file.namelist()
                if f'/{item}'.endswith(expected_suffix)
            ]
            if len(members) != 1:
                raise DatasetLoadError(
                    f'{archive} must contain exactly one {expected_suffix}'
                )
            with zip_file.open(members[0]) as binary_stream:
                with io.TextIOWrapper(binary_stream, encoding='utf-8') as stream:
                    return (
                        _unity_objects_from_stream(
                            stream,
                            f'{archive}!{members[0]}',
                        ),
                        archive,
                    )
    except BadZipFile as exc:
        raise DatasetLoadError(f'invalid simulator ZIP {archive}: {exc}') from exc
    except OSError as exc:
        raise DatasetLoadError(f'cannot read simulator ZIP {archive}: {exc}') from exc


def _open_csv(path: Path) -> tuple[csv.DictReader, TextIO]:
    try:
        stream = Path(path).open(encoding='utf-8-sig', newline='')
    except OSError as exc:
        raise DatasetLoadError(f'cannot read CSV file {path}: {exc}') from exc
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        stream.close()
        raise DatasetLoadError(f'CSV file has no header: {path}')
    return reader, stream


def _require_csv_fields(
    path: Path,
    fieldnames: list[str],
    required: set[str],
) -> None:
    missing = required - set(fieldnames)
    if missing:
        raise DatasetLoadError(f'{path} is missing CSV fields: {sorted(missing)}')


def _load_colours(
    row: dict[str | None, Any],
    *,
    path: Path,
    line_number: int,
) -> tuple[ColourAttribute, ...]:
    colours: list[ColourAttribute] = []
    seen_labels: set[str] = set()
    for index in range(1, 4):
        raw_label = _optional_text(row.get(f'object_color_scheme{index}'))
        if raw_label is None:
            continue
        label = normalize_colour(raw_label)
        if label in seen_labels:
            raise DatasetLoadError(
                f'{path}:{line_number}: duplicate dominant colour {label!r}'
            )
        seen_labels.add(label)
        rgb_values: list[int] = []
        for channel in 'rgb':
            field = f'object_color_{channel}{index}'
            value = _float_value(row.get(field), f'{path}:{line_number} {field}')
            rounded = round(value)
            if value != rounded or not 0 <= rounded <= 255:
                raise DatasetLoadError(
                    f'{path}:{line_number}: {field} must be an integer in [0, 255]'
                )
            rgb_values.append(rounded)
        proportion_field = f'object_color_scheme_percentage{index}'
        proportion = _float_value(
            row.get(proportion_field),
            f'{path}:{line_number} {proportion_field}',
        )
        distance_field = f'object_color_scheme_average_dist{index}'
        raw_distance = _optional_text(row.get(distance_field))
        distance = (
            None
            if raw_distance is None
            else _float_value(raw_distance, f'{path}:{line_number} {distance_field}')
        )
        try:
            colours.append(
                ColourAttribute(
                    label=label,
                    rgb=tuple(rgb_values),
                    proportion=proportion,
                    average_lab_distance=distance,
                )
            )
        except (TypeError, ValueError) as exc:
            raise DatasetLoadError(
                f'{path}:{line_number}: invalid colour attribute: {exc}'
            ) from exc
    return tuple(colours)


def load_vla3d_objects(path: Path) -> tuple[OracleObject, ...]:
    """Load VLA-3D object geometry, labels, regions, and colours."""
    path = Path(path)
    required = {
        'nyu40_label',
        'nyu_label',
        'object_bbox_cx',
        'object_bbox_cy',
        'object_bbox_cz',
        'object_bbox_heading',
        'object_bbox_xlength',
        'object_bbox_ylength',
        'object_bbox_zlength',
        'object_id',
        'raw_label',
        'region_id',
    }
    reader, stream = _open_csv(path)
    try:
        _require_csv_fields(path, list(reader.fieldnames or ()), required)
        objects: list[OracleObject] = []
        object_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            extra = row.get(None)
            if extra and any(_optional_text(item) is not None for item in extra):
                raise DatasetLoadError(f'{path}:{line_number}: unexpected CSV values')
            object_id = _non_empty_text(row.get('object_id'), 'VLA-3D object ID')
            if object_id in object_ids:
                raise DatasetLoadError(f'{path}: duplicate object ID {object_id!r}')
            object_ids.add(object_id)
            raw_label = _non_empty_text(
                row.get('raw_label'),
                f'{path}:{line_number} raw_label',
            )
            centre = tuple(
                _float_value(
                    row.get(f'object_bbox_c{axis}'),
                    f'{path}:{line_number} object_bbox_c{axis}',
                )
                for axis in 'xyz'
            )
            dimensions = tuple(
                _float_value(
                    row.get(f'object_bbox_{axis}length'),
                    f'{path}:{line_number} object_bbox_{axis}length',
                )
                for axis in 'xyz'
            )
            yaw = _normalize_yaw(
                _float_value(
                    row.get('object_bbox_heading'),
                    f'{path}:{line_number} object_bbox_heading',
                )
            )
            colours = _load_colours(row, path=path, line_number=line_number)
            region_id = _optional_text(row.get('region_id'))
            try:
                objects.append(
                    OracleObject(
                        object_id=object_id,
                        class_name=normalize_class_name(raw_label),
                        centre_xyz=centre,
                        dimensions_xyz=dimensions,
                        yaw=yaw,
                        colours=colours,
                        region_id=region_id,
                        raw_class_name=raw_label,
                        nyu_label=_optional_text(row.get('nyu_label')),
                        nyu40_label=_optional_text(row.get('nyu40_label')),
                        sources=('vla3d_object_result',),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise DatasetLoadError(
                    f'{path}:{line_number}: invalid VLA-3D object: {exc}'
                ) from exc
        if not objects:
            raise DatasetLoadError(f'{path}: object CSV is empty')
        return tuple(objects)
    finally:
        stream.close()


def load_vla3d_regions(path: Path) -> tuple[OracleRegion, ...]:
    """Load normalized VLA-3D region records."""
    path = Path(path)
    required = {
        'region_bbox_cx',
        'region_bbox_cy',
        'region_bbox_cz',
        'region_bbox_heading',
        'region_bbox_xlength',
        'region_bbox_ylength',
        'region_bbox_zlength',
        'region_id',
        'region_label',
    }
    reader, stream = _open_csv(path)
    try:
        _require_csv_fields(path, list(reader.fieldnames or ()), required)
        regions: list[OracleRegion] = []
        region_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            region_id = _non_empty_text(
                row.get('region_id'),
                f'{path}:{line_number} region_id',
            )
            if region_id in region_ids:
                raise DatasetLoadError(f'{path}: duplicate region ID {region_id!r}')
            region_ids.add(region_id)
            centre = tuple(
                _float_value(
                    row.get(f'region_bbox_c{axis}'),
                    f'{path}:{line_number} region_bbox_c{axis}',
                )
                for axis in 'xyz'
            )
            dimensions = tuple(
                _float_value(
                    row.get(f'region_bbox_{axis}length'),
                    f'{path}:{line_number} region_bbox_{axis}length',
                )
                for axis in 'xyz'
            )
            try:
                regions.append(
                    OracleRegion(
                        region_id=region_id,
                        label=normalize_token(
                            _non_empty_text(
                                row.get('region_label'),
                                f'{path}:{line_number} region_label',
                            )
                        ),
                        centre_xyz=centre,
                        dimensions_xyz=dimensions,
                        yaw=_normalize_yaw(
                            _float_value(
                                row.get('region_bbox_heading'),
                                f'{path}:{line_number} region_bbox_heading',
                            )
                        ),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise DatasetLoadError(
                    f'{path}:{line_number}: invalid VLA-3D region: {exc}'
                ) from exc
        if not regions:
            raise DatasetLoadError(f'{path}: region CSV is empty')
        return tuple(regions)
    finally:
        stream.close()


def load_vla3d_relations(
    path: Path,
    *,
    known_object_ids: set[str] | None = None,
    expected_scene_id: str | None = None,
) -> tuple[OracleRelation, ...]:
    """Load and flatten VLA-3D scene-graph relation adjacency maps."""
    path = Path(path)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise DatasetLoadError(f'{path}: scene graph must be a JSON object')
    scene_name = normalize_token(
        _non_empty_text(payload.get('scene_name'), f'{path} scene_name')
    )
    if expected_scene_id is not None and scene_name != normalize_token(
        expected_scene_id
    ):
        raise DatasetLoadError(
            f'{path}: scene_name {scene_name!r} does not match '
            f'{normalize_token(expected_scene_id)!r}'
        )
    regions = payload.get('regions')
    if not isinstance(regions, dict):
        raise DatasetLoadError(f'{path}: regions must be a JSON object')

    relations: set[OracleRelation] = set()
    for raw_region_id, region_payload in regions.items():
        region_id = _non_empty_text(raw_region_id, 'scene graph region ID')
        if not isinstance(region_payload, dict):
            raise DatasetLoadError(f'{path}: region {region_id!r} must be an object')
        adjacency_by_relation = region_payload.get('relationships')
        if not isinstance(adjacency_by_relation, dict):
            raise DatasetLoadError(
                f'{path}: region {region_id!r} has no relationship object'
            )
        for raw_relation, adjacency in adjacency_by_relation.items():
            relation = normalize_relation(raw_relation)
            if not isinstance(adjacency, dict):
                raise DatasetLoadError(
                    f'{path}: {raw_relation!r} adjacency must be an object'
                )
            for raw_subject_id, raw_targets in adjacency.items():
                subject_id = _non_empty_text(
                    raw_subject_id,
                    f'{path} relation subject',
                )
                if not isinstance(raw_targets, list):
                    raise DatasetLoadError(
                        f'{path}: targets for {subject_id!r} must be a list'
                    )
                if relation == 'between':
                    target_groups = raw_targets
                else:
                    target_groups = [[target] for target in raw_targets]
                for target_group in target_groups:
                    if not isinstance(target_group, list):
                        raise DatasetLoadError(
                            f'{path}: relation targets must be lists'
                        )
                    object_ids = tuple(
                        _non_empty_text(target, f'{path} relation object ID')
                        for target in target_group
                    )
                    try:
                        item = OracleRelation(
                            relation=relation,
                            subject_id=subject_id,
                            object_ids=object_ids,
                            region_id=region_id,
                        )
                    except (TypeError, ValueError) as exc:
                        raise DatasetLoadError(
                            f'{path}: invalid {relation} relation: {exc}'
                        ) from exc
                    if known_object_ids is not None:
                        referenced = {item.subject_id, *item.object_ids}
                        unknown = referenced - known_object_ids
                        if unknown:
                            raise DatasetLoadError(
                                f'{path}: relation references unknown object IDs: '
                                f'{sorted(unknown)}'
                            )
                    relations.add(item)
    return tuple(
        sorted(
            relations,
            key=lambda item: (
                item.region_id or '',
                item.relation,
                item.subject_id,
                item.object_ids,
            ),
        )
    )


def load_ascii_trajectory_ply(path: Path) -> OracleTrajectory:
    """Load ordered XYZ vertices from a released ASCII trajectory PLY."""
    path = Path(path)
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        raise DatasetLoadError(f'cannot read trajectory {path}: {exc}') from exc
    if not lines or lines[0].strip() != 'ply':
        raise DatasetLoadError(f'{path}: missing PLY signature')

    header_end = None
    format_seen = False
    vertex_count = None
    vertex_properties: list[str] = []
    current_element: str | None = None
    for index, raw_line in enumerate(lines[1:], start=1):
        line = raw_line.strip()
        if line == 'end_header':
            header_end = index
            break
        if not line or line.startswith('comment '):
            continue
        fields = line.split()
        if fields[:2] == ['format', 'ascii']:
            if len(fields) != 3 or fields[2] != '1.0':
                raise DatasetLoadError(f'{path}: only ASCII PLY 1.0 is supported')
            format_seen = True
        elif fields[0] == 'format':
            raise DatasetLoadError(f'{path}: only ASCII PLY is supported')
        elif fields[0] == 'element':
            if len(fields) != 3:
                raise DatasetLoadError(f'{path}: malformed element declaration')
            current_element = fields[1]
            if current_element == 'vertex':
                vertex_count = _int_value(fields[2], f'{path} vertex count')
                if vertex_count <= 0:
                    raise DatasetLoadError(f'{path}: vertex count must be positive')
        elif fields[0] == 'property' and current_element == 'vertex':
            if len(fields) != 3 or fields[1] == 'list':
                raise DatasetLoadError(
                    f'{path}: vertex properties must be scalar values'
                )
            vertex_properties.append(fields[2])
    if header_end is None:
        raise DatasetLoadError(f'{path}: missing end_header')
    if not format_seen:
        raise DatasetLoadError(f'{path}: missing ASCII PLY format declaration')
    if vertex_count is None:
        raise DatasetLoadError(f'{path}: missing vertex element')
    if len(vertex_properties) != len(set(vertex_properties)):
        raise DatasetLoadError(f'{path}: duplicate vertex property')
    missing_properties = {'x', 'y', 'z'} - set(vertex_properties)
    if missing_properties:
        raise DatasetLoadError(
            f'{path}: missing vertex properties {sorted(missing_properties)}'
        )

    data_lines = lines[header_end + 1:]
    if len(data_lines) != vertex_count:
        raise DatasetLoadError(
            f'{path}: declared {vertex_count} vertices but found {len(data_lines)}'
        )
    coordinate_indices = [vertex_properties.index(axis) for axis in 'xyz']
    points: list[tuple[float, float, float]] = []
    for offset, line in enumerate(data_lines, start=1):
        fields = line.split()
        if len(fields) != len(vertex_properties):
            raise DatasetLoadError(
                f'{path}: vertex {offset} has {len(fields)} values; '
                f'expected {len(vertex_properties)}'
            )
        point = tuple(
            _float_value(fields[index], f'{path} vertex {offset}')
            for index in coordinate_indices
        )
        points.append(point)
    try:
        return OracleTrajectory(points_xyz=tuple(points), source_path=path)
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(f'{path}: invalid trajectory: {exc}') from exc


def merge_unity_and_vla_objects(
    unity_objects: tuple[OracleObject, ...],
    vla_objects: tuple[OracleObject, ...],
    *,
    geometry_tolerance: float | None = None,
) -> tuple[OracleObject, ...]:
    """Validate and combine matching Unity and VLA-3D object annotations."""
    if geometry_tolerance is not None:
        if geometry_tolerance < 0.0 or not isfinite(geometry_tolerance):
            raise ValueError('geometry_tolerance must be finite and non-negative')
    unity_by_id = {item.object_id: item for item in unity_objects}
    vla_by_id = {item.object_id: item for item in vla_objects}
    if set(unity_by_id) != set(vla_by_id):
        missing_vla = sorted(set(unity_by_id) - set(vla_by_id))
        missing_unity = sorted(set(vla_by_id) - set(unity_by_id))
        raise DatasetLoadError(
            'Unity/VLA object ID mismatch: '
            f'missing from VLA={missing_vla}, missing from Unity={missing_unity}'
        )

    merged: list[OracleObject] = []
    for object_id in sorted(unity_by_id, key=_object_id_sort_key):
        unity = unity_by_id[object_id]
        vla = vla_by_id[object_id]
        if unity.class_name != vla.class_name:
            raise DatasetLoadError(
                f'object {object_id}: class mismatch '
                f'{unity.class_name!r} != {vla.class_name!r}'
            )
        if geometry_tolerance is not None:
            for name, left, right in (
                ('centre', unity.centre_xyz, vla.centre_xyz),
                ('dimensions', unity.dimensions_xyz, vla.dimensions_xyz),
            ):
                maximum_error = max(abs(a - b) for a, b in zip(left, right))
                if maximum_error > geometry_tolerance:
                    raise DatasetLoadError(
                        f'object {object_id}: {name} differs by '
                        f'{maximum_error:.6f} m'
                    )
        merged.append(
            OracleObject(
                object_id=vla.object_id,
                class_name=vla.class_name,
                centre_xyz=vla.centre_xyz,
                dimensions_xyz=vla.dimensions_xyz,
                yaw=vla.yaw,
                colours=vla.colours,
                region_id=vla.region_id,
                raw_class_name=vla.raw_class_name,
                nyu_label=vla.nyu_label,
                nyu40_label=vla.nyu40_label,
                sources=('unity_object_list', 'vla3d_object_result'),
            )
        )
    return tuple(merged)


def _object_id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _vla_scene_directory(vla_root: Path, scene_id: str) -> Path:
    candidates = (
        Path(vla_root) / scene_id,
        Path(vla_root) / 'Unity' / scene_id,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def load_oracle_scene(
    scene_id: str,
    *,
    questions: tuple[QuestionRecord, ...],
    simulation_root: Path,
    vla_root: Path | None = None,
    require_vla_annotations: bool = True,
) -> OracleScene:
    """Load one scene from Unity geometry plus optional VLA-3D metadata."""
    scene_id = normalize_token(scene_id)
    scene_questions = tuple(
        item for item in questions if item.scene_id == scene_id
    )
    if not scene_questions:
        raise DatasetLoadError(f'no questions found for scene {scene_id!r}')
    unity_objects, source_archive = load_unity_scene_objects(
        simulation_root,
        scene_id,
    )

    objects = unity_objects
    regions: tuple[OracleRegion, ...] = ()
    relations: tuple[OracleRelation, ...] = ()
    if vla_root is not None:
        vla_scene = _vla_scene_directory(Path(vla_root), scene_id)
        object_path = vla_scene / f'{scene_id}_object_result.csv'
        region_path = vla_scene / f'{scene_id}_region_result.csv'
        graph_path = vla_scene / f'{scene_id}_scene_graph.json'
        if all(path.is_file() for path in (object_path, region_path, graph_path)):
            vla_objects = load_vla3d_objects(object_path)
            objects = merge_unity_and_vla_objects(unity_objects, vla_objects)
            regions = load_vla3d_regions(region_path)
            relations = load_vla3d_relations(
                graph_path,
                known_object_ids={item.object_id for item in objects},
                expected_scene_id=scene_id,
            )
        elif require_vla_annotations:
            missing = [
                str(path)
                for path in (object_path, region_path, graph_path)
                if not path.is_file()
            ]
            raise DatasetLoadError(
                f'missing VLA-3D files for {scene_id}: {missing}'
            )
    elif require_vla_annotations:
        raise DatasetLoadError('vla_root is required for complete oracle scenes')

    trajectories: list[tuple[str, OracleTrajectory]] = []
    for question in scene_questions:
        if question.trajectory_path is not None:
            trajectories.append(
                (
                    question.question_id,
                    load_ascii_trajectory_ply(question.trajectory_path),
                )
            )

    extracted_scene = Path(simulation_root) / scene_id
    point_cloud = extracted_scene / 'map.ply'
    traversable_area = extracted_scene / 'traversable_area.ply'
    try:
        return OracleScene(
            scene_id=scene_id,
            objects=objects,
            relations=relations,
            regions=regions,
            questions=scene_questions,
            trajectories=tuple(trajectories),
            scene_point_cloud_path=(point_cloud if point_cloud.is_file() else None),
            traversable_area_path=(
                traversable_area if traversable_area.is_file() else None
            ),
            source_archive_path=source_archive,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(f'invalid OracleScene {scene_id}: {exc}') from exc


def load_development_scenes(
    *,
    questions_path: Path,
    simulation_root: Path,
    vla_root: Path | None = None,
    answers_path: Path | None = None,
    require_vla_annotations: bool = True,
) -> tuple[OracleScene, ...]:
    """Load every released development scene in deterministic question order."""
    questions = load_questions(questions_path, answers_path=answers_path)
    scene_ids = tuple(dict.fromkeys(item.scene_id for item in questions))
    return tuple(
        load_oracle_scene(
            scene_id,
            questions=questions,
            simulation_root=simulation_root,
            vla_root=vla_root,
            require_vla_annotations=require_vla_annotations,
        )
        for scene_id in scene_ids
    )
