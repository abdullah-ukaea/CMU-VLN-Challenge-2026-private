# Ground-Truth Development Data

Q-MapNav keeps released challenge and VLA-3D schemas behind a ROS-independent
adapter in `qmapnav.evaluation`. This data is development/evaluation input; it
must never become a dependency of hidden-scene runtime behavior.

This module only loads and validates data. Oracle solving, semantic geometry,
route planning, and proxy metrics are separate evaluation tasks.

## Sources

### Challenge questions and answers

`questions/questions.json` contains 15 scene objects. Each scene contains one
numerical, two object-reference, and two instruction-following strings. The
loader produces stable IDs in this form:

```text
office_1_numerical_01
office_1_object_reference_01
office_1_object_reference_02
office_1_instruction_following_01
office_1_instruction_following_02
```

The two instruction trajectories are linked by their released question number:

```text
questions/<scene>/trajectory_q4.ply
questions/<scene>/trajectory_q5.ply
```

`questions/<scene>/questions.pdf` contains answer visualizations. The PDF is
linked to each `QuestionRecord`, with a one-based visualization index matching
the released question number. The images are not treated as machine-readable
counts or object IDs.

Optional checked answers use a separate JSON file:

```json
{
  "answers": {
    "scene_01_numerical_01": {
      "expected_count": 3
    },
    "scene_01_object_reference_01": {
      "expected_object_id": "chair_17"
    }
  }
}
```

Exactly one of `expected_count` or `expected_object_id` is allowed per entry.
Instruction answers are represented by their visual evidence and reference
trajectory until a checked semantic annotation is added. Solvers must not
embed manual answers in code.

### Unity simulator objects

Each released simulator scene has an `object_list.txt` with rows:

```text
object_id x y z length width height yaw "raw label"
```

`load_unity_scene_objects()` first checks an extracted `<root>/<scene>` folder,
then reads `<root>/<scene>.zip` directly. Reading only the object member avoids
expanding multi-gigabyte Unity packages.

### VLA-3D Unity metadata

The official VLA-3D Unity subset supplies the attributes omitted by the
simulator archive:

- `<scene>_object_result.csv`: object/region IDs, raw and NYU labels, oriented
  box geometry, and up to three dominant colours;
- `<scene>_region_result.csv`: room/region labels and boxes;
- `<scene>_scene_graph.json`: per-region object relation adjacency maps.

The full official Unity ZIP is approximately 2.09 GB because it also contains
processed point clouds. The provided development tool uses HTTP byte ranges to
retrieve and CRC-check only the 45 small metadata members:

```bash
python3 tools/fetch_vla3d_metadata.py \
  --output-root /home/abdul/cmu-vln/data/vla3d
```

The resulting tree is:

```text
/home/abdul/cmu-vln/data/vla3d/
  Unity/
    office_1/
      office_1_object_result.csv
      office_1_region_result.csv
      office_1_scene_graph.json
```

The downloader rejects ignored byte ranges, unsafe member paths, unexpected
file counts, unsupported compression, truncated data, size mismatch, and
CRC-32 mismatch. Existing files are retained unless `--force` is given.

## Normalized Records

`ground_truth.py` defines immutable records:

- `QuestionRecord`;
- `ColourAttribute`;
- `OracleObject`;
- `OracleRegion`;
- `OracleRelation`;
- `OracleTrajectory`;
- `OracleScene`.

These are deliberately separate from the frozen runtime contracts in
`qmapnav.common`. Raw dataset fields must not leak into runtime perception or
planning APIs.

Geometry uses metres in the released Unity/map coordinate convention.
`dimensions_xyz` is `(length, width, height)` for Unity and `(x, y, z)` lengths
for VLA-3D. Yaw is normalized to `[-pi, pi]`.

All VLA-3D Unity object IDs and normalized raw labels match the corresponding
simulator object lists across the 15 released scenes. Box values are not
required to match: the simulator list and processed VLA-3D annotations use
different box-generation conventions, especially for large structures. The
join therefore validates the complete ID/class correspondence and uses VLA-3D
geometry for the oracle record. A strict geometry tolerance remains available
for controlled fixtures or same-convention sources.

## Normalization

Tokens are lowercase snake case. Important class aliases include:

```text
couch       -> sofa
garbage bin -> trash_can
garbage can -> trash_can
trash bin   -> trash_can
television  -> tv
nightstand  -> night_stand
plant pot   -> potted_plant
```

Colour normalization includes:

```text
gray -> grey
```

Relation normalization includes:

```text
closest    -> closest_to
farthest   -> farthest_from
beside     -> near
next_to    -> near
in         -> inside
under      -> below
on_top_of  -> on
```

The released VLA-3D Unity graphs also contain `hanging_on`; it is preserved as
a separate normalized relation. Binary relations become one explicit edge per
subject/object pair. Each `between` entry remains one ternary edge with exactly
two anchor object IDs. Duplicate edges are removed and results are sorted.

## Trajectory Policy

Only ASCII PLY 1.0 is accepted for released trajectories. The loader requires:

- one positive vertex count;
- scalar vertex properties;
- `x`, `y`, and `z` properties;
- exactly the declared number of rows;
- finite coordinates.

Property order is arbitrary and source point order is preserved. Reference
trajectories are answer evidence and route diagnostics, not the only valid
semantic route.

## Missing Data And Validation

Malformed or incomplete data raises `DatasetLoadError` with a source path and
context. The loader never invents:

- missing colours;
- missing relations;
- expected numerical counts;
- expected object IDs;
- missing trajectory points.

`QuestionRecord.answer_provenance` is one of:

```text
machine_readable
visualization_only
unavailable
```

Scene construction validates unique IDs, region references, relation object
references, question-scene membership, and trajectory-question membership.

## API

```python
from pathlib import Path

from qmapnav.evaluation import load_development_scenes


scenes = load_development_scenes(
    questions_path=Path('/data/questions/questions.json'),
    simulation_root=Path('/data/simulation'),
    vla_root=Path('/data/vla3d'),
)
```

Use `ground_truth_to_data()` or `ground_truth_to_json()` for deterministic,
JSON-compatible diagnostics. These serializers preserve collection order and
sort mapping keys.
