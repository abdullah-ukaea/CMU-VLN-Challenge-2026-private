# Q-MapNav

Q-MapNav is a ROS 2 Jazzy `ament_python` package for query-conditioned
vision-language navigation. It turns a latched challenge question and live
sensor streams into bounded semantic maps, task-specific decisions, and the
official ROS outputs. Runtime algorithms are ROS-independent; `mission` is the
composition and transport boundary.

## Data flow

```text
question ─► language parser ─► task specification
                                │
camera ─► perception ─► projection/lifting ─► object + structural maps
scan ────────────────────────────────────────┘             │
                                                          ▼
                         reasoning/counting/exploration/navigation
                                                          │
                              mission coordinators ─► ROS outputs + trace
```

The system supports numerical counting, object-reference resolution, and the
bounded instruction-following route currently defined by the frozen protocol.
Every buffer, map, retry, and episode decision has an explicit bound and emits
a best-effort result on failure.

## Package map

```text
qmapnav/
  common/       Frozen contracts, colours, and decision traces
  language/     Deterministic full and degraded question parsing
  perception/   Panorama crops, detector adapters, and visual evidence
  mapping/      Projection, lifting, object identity, structure, and grids
  reasoning/    Candidate generation, colours, relations, and resolution
  counting/     Persistent-ID numerical answers and stability
  exploration/  Viewpoint generation, scoring, and support search
  navigation/   Semantic regions, route planning, and waypoint execution
  mission/      ROS transport, configuration, perception runtime, and episodes
  evaluation/   Offline oracle and replay harnesses
```

`common` has no dependency on another Q-MapNav package. Runtime packages do
not import `evaluation`; the evaluation package is development-only. ROS
transport is confined to `mission`, and all domain components remain directly
unit-testable.

## Contracts and frames

The public data contracts, units, timestamps, and map-frame conventions are in
[`docs/contracts.md`](docs/contracts.md) and [`docs/frames.md`](docs/frames.md).
The parser grammar is documented in [`docs/parser.md`](docs/parser.md). The
official topics, question latch, and sequential waypoint protocol are in
[`docs/execution.md`](docs/execution.md).

## Build and launch

Inside the competition AI container:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/docker/ai_module
colcon build --packages-select qmapnav
source install/setup.bash
ros2 launch qmapnav qmapnav.launch.py
```

The frozen submission configuration is
[`configs/submission_v1.yaml`](configs/submission_v1.yaml). It preserves the
official topic names and all runtime parameter keys. Detector weights and the
runtime colour prototypes are resolved from the packaged data directory;
network access is not required during an episode.

## Tests

```bash
colcon test --packages-select qmapnav
colcon test-result --verbose
```

The suite protects contracts and behavior rather than development milestones.
It includes parser-corpus checks, frame mutation tests, map identity
regressions, counting stability, instruction exit scenarios, and the offline
evaluation harness.

## Offline evaluation

The evaluation harness uses released questions and local simulation/VLA-3D
metadata. It never controls ROS and never fabricates unavailable answer labels.
From the repository root, run:

```bash
python3 -m qmapnav.evaluation.benchmark_runner --mode quick
python3 -m qmapnav.evaluation.benchmark_runner --mode full
python3 -c 'from qmapnav.evaluation.numerical_benchmark import main; main()' \
  --questions questions/questions.json \
  --simulation-root ../simulation \
  --vla-root ../data/vla3d \
  --output-directory /tmp/qmapnav/numerical
python3 -m qmapnav.evaluation.object_reference_replay --mode quick \
  --questions-path questions/questions.json \
  --simulation-root ../simulation \
  --vla-root ../data/vla3d \
  --output-root /tmp/qmapnav/object_reference
```

Installed workspaces expose the same harnesses as
`qmapnav_benchmark`, `qmapnav_numerical_benchmark`, and
`qmapnav_object_benchmark`. Reports contain summaries, per-question results,
failure categories, parser audits, and route or marker evidence. See
[`docs/evaluation.md`](docs/evaluation.md) and
[`docs/ground_truth.md`](docs/ground_truth.md) for input and output contracts.

## Runtime subsystem references

- [`docs/perception.md`](docs/perception.md): panorama geometry and detector selection.
- [`docs/frames.md`](docs/frames.md): camera, LiDAR, pose, and timestamp transforms.
- [`docs/projection.md`](docs/projection.md): source-time projection and densification.
- [`docs/mapping.md`](docs/mapping.md): lifting, identity fusion, structure, and occupancy.
- [`docs/reasoning.md`](docs/reasoning.md): colour, support relations, and resolution.
- [`docs/tracing.md`](docs/tracing.md): bounded JSONL decision traces.
