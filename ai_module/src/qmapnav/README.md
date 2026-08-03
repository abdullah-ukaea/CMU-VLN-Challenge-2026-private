# Q-MapNav

Q-MapNav is the competition AI package for the CMU Vision-Language-Navigation
Challenge 2026. It is a ROS 2 Jazzy `ament_python` package.

The package currently includes the deterministic language layer and the Day 2
runtime skeleton: question latching, bounded sequential waypoint execution,
registered-scan accumulation, and observational decision tracing.

Day 3 development infrastructure adds validated adapters for released
questions, Unity object lists and ZIPs, VLA-3D object/colour/relation metadata,
reference trajectories, and answer evidence. The oracle layer resolves counts
and object references and plans ordered semantic routes from perfect objects.
The evaluation harness measures answer, relation, route, and timing proxies in
quick or full regression modes without fabricating missing answer labels.

## Package Structure

```text
qmapnav/
  common/       Shared domain models, geometry, and configuration
  language/     Full and degraded deterministic parsing
  perception/   Query-conditioned visual observation
  mapping/      Object, structure, and occupancy maps
  reasoning/    Spatial predicates and task resolution
  navigation/   Semantic planning and waypoint execution
  mission/      Episode lifecycle and subsystem composition
  evaluation/   Decision traces, metrics, and regression support
```

The dependency direction is toward shared contracts and the `mission`
composition root:

- `common` must not depend on another Q-MapNav subsystem.
- `language`, `perception`, and `mapping` expose data through `common` contracts.
- `reasoning` consumes language and map outputs without owning ROS transport.
- `navigation` consumes resolved tasks and map state without owning perception.
- `mission` is the only layer that composes the complete runtime workflow.
- `evaluation` observes stable outputs and must not control production behavior.

These boundaries keep offline tests independent from ROS orchestration and avoid
circular subsystem dependencies.

## Shared Contracts

The frozen subsystem interfaces are exported from `qmapnav.common`:

- `TaskSpecification` and its language-supporting types;
- `ObjectInstance`;
- `ResolvedTask` and `ResolvedConstraint`;
- `EpisodeResult`.

Field semantics, units, coordinate conventions and validation rules are defined
in [`docs/contracts.md`](docs/contracts.md). Treat changes to these types as API
changes that require corresponding producer, consumer and regression updates.

## Language Parser

The ROS-independent language subsystem provides deterministic task
classification, span-aware feature extraction, full parsing into the frozen
`TaskSpecification`, and a degraded parser for partial recovery. Its supported
grammar, public API, normalized route actions, and fallback behavior are
documented in [`docs/parser.md`](docs/parser.md).

## Protocol Execution

The mission node latches the first valid challenge question, suppresses repeats,
and adapts the ROS-independent sequential waypoint executor to the official
question, odometry, and waypoint topics. Progress monitoring, bounded recovery,
and pose-hold cancellation are documented in
[`docs/execution.md`](docs/execution.md).

The bounded map-frame registered-scan foundation and its conservative
safe-offset policy are documented in [`docs/mapping.md`](docs/mapping.md).
Versioned, bounded asynchronous JSONL traces are documented in
[`docs/tracing.md`](docs/tracing.md).

The development-only normalized ground-truth records, source validation,
selective VLA-3D metadata retrieval, and answer provenance rules are documented
in [`docs/ground_truth.md`](docs/ground_truth.md).

Perfect-object candidate reasoning, semantic floor geometry, and the oracle
grid route planner are documented in
[`docs/oracle_reasoning.md`](docs/oracle_reasoning.md).

Proxy metric definitions, answer-provenance policy, report files, and quick/full
regression commands are documented in
[`docs/evaluation.md`](docs/evaluation.md).

The Day 4 panorama/camera-ray convention, overlapping perspective layout, and
seam-aware perception worker are documented in
[`docs/perception.md`](docs/perception.md).
The measured two-candidate bake-off and selected YOLOE baseline are recorded in
[`docs/day_4_detector_decision.md`](docs/day_4_detector_decision.md).

## Build

From `/home/docker/ai_module` in the AI container:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select qmapnav
source install/setup.bash
```

## Launch

```bash
ros2 launch qmapnav qmapnav.launch.py
```

The launch file currently starts only the composition node. Runtime components
will be added behind the frozen module boundaries in their scheduled tasks.

## Test

```bash
colcon test --packages-select qmapnav
colcon test-result --verbose
```

## Oracle Regression

With the released scene and VLA-3D metadata available in the workspace:

```bash
ros2 run qmapnav qmapnav_benchmark --mode quick
ros2 run qmapnav qmapnav_benchmark --mode full
```
