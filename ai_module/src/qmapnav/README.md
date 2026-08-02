# Q-MapNav

Q-MapNav is the competition AI package for the CMU Vision-Language-Navigation
Challenge 2026. It is a ROS 2 Jazzy `ament_python` package.

The package currently includes the deterministic language layer and the Day 2
runtime skeleton: question latching, bounded sequential waypoint execution,
registered-scan accumulation, and observational decision tracing.

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
