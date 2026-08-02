# Q-MapNav

Q-MapNav is the competition AI package for the CMU Vision-Language-Navigation
Challenge 2026. It is a ROS 2 Jazzy `ament_python` package.

This initial scaffold freezes the subsystem boundaries from the Q-MapNav v2.0
design. It deliberately contains no parser, perception, mapping, reasoning, or
navigation behavior yet.

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
