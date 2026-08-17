# Shared Data Contracts

The types exported by `qmapnav.common` are the stable boundaries between
Q-MapNav subsystems. Change these contracts only when an observed competition
requirement justifies the change, and update every producer, consumer, test and
this document together.

## Conventions

- Task types are `numerical`, `object_reference` or `instruction_following`.
- Parse modes are `full` or `degraded`.
- Entity IDs are parser-local strings unique within one `TaskSpecification`.
- Object IDs are non-negative integers unique within one episode.
- Entity, relation, action and constraint tokens use normalized snake case.
- Confidence and score-map values are finite values in `[0, 1]`.
- Lists preserve semantic order where order is meaningful.
- Mutable inputs are copied during construction so caller-side mutations do not
  silently alter a contract object.

## Language Contracts

`EntityReference` stores a normalized class, optional attributes and optional
explicit cardinality. Relations and route structures refer to entities by ID so
that chained phrases do not duplicate nested entity objects.

`RelationConstraint` has one subject and one or more anchors. This supports both
binary relations such as `on` and multi-anchor relations such as `between`.

`RouteStep` represents one ordered action such as `go_to`, `go_near`,
`pass_between` or `stop_at`. Its `step_index` is zero-based.

`RouteConstraint` represents forbidden semantic regions, for example
`avoid_near` or `avoid_between`.

`TaskSpecification` contains the complete full or degraded parse. Every entity
ID referenced by a relation, route step, forbidden constraint or terminal
target must occur in `entities`.

## Runtime Object Contract

All `ObjectInstance` geometry is represented in the ROS `map` frame:

- XYZ coordinates and dimensions use metres.
- AABB fields contain component-wise minimum and maximum XYZ coordinates.
- OBB dimensions are `[length, width, height]` and must be positive.
- OBB yaw is a right-handed rotation about positive Z in radians, normalized to
  `[-pi, pi]`.
- Empty colour scores mean colour evidence is unavailable.
- Score maps need not sum to one because evidence may be multi-label.

Arrays are normalized to independent `float64` NumPy arrays with shape `(3,)`.

## Reasoning Contract

`ResolvedConstraint` grounds one semantic constraint to one or more object IDs.
`ResolvedTask` carries grounded ordered and forbidden constraints plus explicit
strings for anything that could not be resolved. A numerical answer of zero is
represented as `count=0`; `None` means no count has been resolved.

## Evaluation Contract

`EpisodeResult.execution_time` is elapsed seconds from episode start.
`score_proxy` is a finite non-negative local evaluation score and is not treated
as an official competition score. Constraint strings should match the decision
trace vocabulary. `failure_category` is `None` when no category applies.
