# Oracle Reasoning And Semantic Planning

The evaluation oracle layer proves Q-MapNav task reasoning with released perfect
objects while keeping perception out of the loop. It consumes the protocol
`TaskSpecification` and the normalized `OracleScene`; it does not run in the
hidden-scene perception path and does not contain expected answers.

Proxy metrics, benchmark reports, and quick/full regression commands are a
separate evaluation task and are intentionally not implemented here.

## Numerical And Object Resolution

`qmapnav.reasoning` exposes:

```python
result = solve_numerical(task, scene)
selection = solve_object_reference(task, scene)
```

Both solvers construct candidate sets by normalized class and colour, then
propagate every hard relation in both directions. Bidirectional propagation is
important for an inbound constraint such as:

```text
chairs with pillows on them
```

Here the parsed relation is `pillow on chair`, but the counted target is the
chair. Closest/farthest clauses geometrically rank their subject candidate and
then hard constraints are propagated again. `between` remains ternary; a
single cardinality-two entity is assigned two distinct object IDs.

The released scene graph is used when an explicit matching edge exists.
Deterministic OBB geometry supplements sparse relation annotations for `on`,
`above`, `below`, `near`, `inside`, and `between`. Numerical and object answer
solvers apply these constraints strictly.

Results include one `CandidateDecision` for every class-matching target. A
decision records `accepted` or specific rejection reasons such as a colour,
support relation, or ranking failure. Ambiguous object references return the
lowest stable object ID, a zero confidence margin, and an explicit warning;
they are never silently presented as unique.

Released query/data vocabulary differences are explicit. Examples include
`flower -> flowers`, `stone_decoration -> zen_stone_decoration`, generic
`cabinet` matching cabinet subtypes, and using walls only when a scene has no
window object. Released VLA colour schemes are supplemented conservatively:
`maroon` satisfies red, very dark RGB clusters satisfy black, and blue-biased
clusters recover blue objects whose scheme label is grey.

## Semantic Geometry

The pure-Python geometry layer has no ROS, Shapely, Open3D, or point-cloud
dependency. It provides:

- `Polygon2D`: validated simple floor-plane polygons;
- `object_footprint()`: the oriented XY projection of a released 3D OBB;
- `make_approach_region()`: collision-clear goal annulus around an object;
- `make_near_region()`: configurable acceptable near annulus;
- `make_between_gate()`: a thin gate spanning two oriented footprints;
- `GateResult`: valid/invalid state, measured gap, and failure reason;
- `SemanticRegion`: required or forbidden polygon plus source object IDs.

The geometry API rejects non-finite values, invalid distances, coincident gate
anchors, and gaps below the requested robot clearance. The route planner may
explicitly infer a gate from a released `pass_between` instruction when
processed OBBs overlap conservatively; that fallback is recorded as a warning.

## Oracle Route Planner

```python
plan = plan_semantic_route(
    task,
    scene,
    start_xy=(0.0, 0.0),
)
```

The planner:

1. resolves all route and avoidance entities;
2. creates one or more semantic-region choices for each ordered step;
3. converts `avoid_near` and `avoid_between` clauses to blocked regions;
4. rasterizes floor-level object OBBs into a bounded occupancy grid;
5. selects a reachable candidate region using deterministic eight-connected
   Dijkstra search;
6. concatenates the segments in parser order.

Small objects on supports use both their own region and an approach region
around the smallest containing support. This prevents a goal such as a cup on
a table from asking the base to enter the tabletop footprint.

Floor, ceiling, rugs, high wall decorations, windows, doors, unknown aggregate
meshes, and tree-crown OBBs are not rasterized as floor obstacles. These
released boxes either do not occupy base height or are conservative aggregate
geometry. Ordinary floor-level furniture remains inflated by the configurable
oracle radius.

When a released instruction asserts a between corridor but conservative OBBs
cover the corridor, only the semantic gate cells are cleared. Forbidden
regions are immediately re-applied, so a required-gate fallback cannot erase
an avoidance constraint. Every such override appears in `OracleRoutePlan`
warnings.

Instruction planning uses `strict_relations=False`: if released relation data
cannot ground an intermediate clause, class and attribute candidates survive
for a documented partial-route attempt. Missing route entities, invalid
geometry, oversized grids, and unreachable regions still raise a specific
`RoutePlanningError` rather than returning an empty path.

## Configuration And Output

`OraclePlannerConfig` keeps all distances and raster limits configurable. The
evaluation defaults are development proxies and are not frozen executor parameters:

```text
resolution:                    0.25 m
oracle obstacle inflation:     0.15 m
approach clearance/range:      0.45 m / 2.00 m
near range:                    0.60 m to 1.80 m
forbidden-near distance:       1.50 m
between clearance/depth:       0.05 m / 1.00 m
maximum raster cells:          1,000,000
```

`OracleRoutePlan` returns ordered XY waypoints, the selected semantic region
for every route step, forbidden regions, resolved object IDs, the final grid,
and warnings. It intentionally does not compute scores or compare against
reference trajectories; those belong to the later proxy-evaluation task.
