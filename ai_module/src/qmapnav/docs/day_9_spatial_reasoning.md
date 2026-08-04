# Day 9 Spatial Reference Resolution

Day 9 adds a ROS-independent spatial reasoning layer over persistent
`ObjectMap` records and `StructuralMap` anchors. It does not consume raw
detections, publish an official marker, or build a final semantic route.

## Pipeline

```text
EntityReference
  -> conservative class/colour candidate generation
  -> exact object, set, pair, and role-product enumeration
  -> footprint-aware near/distance/between evaluation
  -> optional physical between-gate validation
  -> complete hard/soft constraint scoring
  -> top-two margin and explicit resolution status
```

The frozen shared contracts in `qmapnav/common/contracts.py` are unchanged.
Day 9 contracts live in `reasoning/resolution_contracts.py` and serialize every
satisfied, violated, and unresolved constraint without forcing a selection.

## Modules

- `candidate_generation.py` snapshots persistent maps and records every retain
  or reject reason. Colour and weak geometry remain soft evidence by default.
- `cardinality.py` enforces exact cardinality, canonical unordered pairs, and
  distinct role occupancy.
- `spatial_relations.py` implements shared centre/footprint distance, symmetric
  scale-aware `near`, exhaustive `closest`/`farthest`, and finite-segment
  object-level `between`.
- `corridor_evaluation.py` constructs a map-frame gate from nearest footprint
  boundaries and checks robot clearance, inflated occupancy, blockers, and
  reachable approach and exit cells. It never plans the final route.
- `hypothesis_scoring.py` enumerates complete role products and prevents hard
  violations from being hidden by soft scores.
- `ambiguity.py` calculates raw and normalized top-two margins and returns
  `resolved`, `ambiguous`, `underconstrained`, `no_candidates`,
  `conflicting_constraints`, or `low_confidence`.
- `reference_resolver.py` provides single-target, distance-ranking, and
  numerical-set adapters.
- `reasoning_visualisation.py` saves the complete score table, structured
  resolution/pair traces, and labelled top-down relation view.

## Configuration

The mission node exposes the Day 9 baselines as ROS parameters:

```text
reasoning_minimum_class_probability = 0.15
reasoning_minimum_colour_probability = 0.10
reasoning_minimum_geometry_confidence = 0.20
reasoning_near_base_margin_m = 0.40
reasoning_near_size_scale = 0.75
reasoning_between_projection_tolerance = 0.05
reasoning_between_max_relative_perpendicular_distance = 0.35
reasoning_between_min_anchor_separation_m = 0.30
reasoning_resolved_minimum_score = 0.65
reasoning_resolved_minimum_margin = 0.12
reasoning_ambiguous_margin = 0.08
corridor_safety_clearance_m = 0.15
corridor_minimum_depth_m = 0.60
corridor_occupancy_free_fraction = 0.90
corridor_maximum_anchor_separation_m = 5.0
```

`CorridorConfig` deliberately requires `robot_width_m`. The runtime value is
loaded from the system-level `robot_footprint_width_m` parameter; it is not a
second reasoning-only robot-size constant. Corridor occupancy must be supplied
as a finite `PlanningGrid` whose obstacles have already been inflated for the
configured robot footprint.

## Evidence And Reproduction

The synthetic tests include four-table six-pair enumeration, closest ties,
weak farthest geometry, beyond-endpoint `between`, narrow and occupied gates,
blocked approach/exit, a third-object blocker, wrong-colour distractors,
missing anchors, and complete scoring that defeats a nearest-only shortcut.

The saved Office 1 simulator audit is under:

```text
benchmark/day9_office1/
```

It uses all 112 released Office 1 object annotations. The audit considers 12
chair candidates, all 12 chair-window products, 66 chair pairs, and the one
available table pair. It saves the full candidate table, pair evidence,
resolution trace, report, and top-down image.

Run the audit in the competition image with the workspace data mounted:

```bash
python3 tools/day9_office1_reasoning.py \
  --questions /data/questions/questions.json \
  --simulation-root /data/simulation \
  --output-directory /output
```

Run focused tests with:

```bash
python3 -m pytest -q test/test_day9_*.py
```

The authoritative acceptance gate remains the clean Docker build followed by
`colcon test --packages-select qmapnav` and the saved Day 3, Day 6, Day 7, and
Day 8 regression commands documented in the corresponding day files.
