# Day 10 Object-Reference Benchmark

Date: 4 August 2026  
Status: complete  
Frozen run: `full_annotated_map_frozen`

## Outcome

All 30 released object-reference questions reached a terminal record and logged
a response. All 30 produced one protocol-valid map-frame marker. The annotated
development proxy selected 24/30 targets and passed the marker proxy for 23/30,
for 46/60 proxy points. Mean controlled episode time was 0.440 seconds.

The frozen run is in
`data/day10/benchmark/full_annotated_map_frozen`. It contains the exact manifest,
run metadata, per-case records, parser audit, scene/tag aggregates, diagnostic
evidence, fix ranking, and human-readable report.

These correctness labels are development proxies derived from the same released
VLA3D annotations used to construct the scene map. The controlled run injects
those boxes through the production persistent `ObjectMap`; it therefore verifies
parsing, fusion, joint relation solving, ranking, OBB adaptation, marker protocol,
failure classification, and bounded logging, but is not an independent estimate
of detector or LiDAR accuracy.

## Baseline And Final Comparison

| Metric | Baseline | Frozen | Change |
| --- | ---: | ---: | ---: |
| Terminal logged responses | 30/30 | 30/30 | 0 |
| Protocol-valid markers | 28/30 | 30/30 | +2 |
| Proxy-correct target selections | 22/30 | 24/30 | +2 |
| Proxy marker successes | 21/28 | 23/30 | +2 |
| Proxy score | 42/60 | 46/60 | +4 |

Two bounded fixes were made from baseline evidence:

- canonical class aliases now cover `desk_light`/`lamp`,
  `flowers`/`flower`, `nightstand`/`night_stand`, and
  `zen_stone_decoration`/`stone_decoration`;
- repeated textual anchor roles may bind the same physical persistent instance,
  while true self-relations still reject an identical participant.

Neither fix contains a scene coordinate, released object ID, trajectory, or
question-specific answer.

## Pipeline Verification

- Parser: 30/30 used the full deterministic parse. The audit records the target,
  attributes, anchors, relation arguments, and absence of instruction steps.
- Target and anchors: all requested classes were available in the frozen
  annotated control after canonicalisation. The real Office 1 smoke detected two
  potted-plant candidates and one file-cabinet anchor.
- Lifting: every controlled object has explicit stage counts. The perceived smoke
  saved projected-point counts, post-ground/depth/cluster counts, statuses, and
  projection images from the real camera/LiDAR path.
- Fusion: every case records association events and persistent IDs. In the live
  smoke, the file cabinet and two plant candidates retained their IDs across
  three observations before commitment.
- Relations and ranking: every case stores all ranked target hypotheses, score
  decompositions, margins, satisfied/violated constraints, and unresolved terms.
- OBB: the final adapter snapshots the selected persistent instance, canonicalises
  dimensions, applies the low-confidence yaw fallback, and validates finite map
  geometry before publication.
- Protocol and timing: 30/30 frozen markers passed frame, type, action,
  quaternion, and dimension checks and were published exactly once after commit.
  Matching waypoints use the same centre.
- Targeted viewpoint: low target evidence, a missing anchor, a low ranking margin,
  weak geometry, or likely occlusion can request one safe viewpoint. A hard guard
  prevents a second request. The real smoke triggered one lateral viewpoint for
  a 0.016 margin, reranked once, and committed at 24.65 seconds.
- Failure attribution: the deterministic decision tree covers parsing, missed
  target, missed anchor, incorrect colour, bad lifting, duplicate instance, bad
  relation, incorrect OBB, and protocol failure, always choosing the earliest
  causal stage.

## Frozen Failure Counts

| Primary category | Count |
| --- | ---: |
| Parsing | 0 |
| Missed target | 0 |
| Missed anchor | 0 |
| Incorrect colour | 0 |
| Bad lifting | 0 |
| Duplicate instance | 0 |
| Bad relation | 6 |
| Incorrect OBB | 1 |
| Protocol failure | 0 |

The six relation disagreements are in Arabic Room 02, Chinese Room 02,
Home Building 2 02, Hotel Room 1 02, Living Room 2 02, and Living Room 3 01.
Hotel Room 2 02 is the OBB proxy failure: the selected picture and centre are
correct, but the released 8 mm thickness conflicts with the conservative 50 mm
minimum marker thickness, producing 0.161 OBB IoU.

## Three Largest Failure Sources

1. Joint relation/ranking disagreement affected six frozen cases. The complete
   inputs exist, but production scoring disagrees with the annotation-derived
   proxy target.
2. Query-to-map vocabulary and repeated-anchor coreference affected three
   baseline cases. The bounded aliases/coreference fix recovered the missing
   candidates and remains a regression risk even though its frozen count is zero.
3. Thin-object OBB conservatism affects one case. It is a proxy/scoring-adapter
   trade-off rather than a target-selection failure.

## Fix Priority

Priority is expected recovered score divided by implementation effort and risk.

| Rank | Source | Affected | Confidence | Effort | Priority | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Vocabulary/coreference | 3 baseline | 0.90 | 0.5 | 10.80 | Implemented and retained |
| 2 | Bad relation | 6 frozen | 0.70 | 2.0 | 4.20 | Next measured tuning target |
| 3 | Thin-object OBB | 1 frozen | 0.80 | 1.5 | 1.07 | Defer pending scoring clarification |

The relation fix is not applied on Day 10 because threshold tuning against only
these released answers would risk benchmark overfitting. The OBB minimum is not
lowered because an 8 mm box is fragile under perceived geometry and the official
IoU convention remains unverified.

## Perceived Smoke

`office_1_object_reference_01` ran against the actual Unity camera, LiDAR, YOLOE
Day 4 baseline, projection/lifting worker, persistent maps, solver, targeted
viewpoint, and ROS publishers. It produced a full parse, three 3D candidates,
three persistent instances, one targeted viewpoint, one final map-frame marker,
one matching waypoint, no marker validation error, and a terminal record in
24.65 seconds. Because no independent selected-instance label is available for
this runtime episode, its semantic correctness remains explicitly `null`.

The detector was the frozen `yoloe-11s-seg.pt` baseline with confidence 0.20,
eight perspective crops, and cross-crop IoU 0.40. The source began from Git commit
`878dc3a2f0c565222c76a4065e1da0f2822e24cf` plus the documented uncommitted Day 10
working-tree changes.

## Reproduction

The installed console entry is `qmapnav_object_benchmark`. Required inputs are
the released `questions.json`, all 15 extracted simulation scenes, and the local
VLA3D `Unity` metadata directory. Use `--mode quick` for the six-case regression
subset or `--mode full` for all 30 cases. A run ID is immutable unless the runner
is explicitly invoked with its resume mode.

The principal artifacts are:

- `manifest.json` and `run_metadata.json`;
- `parser_audit.json` and `grouped_metrics.json`;
- `summary.json`, `per_case.json`, `fix_priorities.json`, and `report.md`;
- one evidence directory per case containing the task specification, lifting and
  fusion diagnostics, complete ranking, marker validation, trace, and result.

The final validation gate is the full package test suite, including lint and
docstring checks. No Day 11 exploration or instruction-following work is part of
this implementation.
