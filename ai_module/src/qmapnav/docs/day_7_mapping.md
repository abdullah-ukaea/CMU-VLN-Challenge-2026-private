# Day 7 Persistent Objects And Structural Map

Day 7 converts independent Day 6 candidates into episode-local identities and
represents architectural features separately from ordinary objects.

```text
ObjectCandidate3D + ViewpointObservation
  -> class and spatial gates
  -> auditable association score
  -> one-to-one viewpoint assignment
  -> bounded evidence fusion
  -> frozen common.ObjectInstance snapshot + persistent metadata

dense map points -> vertical support -> wall segments
Detection2D camera ray + T_map_from_camera + wall plane
  -> StructuralAnchor
```

No Day 8 colour evidence or spatial-relation reasoning is performed.

## Contract Compatibility

`qmapnav.common.ObjectInstance` has a Day 1 regression-frozen field order. Day
7 does not change it. `ObjectMap.get()` and `active_instances()` return that
reasoning-facing contract. `PersistentObjectRecord` wraps it with the Day 7
state that earlier consumers do not need: canonical class, geometry confidence,
first/last source timestamps, unique viewpoint and detection IDs, bounded
observation history, fused points, best crop and score, best-view candidate ID,
and status.

`ViewpointObservation` validates a map pose encoded as XYZ plus yaw, source
time, detection and point-count agreement, geometry confidence, and
`full`/`partial`/`sparse` visibility. Crops are defensively copied.

Architectural features use `StructuralAnchor`, never `ObjectInstance`. Walls
carry an XY segment and vertical plane. Windows, openings, and wall-mounted
objects carry a stable map position, approximate extent, wall-aligned yaw, and
the supporting wall ID.

## Object Association

Detector names are normalized to snake case and aliases include
`couch -> sofa`, `monitor -> computer_monitor`, and waste/garbage-bin names to
`trash_can`. Exact class matches score `1.0`. Measured compatibility rules can
permit ambiguous pairs such as chair/stool without allowing arbitrary labels.

Incompatible classes and centroid distances outside a class/size/confidence
gate are rejected before detailed scoring. Accepted comparisons combine:

```text
0.20 class compatibility
0.32 Gaussian centroid distance
0.18 AABB IoU
0.20 dimension similarity
0.10 reliable yaw agreement
```

Yaw is omitted and the active weights are renormalized unless both orientation
confidences and geometry confidences reach `0.50`. Partial or low-confidence
geometry omits IoU and treats its measured dimensions as incomplete lower-bound
evidence. Every result retains component scores, the final score, decision, and
explicit gate/rejection reasons.

The starting bands are:

```text
score >= 0.62             merge
0.55 <= score < 0.62     retain a separate possible_duplicate hypothesis
score < 0.55              create a new identity
```

The `0.62` merge threshold was selected from the saved real Office 1 replay:
it reduced split identities without introducing a labelled false merge. Hard
class and distance gates remain in force. Within one physical keyframe, highly
overlapping crops are suppressed; low-confidence or partial fragments may also
share an identity when their centres are within `0.30 m`.

`add_viewpoint_candidates()` first suppresses high-overlap same-keyframe crop
duplicates, then greedily applies deterministic score-sorted one-to-one
assignment. One candidate and one existing identity can each participate in at
most one accepted match. This protects neighbouring same-class objects.

## Fusion And Bounds

Centres use geometry-confidence weights. Class scores preserve additive
canonical evidence and secondary hypotheses. Yaw uses a 180-degree-symmetric
circular average only when reliable. Confidence increases only through
consistent geometry and is capped below certainty. Partial, sparse, uncertain,
and possible-duplicate states remain explicit.

Points are voxelised at `0.03 m`; novel voxels trigger AABB/OBB refitting.
Refit failure preserves the prior reliable geometry. The best single-view
candidate remains available as a guard against degraded fusion. Defaults bound
each identity to 50,000 points, the map to 500,000 points, history to 100
observations, and identities to 512. Debug serialization excludes raw images
and points by default. Reset clears every identity and restarts ID allocation
at zero.

The best-view crop score is bounded to `[0, 1]` and combines detection
confidence (`0.45`), geometry confidence (`0.35`), crop area (`0.10`), and
projected point support (`0.10`). Only a strictly better score replaces the
stored crop.

## Walls And Structural Anchors

Wall extraction removes ground/low clutter, retains XY cells with measured
vertical support, and fits bounded deterministic RANSAC lines. Segments record
support count, median residual, vertical coverage, plane, extent, yaw, and
confidence. Collinear fragments merge only when orientation and perpendicular
distance agree and the gap is below `0.60 m`; doorway-sized openings remain.
At most 50,000 supported points enter the line fitter per update.

Camera rays are rotated into map; camera translation is used only as ray
origin. For plane `n^T p + d = 0`:

```text
t = -(n^T o + d) / (n^T r)
p = o + t r
```

Parallel, behind-camera, non-finite, out-of-extent, and implausible-height hits
are rejected. Multiple walls are ranked by forward distance; nearly equal hits
are rejected as ambiguous. When a positive `lidar_depth_m` is present in the
detection metadata, depth agreement ranks the plausible forward walls and
attenuates confidence; ray-only anchoring remains available for glass. Anchor
fusion is projected into the supporting wall's along-wall/height coordinates.
Reobservations merge only on the same supporting wall, same canonical class,
and within `0.55 m`. The initial `0.40 m` proposal
split the same real Office 1 clock across poses A and B; `0.55 m` merged those
two wall-local observations while the supporting-wall and class gates remained
active.

## Runtime Diagnostics

The mission composition adds only internal debug topics:

```text
/qmapnav/debug/object_map       visualization_msgs/MarkerArray
/qmapnav/debug/structural_map   visualization_msgs/MarkerArray
```

Fused OBBs include identity, class, observation count, and confidence labels.
Association lines connect current candidates to persistent centres. Structural
markers include wall lines/normals, anchors, semantic labels, and supporting
wall IDs. Accepted camera rays are drawn from their map-frame camera origin to
the selected wall intersection. `persistent_map.png` renders the current path
point, objects, walls, and anchors when projection debug output is enabled.
None of these paths can publish `/selected_object_marker`.

Every object and structural association emits a bounded decision-trace record.
Trace-level known-object and known-structure counts now reflect the persistent
maps rather than fixed placeholders.

## Evaluation

`evaluation.instance_fusion` reports duplicate rate, false merges, mean and
maximum IDs per physical object, first-view versus fused centre/dimension error,
oriented 3D IoU and yaw error, anchor position variance, and supporting-wall
consistency. Synthetic
regressions cover:

- three-view identity persistence and revisit;
- panorama overlap suppression;
- neighbouring same-class one-to-one protection;
- partial views and chair/stool disagreement;
- hard point-memory bounds and deterministic reset;
- ground/clutter rejection and stable walls;
- doorway-gap preservation;
- perpendicular, parallel, behind, and out-of-bounds rays;
- nearest-wall selection and repeated window anchors;
- combined two-view objects, wall, and window integration;
- marker, trace-payload, metric, and top-down visualisation contracts.

The starting thresholds are measurement-driven configuration, not frozen
truths. Office 1 replay and live tuning should adjust them without changing the
contracts or conservative three-band behavior.

The final saved real Office 1 replay processed 43 labelled object observations
covering 17 proxy physical IDs. It retained 20 persistent IDs: 3 conservative
splits (`17.65%` duplicate rate), zero false merges, mean `1.176` IDs per proxy
object, and maximum 2. The structural replay processed three unique scans,
extracted 33 bounded wall segments (10 reobserved, maximum 4 observations), and
fused three clock detections from two viewpoints into one anchor on
`wall_0005`. This proxy uses nearest released cuboids as identity labels, so it
is reported alongside, rather than in place of, the controlled identity tests.

The rebuilt competition image passed all 630 package tests. The Day 3 oracle
regressions remained 6/6 quick and 75/75 full with 100% available-label proxy
score, and all five saved Day 6 lifting cases replayed with valid checksums and
zero centre, dimension, yaw, status, or point-index drift.
