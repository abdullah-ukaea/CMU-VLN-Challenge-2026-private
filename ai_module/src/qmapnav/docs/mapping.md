# Registered-Scan Map Foundation

`RegisteredScanAccumulator` is the protocol persistent-map foundation. It consumes
XYZ arrays through a ROS-independent API; `mapping.point_cloud` is the only
adapter that decodes `sensor_msgs/msg/PointCloud2`.

The ROS node subscribes to the permitted `/registered_scan` topic. Input must
declare the exact `map` frame. Other frames are counted and rejected so scans
are never silently mixed without a transform.

## Bounded Policy

The provisional defaults are intentionally configurable:

| Setting | Default |
|---|---:|
| Voxel size | `0.20 m` |
| Rolling XY range from the current sensor origin | `30 m` |
| Maximum observation age | `120 s` |
| Hard occupied-voxel cap | `200,000` |
| Retained scan views | `16` |

Non-finite points are ignored. Equal voxel keys merge, so replaying an
identical scan does not increase occupied-map size. Age and rolling-range
eviction run before insertion; deterministic oldest-first eviction enforces the
hard cap afterward. Reset clears contents and statistics at an episode/process
boundary.

The accumulator exposes defensive voxel-centre snapshots plus accepted,
empty, rejected, stale, invalid-point, and eviction counters.

## Conservative Recovery Evidence

When a sensor origin is available, each scan retains a bounded 2D angular view.
For navigation-height returns (`0.10-1.80 m`), the nearest hit in each two-degree
bin bounds observed free ray space. A recovery candidate is accepted only when:

- a retained ray extends beyond the candidate by the configured clearance;
- the candidate is within `1.0 m` of that observed origin;
- no accumulated navigation-height voxel violates its clearance.

The deterministic safe-offset policy checks lateral, diagonal-backward, and
backward candidates. It does not treat absence of occupied points as proof of
free space. This is a conservative protocol recovery primitive, not the full
mapping or planning system scheduled for later days.

## projection dense projection map

`DenseRegisteredScanAccumulator` deliberately remains separate from this
coarse navigation/recovery map. It uses configurable age, radius, raw-point,
and centroid-voxel bounds to preserve denser tabletop and furniture geometry
for projection. Its contract and measured behavior are documented in
[`projection.md`](projection.md).

## lifting single-observation object geometry

lifting consumes perception detections and projection projected support without changing
either contract. It selects wrap-aware box or optional mask proposals, removes
ground and background depth layers, selects a coherent distance-aware cluster,
and fits both a robust AABB and upright OBB. Orientation and overall geometry
confidence remain separate, and weak yaw evidence produces an explicit
map-aligned fallback.

These outputs are observation candidates only. They do not implement persistent
identity, cross-view fusion, structural extraction, relations, or final answer
selection. Contracts, defaults, measured Office 1 results, marker-topic safety,
and the five-case replay pack are documented in
[`lifting_lifting.md`](lifting_lifting.md).

## mapping persistent objects and structures

mapping adds conservative episode-local IDs, one-to-one association, bounded
multi-view fusion, wall extraction, and ray-to-wall anchoring for windows,
openings, and wall-mounted landmarks. The frozen shared `ObjectInstance`
remains the reasoning snapshot; histories, fused points, crops, statuses, and
diagnostics live in bounded mapping-local records. Architectural features use a
separate `StructuralAnchor` contract. The score, gates, memory policy, debug
topics, trace records, and regressions are documented in
[`mapping_mapping.md`](mapping_mapping.md).


## Object lifting and geometry

lifting converts one perception `Detection2D` and one projection `ProjectionResult` into
an explicitly uncertain, single-observation `ObjectCandidate3D`. It does not
assign persistent IDs, merge observations, select an answer, or perform any
mapping+ structural or relational reasoning.

```text
Detection2D + ProjectionResult
  -> wrap-aware image-region proposals
  -> local ground removal
  -> foreground depth layer
  -> distance-aware spatial clustering
  -> robust AABB and upright OBB
  -> orientation/geometry confidence
  -> internal map-frame CUBE marker
```

## Contracts And Provenance

`ObjectCandidate3D` retains immutable cleaned map points and their source
projection indices, the detection ID/class/confidence, all intermediate point
counts, the robust point median, AABB, OBB, raw estimated yaw, publication yaw,
orientation confidence, geometry confidence, partial-geometry flag, timestamps,
and whether support came from the current, accumulated, or combined cloud.
The pose timestamp is the projection interpolated image-time pose; association mode,
image/scan delta, surrounding-pose deltas, and the timing-warning flag are
copied into the candidate rather than reduced to receipt time.

Normal weak-evidence outcomes are structured states rather than exceptions:

```text
good
sparse
no_points
ground_dominated
background_contaminated
multiple_clusters
unstable_orientation
invalid_geometry
```

The production composition defaults to the accumulated projection cloud. Current
and combined support remain configurable diagnostic modes. Candidates are
still independent observations: using an accumulated cloud is densification,
not mapping identity fusion.

## Point Selection And Cleaning

The baseline contracts every wrap-aware `PanoramaBox` by five percent on all
sides. Horizontal membership is circular, so a box split over the panorama
seam remains one region. A full panorama mask or one or more mapped detector
polygons can replace the box through the same selector. Invalid or missing
masks deterministically fall back to the contracted box.

Ground is estimated in `map` from the lowest point in local XY cells, followed
by a bounded robust plane refit. Furniture uses `0.07 m` clearance. Known
floor-standing classes such as potted plants and bins use `0.02 m` so their
bases are not erased. An unavailable or low-confidence plane is reported and
does not silently remove points.

The foreground depth filter chooses the nearest substantial contiguous
histogram mode. Its retained band is bounded to `1.5 m`; a robust front-depth
percentile is the explicit fallback. This rejects a wall behind an object
without allowing a handful of nearer noise points to become the object.

The spatial stage uses deterministic grid-neighbour DBSCAN with:

```text
epsilon(depth) = 0.07 m + 0.015 * median_depth_m
minimum samples = 5
```

Epsilon is bounded to `[0.07, 0.30] m`. Cluster selection combines point
support, foreground depth, projected-centre alignment, and compactness. The
nearest or largest cluster is therefore not accepted blindly. Close scores
produce `multiple_clusters` while preserving the selected primary cluster and
alternative scores in diagnostics.

## AABB, OBB, And Confidence

Both boxes use the cleaned primary cluster and robust 2.5/97.5 percentiles.
The AABB remains in map axes for fast conservative geometry. The upright OBB
uses the minimum-area rectangle of the XY convex hull and robust Z extent. PCA
is computed independently as an orientation diagnostic.

The canonical representation is:

```text
length >= width
yaw describes the length axis
yaw in [-pi/2, pi/2)
roll = pitch = 0
```

Orientation confidence combines point support, PCA anisotropy, agreement
between minimum-area and PCA yaw, deterministic subset-resampling stability,
depth consistency, cluster purity, timing quality, and support away from the
detection boundary. It is separate from geometry confidence.

```text
confidence >= 0.70       use supported OBB yaw
0.40 <= confidence < .70 use yaw, explicitly uncertain
confidence < 0.40        yaw = 0 and use map-frame AABB dimensions
```

The final case avoids false angular precision while retaining a credible
position and conservative size. Near-square, circular, sparse, partial, and
one-surface observations are expected to receive low confidence.

## Marker Safety Boundary

The ROS-independent `MarkerSpec` adapter always produces an upright `CUBE` in
`map`, with positive dimensions (minimum `0.05 m`) and the yaw quaternion
`[0, 0, sin(yaw/2), cos(yaw/2)]`.

Two topics have intentionally different roles:

```text
/qmapnav/debug/object_candidates  visualization_msgs/MarkerArray
/selected_object_marker          visualization_msgs/Marker
```

Every projection frame may replace the internal candidate array. The official
publisher is reachable only through `FinalMarkerGuard.commit()`, which permits
one explicit committed candidate per episode. lifting never calls that method
automatically. The live validation produced three map-frame debug CUBE markers
and no message on `/selected_object_marker`.

## Evaluation

Pure evaluation helpers report 3D/XY/Z centre error, canonical dimension
absolute and relative error, 180-degree-symmetric yaw error, AABB IoU, upright
oriented 3D IoU, and the evidence bins `0`, `1-5`, `6-10`, `11-30`, `31-100`,
and `>100`. Oriented intersection is computed from the two convex XY
footprints and their vertical overlap.

Four saved Office 1 panoramas yielded:

| Measurement | Box | Native YOLOE mask |
|---|---:|---:|
| Successful lifts / 68 detections | 44 | 45 |
| GT-matched pairs | 43 | 43 |
| Median centre error | `0.397 m` | `0.416 m` |
| Median oriented IoU | `0.02474` | `0.02472` |
| Median lifting time | `18.02 ms` | `44.10 ms` |

The report also retains attempts, lift rate, evaluated count, median centre
error, oriented IoU, and yaw error separately for every required point-count
bin. In the box path, the `1-5` point bin had `0.809 rad` median yaw error,
whereas `11-30` points had `0.066 rad`; this supports explicit low-evidence
handling rather than treating every fitted angle as equally trustworthy.

The low median IoU is an honest single-scan partial-surface result, not an
object-fusion result. The strongest visible shelf candidate used 461 points,
had orientation confidence `0.736`, centre error `0.389 m`, yaw error
`0.061 rad`, and oriented IoU `0.301`. Its estimated dimensions under-cover
the full reference shelf because the current scan sees only part of it; the
candidate correctly records this as one observation and does not apply a class
prior to fabricate the unseen volume.

The complete benchmark took `37.34 s` including four panorama detector runs,
136 lifts, and saved diagnostics. Median diagnostic rendering and file output
per detection was `169.94 ms`, deliberately outside ROS callbacks. Peak RSS
was `2,018,296 KiB`. PyTorch reported `109,215,744` peak allocated and
`138,412,032` peak reserved CUDA bytes during this run; the independently
measured perception full process peak for the selected detector remains 662 MiB.

## Segmentation Decision

Contracted boxes are the production default. The native YOLOE mask selector is
retained behind `lifting_use_masks=false` because it is useful for controlled
cases and costs no second model, but it did not improve median centre error or
oriented IoU and approximately doubled lifting time in this benchmark. No SAM2
or external segmentation pass is added. Reconsidering this decision requires a
material measured geometry or failure-rate improvement under the existing
runtime and VRAM limits.

## Regression Evidence

Five source-complete real cases are stored below:

```text
/home/abdul/cmu-vln//home/abdul/cmu-vln/data/lifting/lifting_regressions/regressions/
  large_box_like/   bookshelf, 461 points, supported yaw
  narrow_object/    chair, conservative low-yaw-confidence fallback
  floor_standing/   potted plant, low-clearance sparse support
  small_tabletop/   computer monitor, 24-point geometry
  wall_adjacent/    wall clock, thin wall-adjacent geometry
```

Each contains raw projected arrays, the exact `Detection2D`, ground plane,
lifting policy, source indices, baseline state/box/confidence, panorama, stage
overlay, depth histogram, orthographic geometry view, and SHA-256 checksums.
The replay tool requires exact status, selected indices, centre, dimensions,
yaw, and valid checksums:

```bash
python3 tools/external_lifting_replay.py \
  /home/abdul/cmu-vln//home/abdul/cmu-vln/data/lifting/lifting_regressions/regressions
```

Synthetic tests cover rotated 0/30/45/80-degree boxes, square and circular
footprints, ground, wall background, foreground outliers, two nearby objects,
sparse 3/8/20-point behavior, seam-aware selection, invalid masks, confidence
fallback, marker guarding, and deliberately mutated saved baselines.

## Final Validation

The network-disabled rebuilt competition image reports:

```text
Q-MapNav package: 605 passed, 0 failed
evaluation quick:      6/6 structural, 3/3 instruction, 18/18 proxy
evaluation full:       75/75 structural, 30/30 instruction, 180/180 proxy
lifting replay:     5/5 exact, checksums valid
```

The live Office 1 node published only internal map-frame markers. A sampled
candidate CUBE had centre `[-0.697, -1.507, 0.735] m`, dimensions
`[0.265, 0.089, 0.420] m`, and non-identity yaw quaternion
`[0, 0, 0.645, 0.764]`. Two additional candidates included the intended
low-confidence identity-yaw fallback. A five-second official-topic audit
received no marker.

Remaining limitations are explicit: a single partial scan often under-covers
object size; sparse tabletop geometry may fail normally; nearest-reference
matching is development-only evaluation and never enters runtime; and improved
multi-view box completeness belongs to mapping fusion, not lifting.


## Persistent objects and structural anchors

mapping converts independent lifting candidates into episode-local identities and
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

No colour colour evidence or spatial-relation reasoning is performed.

## Contract Compatibility

`qmapnav.common.ObjectInstance` has a baseline regression-frozen field order. Day
7 does not change it. `ObjectMap.get()` and `active_instances()` return that
reasoning-facing contract. `PersistentObjectRecord` wraps it with the mapping
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

The rebuilt competition image passed all 630 package tests. The evaluation oracle
regressions remained 6/6 quick and 75/75 full with 100% available-label proxy
score, and all five saved lifting lifting cases replayed with valid checksums and
zero centre, dimension, yaw, status, or point-index drift.
