# Day 6 Single-Observation 3D Lifting

Day 6 converts one Day 4 `Detection2D` and one Day 5 `ProjectionResult` into
an explicitly uncertain, single-observation `ObjectCandidate3D`. It does not
assign persistent IDs, merge observations, select an answer, or perform any
Day 7+ structural or relational reasoning.

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
The pose timestamp is the Day 5 interpolated image-time pose; association mode,
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

The production composition defaults to the accumulated Day 5 cloud. Current
and combined support remain configurable diagnostic modes. Candidates are
still independent observations: using an accumulated cloud is densification,
not Day 7 identity fusion.

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
one explicit committed candidate per episode. Day 6 never calls that method
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
measured Day 4 full process peak for the selected detector remains 662 MiB.

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
/home/abdul/cmu-vln/data/day6/day6_regressions_office1/regressions/
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
python3 tools/day6_replay_regressions.py \
  /home/abdul/cmu-vln/data/day6/day6_regressions_office1/regressions
```

Synthetic tests cover rotated 0/30/45/80-degree boxes, square and circular
footprints, ground, wall background, foreground outliers, two nearby objects,
sparse 3/8/20-point behavior, seam-aware selection, invalid masks, confidence
fallback, marker guarding, and deliberately mutated saved baselines.

## Final Validation

The network-disabled rebuilt competition image reports:

```text
Q-MapNav package: 605 passed, 0 failed
Day 3 quick:      6/6 structural, 3/3 instruction, 18/18 proxy
Day 3 full:       75/75 structural, 30/30 instruction, 180/180 proxy
Day 6 replay:     5/5 exact, checksums valid
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
multi-view box completeness belongs to Day 7 fusion, not Day 6.
