# Registered-Scan Map Foundation

`RegisteredScanAccumulator` is the Day 2 persistent-map foundation. It consumes
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
free space. This is a conservative Day 2 recovery primitive, not the full
mapping or planning system scheduled for later days.

## Day 5 dense projection map

`DenseRegisteredScanAccumulator` deliberately remains separate from this
coarse navigation/recovery map. It uses configurable age, radius, raw-point,
and centroid-voxel bounds to preserve denser tabletop and furniture geometry
for projection. Its contract and measured behavior are documented in
[`projection.md`](projection.md).

## Day 6 single-observation object geometry

Day 6 consumes Day 4 detections and Day 5 projected support without changing
either contract. It selects wrap-aware box or optional mask proposals, removes
ground and background depth layers, selects a coherent distance-aware cluster,
and fits both a robust AABB and upright OBB. Orientation and overall geometry
confidence remain separate, and weak yaw evidence produces an explicit
map-aligned fallback.

These outputs are observation candidates only. They do not implement persistent
identity, cross-view fusion, structural extraction, relations, or final answer
selection. Contracts, defaults, measured Office 1 results, marker-topic safety,
and the five-case replay pack are documented in
[`day_6_lifting.md`](day_6_lifting.md).

## Day 7 persistent objects and structures

Day 7 adds conservative episode-local IDs, one-to-one association, bounded
multi-view fusion, wall extraction, and ray-to-wall anchoring for windows,
openings, and wall-mounted landmarks. The frozen shared `ObjectInstance`
remains the reasoning snapshot; histories, fused points, crops, statuses, and
diagnostics live in bounded mapping-local records. Architectural features use a
separate `StructuralAnchor` contract. The score, gates, memory policy, debug
topics, trace records, and regressions are documented in
[`day_7_mapping.md`](day_7_mapping.md).
