# Camera-LiDAR Projection and Rolling Densification

projection connects perception panorama detections to map-frame geometry without making
semantic point-membership or object-box claims. Those remain lifting work.

## Runtime flow

The mission node subscribes only to permitted challenge inputs:

```text
/state_estimation -> TimedPose buffer
/registered_scan  -> TimedRegisteredScan buffer + dense accumulator
/camera/image     -> bounded projection worker
```

`ProjectionSynchronizer` associates an image with an exact/interpolated pose
and nearest acceptable scan by source timestamp. `projectionProjectionPipeline`
projects both the current scan and a voxelized rolling snapshot using the same
image-time transform. The current scan diagnoses timing/calibration; the
accumulated scan diagnoses registration stability and densifies geometry.

ROS `PointCloud2` conversion validates `x/y/z`, optionally retains intensity,
rejects an unexpected frame, removes non-finite XYZ points, and handles empty
clouds. Point transforms and angular projection are vectorized NumPy
operations. A full panorama retains front and rear points; filtering uses
Euclidean range and the verified 120-degree vertical coverage.

`project_result_into_crops()` reuses perception `PerspectiveGeometry`. Overlapping
crops may legitimately contain the same source point, and panorama projection
remains the global representation.

## Dense scan policy

The projection dense accumulator is separate from the conservative protocol navigation
map. Its measured defaults are:

| Setting | Default |
|---|---:|
| Maximum age | `15 s` |
| Maximum radius from latest sensor origin | `12 m` |
| Raw point cap | `1,000,000` |
| Centroid voxel size | `0.04 m` |

It stores time-ordered map-frame chunks, evicts old/out-of-radius data, applies
an exact oldest-first raw-point cap, and creates deterministic centroid voxels.
Image-time snapshots exclude later scans even when the bounded worker is
processing an older keyframe. Snapshots retain observation count and latest
source timestamp per voxel. The
Office 1 replay reached a bounded 61-62 scans and 646k-657k raw points at later
poses. At the middle pose, 657,127 raw points reduced to 50,491 projected voxel
centroids while preserving furniture and tabletop surfaces.

The measured `0.03/0.04/0.05 m` sweep produced 69,116/50,491/38,055 voxels.
The `0.04 m` snapshot was fastest (`0.836 s`), retained 33 percent more detail
than `0.05 m`, and used 27 percent fewer voxels than `0.03 m`; visual replay
still preserved desk, monitor, shelf, and chair geometry. The sweep peaked at
264,080 KiB RSS while holding all three accumulators simultaneously. Results
are in `/home/abdul/cmu-vln/data/perception/projection_voxel_sweep.json`.

## Detection support

`DetectionProjection` summarizes points whose pixels fall inside a perception
`Detection2D`; it does not assert those points belong to the object. It records:

- point count and source projection indices;
- min/median/max depth and depth IQR;
- occupied box-cell fraction;
- image/scan and pose timing deltas;
- `good`, `sparse`, `high_depth_spread`, `timing_warning`, or `no_points`.

Panorama boxes with two seam intervals are handled directly. Empty and sparse
regions never fabricate a depth estimate.

## Diagnostics

The visualization module provides:

- current-scan and accumulated-scan depth overlays;
- a sparse nearest-depth z-buffer for readable displays;
- detection boxes with count, median, IQR, and quality;
- perspective-crop point/box overlays;
- a top-down current/accumulated map and sensor heading.

The ROS node can save bounded debug frames using
`projection_debug_directory` and `projection_max_saved_frames`. Optional
`projection_regression_*` parameters add source-complete replay metadata.

## Saved regression pack

Real Office 1 examples are stored at:

```text
/home/abdul/cmu-vln/data/perception/projection_regressions_multi_pose/
  nearby_furniture/   # pose B
  walls/              # pose A
  tabletop_objects/   # pose A
  panorama_seams/     # pose C
  sparse_detections/  # pose B
```

Each case contains `panorama.png`, `overlay.png`, raw scan `inputs.npz`, and a
`manifest.json` with pose, frames, transform, timing, projection policy, and up
to 2048 source-indexed baseline pixels. Detector runs add
`detection_overlay.png`, `detections.json`, and eight crop overlays. The pack's
three poses and accumulator measurements are indexed by
`multi_pose_summary.json`.

`replay_projection_regression_case()` recomposes the transform from saved pose
and extrinsic, reprojects raw map points, handles circular pixel discrepancy,
and checks valid point count plus every sampled source point. All five real
cases replay at zero pixel error. Replaying with `u_yaw_sign=+1` fails all five,
demonstrating that calibration/sign regressions are detected.

On the 10,619-point moving-pose case, pure panorama projection measured
`4.17 ms` median, mapping into all eight crops `5.76 ms`, and a source-complete
saved replay including I/O `36.93 ms`. Debug depth rendering measured
`151.62 ms` and therefore remains outside ROS callbacks. The timing process
peaked at 191,188 KiB RSS; the perception exact-question detector allocated 64 MiB
of CUDA memory after inference. Full measurements are in
`/home/abdul/cmu-vln/data/perception/projection_projection_benchmark.json`.

Development tools:

```bash
python3 tools/external_projection_bag_audit.py BAG OUTPUT.json
python3 tools/external_projection_replay.py BAG OUTPUT_DIRECTORY
python3 tools/external_projection_overlay.py CASE_DIRECTORY
python3 tools/external_projection_voxel_sweep.py BAG SUMMARY OUTPUT.json
python3 tools/external_projection_benchmark.py CASE OUTPUT.json
```

## Failure behavior

No pose or scan inside its threshold returns a structured association failure.
Wrong frames and invalid transforms are rejected. Empty clouds return a valid
empty projection. All-behind is not an error for a 360-degree panorama; rear
points map to the opposite horizontal side. Worker overload drops the oldest
queued image and retains newest work, and worker failures are counted with the
latest error string.

## Focused tests

```bash
pytest -q \
  test/test_transforms.py \
  test/test_timed_buffers.py \
  test/test_point_cloud.py \
  test/test_lidar_camera_projection.py \
  test/test_dense_scan_accumulator.py \
  test/test_projection_quality.py \
  test/test_projection_visualisation.py \
  test/test_projection_pipeline.py \
  test/test_projection_worker.py \
  test/test_projection_regression.py \
  test/test_mission_node.py
```
