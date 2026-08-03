# Day 5 Frame and Timestamp Contract

This document records the measured Office 1 simulator contract used by the
camera-LiDAR projection path. Frame direction is explicit; leading slashes in
`/tf_static` names are treated as ROS naming syntax, not different frames.

## Live topic evidence

| Topic | Type | Header frame | Source timestamp | Measured source rate |
|---|---|---|---|---:|
| `/camera/image` | `sensor_msgs/msg/Image` | `camera` | camera message header | `3.965 Hz` |
| `/registered_scan` | `sensor_msgs/msg/PointCloud2` | `map` | scan message header | `3.965 Hz` |
| `/state_estimation` | `nav_msgs/msg/Odometry` | parent `map`, child `sensor` | odometry header | `197.65 Hz` |

The scan fields are float32 `x`, `y`, `z`, and `intensity`. The challenge
system describes `/registered_scan` as registered by state estimation and its
header is `map`; Q-MapNav therefore consumes its points as `p_map` and rejects
any other header frame. The code does not claim that an individual sweep is
internally motion compensated: that detail is not exposed by the interface.
Static-scene alignment is the Day 5 empirical contract.

The 113.18-second audit contains 448 images, 448 registered scans, and 22,371
poses. Image-to-nearest-pose source delta is exactly `0 ms` for every image.
Image-to-nearest-scan delta is `0 ms` through the 95th percentile and within
`50 ms` for 447/448 images. The one `245.15 ms` bag-edge sample is rejected by
the configured `150 ms` scan threshold. Receipt latency is not used for
association.

The source data and complete report are outside the submission repository:

```text
/home/abdul/cmu-vln/data/day4/day5_office1_multi_pose/
/home/abdul/cmu-vln/data/day4/day5_sensor_audit.json
```

## Axis conventions

ROS optical camera axes are:

```text
+X right
+Y down
+Z forward
```

The Day 4 panorama and robot-style sensor basis is:

```text
+X forward
+Y left
+Z up
```

The fixed optical-to-internal rotation is:

```text
R_internal_from_optical =
[[ 0,  0,  1],
 [-1,  0,  0],
 [ 0, -1,  0]]
```

The panorama uses `u_yaw_sign=-1`, so increasing `u` rotates toward camera
right. Image `v` increases downward. Its verified vertical range is pitch
`[-60 deg, +60 deg]`; points outside it are invalid, not clipped. Horizontal
coordinates wrap modulo 1920 pixels.

## Static camera extrinsic

The simulator publishes this static transform:

```text
parent: /sensor
child:  /camera
translation sensor_from_camera: [0.0, 0.0, 0.1] m
quaternion sensor_from_camera XYZW:
[-0.5, 0.4999999866, -0.5, 0.5000000134]
```

It is `T_sensor_from_camera_optical`, not its inverse. The nearly exact
rotation is:

```text
R_sensor_from_camera_optical =
[[ 0,  0,  1],
 [-1,  0,  0],
 [ 0, -1,  0]]
```

At the initial sensor pose (`z=0.75 m`), this places the camera origin at
`z=0.85 m`, consistent with the visible projection.

This matches both the active autonomy-stack and SysNav
`local_planner.launch` static arguments (`0 0 cameraOffsetZ -pi/2 0 -pi/2`)
and their simulation launch default `cameraOffsetZ=0.1`. Bagfile/real-robot
launches use `0.25 m`, so the extrinsic remains configurable and must be
re-audited before real-robot replay.

## Transform chain

`/state_estimation` supplies `T_map_from_sensor(t)`. Projection composes:

```text
T_camera_internal_from_map(t)
  = T_internal_from_camera_optical
  * inverse(T_sensor_from_camera_optical)
  * inverse(T_map_from_sensor(t))

p_camera_internal
  = T_camera_internal_from_map(t_image) * p_map
```

Every SE(3) input is checked for finite values, homogeneous bottom row,
orthonormal rotation, and determinant `+1`. Quaternions are normalized and
zero-norm quaternions are rejected. Tests cover forward, left, right, above,
below, behind, and `+pi/-pi` seam directions.

## Timing policy

The production defaults are:

```text
pose buffer:           5 s / 2000 samples
scan buffer:           5 s / 64 samples
maximum pose delta:   50 ms
maximum scan delta:  150 ms
```

Translation is linearly interpolated and orientation uses shortest-arc
quaternion SLERP when bracketing poses are available. An exact pose is used
when present. A single nearest pose is accepted only inside the pose threshold.
Ties select the earlier source sample. Callbacks only decode and insert into
short locked buffers; a bounded worker projects selected panoramas.

## Empirical result

The saved bag spans `1.54 m` of robot translation and almost the full yaw
range. Replayed keyframes at `(0.00, 0.00, 0.00 rad)`,
`(1.47, 0.14, 0.48 rad)`, and `(0.19, 0.20, -2.61 rad)` all use exact source
timestamps and visibly align wall, floor, shelving, desk, and chair returns.
Changing the panorama yaw sign intentionally causes every saved regression
case to fail.
