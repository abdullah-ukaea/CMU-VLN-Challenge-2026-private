# Panoramic Perception Geometry

Day 4 introduces a ROS-independent front end for the challenge's cropped
equirectangular camera image. This document covers only perspective tiling,
retained coordinate transforms, and the bounded two-candidate detector smoke
harness. Cross-crop merging, scored detector selection, 3D lifting, and the
perception worker are intentionally not part of this implementation increment.

## Image and camera convention

Pure perception code receives an RGB array with shape `(height, width, 3)`.
The recorded simulation source is `/camera/image`, `bgr8`, `1920 x 640`, frame
`camera`; the extraction tool converts BGR to RGB exactly once.

`PanoramaCameraModel` uses a right-handed internal basis:

```text
+X  forward
+Y  left
+Z  up
```

Image `v` increases downward. The default panorama centre is yaw `0`, pitch
`0`, and the documented 120-degree vertical span is represented as pitch
`[-60 deg, +60 deg]`. It is not treated as a full 180-degree sphere. The
default `u_yaw_sign=-1` means increasing image `u` decreases yaw (moves toward
camera right). This sign is an explicit camera-model parameter; final physical
camera/LiDAR calibration remains a Day 5 responsibility.

All public pixel mappings use continuous pixel-edge coordinates. Therefore,
the centre of array element `(row, column)` is `(column + 0.5, row + 0.5)`,
while crop boundaries are `0/width` and `0/height`. Horizontal panorama
coordinates wrap modulo the image width. Vertical coordinates never wrap.

## Initial crop layout

`eight_view_layout()` creates one pitch row with deterministic crop IDs and yaw
centres:

```text
crop IDs:    0   1   2    3    4    5    6    7
yaw (deg):   0  45  90  135  180  225  270  315
pitch (deg): 0   0   0    0    0    0    0    0
```

The initial crop size is `640 x 640`, with a 60-degree horizontal FOV and a
90-degree vertical FOV. Forty-five-degree yaw spacing gives 25 percent
horizontal angular overlap. The useful central 90 degrees are covered; this
does not claim to cover the panorama's complete 120-degree vertical span.
Pitch rows, both FOVs, dimensions, and arbitrary six/eight-view layouts remain
configurable.

`PerspectiveCropGenerator` inverse-maps crop pixel centres to camera rays and
then panorama pixels. It caches the image-independent maps, samples the source
with bilinear interpolation, wraps horizontally, and masks rays outside the
configured vertical span.

## Retained transformations

Each `PerspectiveView` retains an immutable `PerspectiveGeometry`, including
the crop dimensions, yaw, pitch, FOV, and the right-handed orthonormal
`rotation_camera_from_crop` matrix. The following mappings are analytic and
bidirectional:

```text
crop pixel <-> normalized camera ray
panorama pixel <-> normalized camera ray
crop pixel <-> panorama pixel
```

Crop boxes are projected using four corners plus four edge midpoints. The
resulting `PanoramaBox` stores the projected boundary samples and one or two
horizontal intervals. Two intervals represent a box crossing the `0/width`
seam without inflating it to almost the full panorama width. Detection centre
rays are calculated directly from crop geometry, never by averaging wrapped
panorama columns.

## Detector candidate boundary

The bake-off is hard-limited by `TwoCandidateDetectorBenchmark` to one or two
candidates. The only candidates wired for Day 4 are:

- compact YOLOE: `yoloe-11s-seg.pt`, Ultralytics `8.3.162`;
- GroundingDINO-Tiny: `IDEA-Research/grounding-dino-tiny`, Transformers
  `4.53.2`, model revision
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`.

Both adapters accept the same immutable `DetectorClass` prompts and return
only normalized `CropDetection` values. They load once, reuse the model across
all crops, convert aliases back to canonical Day 2 names, and retain the
effective prompt. Detector-specific tensors do not leave the adapters.

`tools/day4_detector_smoke.py` is deliberately prediction-only. It runs one
candidate on the same eight views and saves raw crop boxes, wrap-aware panorama
envelopes, and normalized centre rays. It does not perform NMS, calculate
recall or false-positive metrics, measure latency/VRAM, or select a winner.
Those activities remain pending until a manually verified benchmark set is
assigned.

Model weights are kept in the local development cache rather than committed to
the repository. The YOLOE text encoder and both checkpoints must be packaged
with the final offline image during the later qualification step.

## Focused tests

```bash
pytest -q \
  test/test_panorama_projection.py \
  test/test_crop_generator.py \
  test/test_detector_benchmark.py
```

The tests cover the 120-degree vertical model, horizontal wrap, crop/panorama
round trips, camera-ray normalization, seam-safe box projection, dense yaw
coverage, 25-percent overlap, deterministic crop generation, vertical masks,
adapter-output validation, and the two-candidate ceiling.
