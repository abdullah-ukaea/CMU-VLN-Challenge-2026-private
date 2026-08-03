# Panoramic Perception Geometry

Day 4 introduces a ROS-independent front end for the challenge's cropped
equirectangular camera image. It covers perspective tiling, retained coordinate
transforms, the bounded two-candidate detector bake-off, seam-aware duplicate
merging, and the detector-independent perception worker. It deliberately stops
before camera-LiDAR projection and all 3D object processing.

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
with OpenCV's optimized bilinear remap, wraps horizontally, and masks rays
outside the configured vertical span. A NumPy reference sampler remains as a
dependency-free fallback. The production sampler measured 71 ms median for all
eight views on the development machine.

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

## Detector boundary and selected baseline

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

The measured winner is compact YOLOE at confidence `0.20`, FP16, and input size
`640 x 640`. `perception/baseline.py` contains the frozen Day 4 constants and
worker factory. GroundingDINO remains only as the measured second adapter.

The full threshold and resource results are recorded in
`docs/day_4_detector_decision.md`. `tools/run_day4_detector_benchmark.py`
measures both original and horizontally rolled panoramas, including target,
anchor, rare, and small recall, false positives, cold/warm latency, component
timings, VRAM, seam recall, and seam duplicate excess.

Model weights are not committed to the repository. The Docker build downloads
the selected YOLOE checkpoint and MobileCLIP prompt encoder into
`/home/docker/models`; runtime adapters resolve both there and do not require
internet access. GroundingDINO weights are benchmark-only and are not packaged
in the selected runtime image.

## Seam-aware merging

Every crop detection is projected into a common `PanoramaBox`. Horizontal
support is represented as one interval or two intervals meeting opposite image
edges, so IoU remains meaningful at `u=0/W`.

The merge policy compares only canonical classes. Cross-crop fragments merge
at panorama IoU `0.40` or intersection-over-smaller `0.30`; strict same-crop
suppression uses `0.60/0.85`. The highest-confidence detection supplies the
reported envelope and score. All source crop IDs and boxes are retained,
centre rays are confidence-averaged in 3D, and panorama `u` values are never
averaged linearly across the seam.

The deliberate original/rolled audit at the selected threshold detects three
of four seam-positioned objects and produces zero duplicate excess boxes. The
missed fourth object is a detector miss rather than a wrap or merge failure.

## Perception worker

`PerceptionWorker` owns only this flow:

```text
PerceptionRequest
  -> cached perspective crops
  -> selected detector
  -> crop-to-panorama projection
  -> seam-aware merge
  -> PerceptionResult with raw and clean Detection2D tuples
```

`detector_classes_from_task_specification()` turns the Day 2 parse into a
deduplicated query-conditioned vocabulary. The worker does not perform 3D
lifting, fusion, relations, colours, planning, or keyframe policy.

`tools/day4_perception_integration.py` exercises the selected worker using a
real parsed question and writes only detector-independent JSON.

## Debug outputs

Each benchmark saves crop layout, crop contact sheet, panorama detections
before NMS, panorama detections after NMS, and a JSON trace containing panorama
boxes, centre pixels, normalized camera rays, crop IDs, and merge flags. Both
the original Office 1 panorama and its rolled seam variant are saved.

## Focused tests

```bash
pytest -q \
  test/test_panorama_projection.py \
  test/test_crop_generator.py \
  test/test_detector_benchmark.py \
  test/test_cross_crop_nms.py \
  test/test_detector_dataset.py \
  test/test_detector_metrics.py \
  test/test_perception_worker.py \
  test/test_visualisation.py
```

The tests cover the 120-degree vertical model, horizontal wrap, crop/panorama
round trips, camera-ray normalization, seam-safe box projection, dense yaw
coverage, 25-percent overlap, deterministic crop generation, vertical masks,
adapter-output validation, the two-candidate ceiling, wrap-aware IoU,
duplicate/distinct-object behavior, metric matching, manifest validation,
parsed-task vocabulary, worker output, and debug artifacts.
