# Panoramic Perception Geometry

perception introduces a ROS-independent front end for the challenge's cropped
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
camera/LiDAR calibration remains a projection responsibility.

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
candidates. The only candidates wired for perception are:

- compact YOLOE: `yoloe-11s-seg.pt`, Ultralytics `8.3.162`;
- GroundingDINO-Tiny: `IDEA-Research/grounding-dino-tiny`, Transformers
  `4.53.2`, model revision
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`.

Both adapters accept the same immutable `DetectorClass` prompts and return
only normalized `CropDetection` values. They load once, reuse the model across
all crops, convert aliases back to canonical protocol names, and retain the
effective prompt. Detector-specific tensors do not leave the adapters.

The measured winner is compact YOLOE at confidence `0.20`, FP16, and input size
`640 x 640`. `perception/baseline.py` contains the frozen perception constants and
worker factory. GroundingDINO remains only as the measured second adapter.

The full threshold and resource results are recorded in
`docs/perception_detector_decision.md`. `tools/run_external_detector_benchmark.py`
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

`detector_classes_from_task_specification()` turns the protocol parse into a
deduplicated query-conditioned vocabulary. The worker does not perform 3D
lifting, fusion, relations, colours, planning, or keyframe policy.

`tools/external_perception_integration.py` exercises the selected worker using a
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


## Measured detector selection

## Selected baseline

Q-MapNav uses **compact YOLOE** with:

- checkpoint: `yoloe-11s-seg.pt`;
- Ultralytics: `8.3.162`;
- input: eight `640 x 640` crops;
- precision: FP16 on CUDA;
- confidence threshold: `0.20`;
- cross-crop IoU threshold: `0.40`;
- cross-crop intersection-over-smaller threshold: `0.30`;
- same-crop IoU/intersection-over-smaller thresholds: `0.60/0.85`.

The frozen constants and worker factory live in `perception/baseline.py`.

## Candidates and test environment

Exactly two candidates were compared:

1. compact YOLOE, `yoloe-11s-seg.pt`;
2. GroundingDINO-Tiny, revision
   `a2bb814dd30d776dcf7e30523b00659f4f141c71`.

The benchmark ran inside the competition AI Docker environment on an NVIDIA
GeForce RTX 4060 Ti with 16 GB VRAM. It used six manually verified live
simulator panoramas from six scenes and six horizontally rolled copies. The
roll changes the artificial panorama seam while preserving the scene. Both
detectors received identical crops, prompts, thresholds, warm-up handling,
and merge policy.

The 59 visible instances include 41 targets, 28 anchors, 11 rare instances,
and 6 small instances. A detection matches at same-class panorama IoU `0.25`.

## Threshold sweep

Recall and FP/panorama below are averages of the original and rolled sets.
Latency is the measured warm median for one complete eight-crop panorama.

| Detector | Threshold | Target | Anchor | Rare | Small | FP/pano | Warm ms | Seam recall | Seam duplicate excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOE | 0.10 | 0.829 | 0.893 | 0.773 | 0.833 | 14.75 | 1379.6 | 4/4 | 1 |
| **YOLOE** | **0.20** | **0.793** | **0.768** | **0.682** | **0.833** | **9.00** | **1230.2** | **3/4** | **0** |
| YOLOE | 0.30 | 0.756 | 0.696 | 0.682 | 0.833 | 6.25 | 1189.6 | 3/4 | 0 |
| YOLOE | 0.40 | 0.683 | 0.554 | 0.591 | 0.667 | 3.67 | 1269.2 | 3/4 | 0 |
| GroundingDINO-T | 0.10 | 0.854 | 0.857 | 0.864 | 0.833 | 45.58 | 3867.9 | 3/4 | 0 |
| GroundingDINO-T | 0.20 | 0.854 | 0.857 | 0.864 | 0.833 | 45.58 | 3819.0 | 3/4 | 0 |
| GroundingDINO-T | 0.30 | 0.854 | 0.821 | 0.864 | 0.833 | 26.00 | 3784.3 | 3/4 | 0 |
| GroundingDINO-T | 0.40 | 0.780 | 0.714 | 0.773 | 0.833 | 14.00 | 3721.1 | 3/4 | 0 |

The table records the fair two-candidate run before the crop sampler was
optimized. The optimized sampler is geometrically regression-tested and is
shared by every detector adapter.

## Resource results

| Metric | YOLOE | GroundingDINO-Tiny |
|---|---:|---:|
| Model load | 293 ms | 6,331 ms |
| First panorama after load | 6,434 ms | 6,128 ms |
| Combined cold load and first panorama | 6,727 ms | 12,459 ms |
| Warm median panorama | 1,230 ms | 3,721 ms |
| Warm p90 panorama | 1,939 ms | 3,891 ms |
| Peak process VRAM | 667 MiB | 1,503 MiB |
| Resident process VRAM after benchmark | 61 MiB | 668 MiB |

With the production OpenCV remap, selected YOLOE at `0.20` measured 71 ms
median crop generation and 653 ms warm median total panorama latency. Its
recall was unchanged, peak process VRAM was 662 MiB, and seam duplicate excess
remained zero.

## Reason

GroundingDINO-Tiny has stronger rare-object recall, but at its practical
`0.40` threshold it has lower target and anchor recall than YOLOE `0.20`, about
56 percent more false positives, roughly three times the warm latency, and
more than twice the peak process VRAM. Lower GroundingDINO thresholds produce
43-48 false positives per panorama.

YOLOE `0.10` improves recall but retains one deliberate seam duplicate and
substantially increases false positives. YOLOE `0.20` is therefore the
strongest measured online baseline that satisfies the seam exit criterion.

## Known weaknesses

- Rare-instance recall averages 0.682; missed examples need query-aware closer
  viewpoints on later days.
- One of four seam-positioned objects is missed at `0.20`; the failure is
  detector recall, not duplicate merging.
- The benchmark is intentionally compact and contains six independent source
  panoramas. Rolled copies test seam invariance but are not new scenes.
- Initial text-prompt embedding is the dominant cold-start cost and must occur
  before time-critical motion when possible.
- Ultralytics runtime licensing must be reviewed before public redistribution.

## Evidence

- selected production report:
  `/home/abdul/cmu-vln/data/perception/results/yoloe_selected_runtime/compact_yoloe.json`;
- fair YOLOE sweep: `/home/abdul/cmu-vln/data/perception/results/yoloe_final/compact_yoloe.json`;
- fair GroundingDINO sweep:
  `/home/abdul/cmu-vln/data/perception/results/grounding_dino_final/grounding_dino_tiny.json`;
- original and rolled crop/NMS visualizations are beside each report;
- benchmark manifest:
  `/home/abdul/cmu-vln/data/perception/detector_manifest.json`.
