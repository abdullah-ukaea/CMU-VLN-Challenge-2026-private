# Day 4 Detector Decision

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
  `data/day4/results/yoloe_selected_runtime/compact_yoloe.json`;
- fair YOLOE sweep: `data/day4/results/yoloe_final/compact_yoloe.json`;
- fair GroundingDINO sweep:
  `data/day4/results/grounding_dino_final/grounding_dino_tiny.json`;
- original and rolled crop/NMS visualizations are beside each report;
- benchmark manifest:
  `src/qmapnav/benchmark/day4_detector_manifest.json`.
