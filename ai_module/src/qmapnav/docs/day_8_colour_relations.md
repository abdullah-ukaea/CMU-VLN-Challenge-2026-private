# Day 8 Colour and Support Relations

Day 8 adds uncertainty-aware colour evidence and conservative map-frame
`above`, `below`, `on`, and `supports` relations. These are derived properties
of persistent Day 7 IDs. They do not change detector prompting, object
association, task resolution, planning, or official answer publication.

## Colour vocabulary and data split

The canonical query vocabulary is `black`, `blue`, `brown`, `green`, `grey`,
`orange`, `pink`, `purple`, `red`, `white`, and `yellow`. Query aliases include
`aqua`, `navy`, and `light_blue` as blue, `maroon` as red, `olive` as green,
and `gray` as grey. Raw audit normalization preserves released distinctions so
the audit does not silently rewrite the source data.

The fixed split in `benchmark/day8_colour_split.json` fits on 12 scenes and
holds out `hotel_room_1`, `office_1`, and `office_2`. The audit found 1,967
released objects. Released colour attributes contain no explicit white label.
The persisted white prototype therefore uses only the brightest 5% tail of
fit-scene grey RGB metadata, capped at CIELAB L*=82. This proxy limitation is
recorded in `benchmark/day8_colour_prototypes.json`; white remains distinct at
query and runtime level.

## Observation colour pipeline

Pixel selection follows this strict order:

1. the selected detection's panorama segmentation component, conservatively
   eroded and intersected with valid crop bounds;
2. the Day 6 selected cluster reprojected to that same source crop and dilated;
3. a contracted inner box with boundary weighting and contamination status.

The cluster coordinates are already the selected object's depth layer, so
using its `source_projection_indices` rejects unrelated foreground/background
depth. Shadows and specular highlights are downweighted, while consistently
black and mostly white observations are retained. Low saturation, exposure,
too few pixels, missing crops, and likely background contamination have
explicit statuses.

Weighted medians, robust Lab covariance, circular HSV hue centre/spread,
saturation, value, and lightness percentiles are classified against fitted
prototypes. The classifier uses covariance-aware Lab distance, circular hue,
and a neutral/chromatic branch. It always returns a full normalized
distribution for a valid observation and lowers confidence for small margins,
low effective pixel count, inconsistent pixels, or poor exposure.

Persistent fusion accumulates `weight * probability` with crop, mask,
geometry, exposure, confidence, and pixel-count quality terms. Evidence is
capped at 12 units, each viewpoint is capped at 1.5 units, and history is
bounded at 32 records. Invalid evidence remains visible in history but cannot
overwrite the best estimate or the accumulated distribution.

## Held-out colour result

`benchmark/day8_colour_heldout_report.json` evaluates all 522 colour entries
from the three held-out scenes using released per-object RGB metadata:

- top-1 accuracy: 75.48%;
- top-2 accuracy: 97.13%;
- valid coverage: 100%;
- approximate expected calibration error: 0.1584;
- statuses: 202 good, 68 ambiguous, 84 low-saturation, 168 underexposed.

The visible errors are 102 grey-to-white cases from the documented white proxy,
15 red-to-brown cases, and 11 green-to-orange cases. The stored confusion and
calibration report is the source of truth; these results are not presented as
camera-crop ground truth.

## Common support geometry

`SupportGeometry` adapts an `ObjectInstance`, `PersistentObjectRecord`, or
`StructuralAnchor` into an ID, class, upright footprint, bottom/top Z,
confidence, quality status, and source type. Object Z extents use the robust
fused AABB. XY extents use the upright OBB. Structural supports require a known
extent and use their polygon when available.

`above(A, B)` requires A's centre to be higher, A's bottom to clear B's top
within 0.08 m tolerance, and either footprint overlap or at most 0.50 m edge
separation. `below(A, B)` is evaluated exactly as `above(B, A)`.

`on(A, B)` additionally requires:

- subject-bottom/support-top gap in [-0.08, 0.15] m;
- subject footprint coverage, not ordinary pair IoU, with a 0.50 target;
- nearby candidate generation within a 2.0 m search radius;
- semantic support plausibility and both geometry confidences;
- confidence reduction for sparse, partial, or uncertain geometry.

The acceptance threshold is 0.70 and uncertain hypotheses are retained from
0.40. Semantic classes are a score, never an absolute prohibition. Accepted
`on(A, B)` implies `above(A, B)`, `below(B, A)`, and `supports(B, A)`.

## Relation graph and diagnostics

The relation graph is cleared and rebuilt whenever persistent geometry is
updated, so refitted boxes cannot leave stale edges. Edges are unique by
relation and ordered pair, self-relations are rejected, ambiguous supports stay
ranked, and high-confidence vertical contradictions are reported as possible
geometry or identity problems.

Colour diagnostics save original/selected/rejected panels, shadow/highlight
masks, hue histograms, Lab summaries, stage counts, probabilities, and status.
Relation diagnostics save side and top views plus gap, subject overlap,
geometry confidence, and final confidence. Runtime arrows and labels use the
separate `/qmapnav/debug/relations` topic. `/selected_object_marker` remains
guarded and unchanged.

The generated representative artifacts are retained outside the source tree at
`/home/abdul/cmu-vln/data/day8/diagnostics`.

The representative relation report contains clean book/table, picture/desk,
floating, distant, sparse, and structural shelf cases. It records precision
and recall of 1.0 and false-support rate 0.0 on this small manually specified
geometry set; it is a deterministic regression pack, not a population claim.

## Reproduce

```bash
python3 tools/day8_colour_audit.py /data/vla3d/Unity \
  benchmark/day8_colour_split.json
python3 tools/day8_fit_colour_prototypes.py /data/vla3d/Unity \
  benchmark/day8_colour_split.json
python3 tools/day8_evaluate_colour.py /data/vla3d/Unity \
  benchmark/day8_colour_split.json benchmark/day8_colour_prototypes.json
python3 tools/day8_evaluate_relations.py
pytest -q test/test_colour_*.py test/test_day8_*.py
```
