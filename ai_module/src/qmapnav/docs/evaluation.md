# Oracle Evaluation And Regression Harness

The evaluation evaluator is an offline, ROS-independent development tool. It checks
Q-MapNav's parser, perfect-object reasoning, and semantic route planner without
using camera detections or controlling the robot. Its score is a proxy for
development regression, not a reproduction of the hidden challenge evaluator.

## Metrics

The pure metric functions in `qmapnav.evaluation.metrics` expose the raw result
behind every aggregate:

- object selection uses exact normalized object-ID equality;
- numerical answers use exact match and absolute count error;
- relation geometry reports true/false positives and negatives, precision,
  recall, and F1 for `above`, `below`, `between`, `inside`, `near`, and `on`;
- required semantic regions report satisfaction and first trajectory hit;
- required regions retain their global first-hit positions, while ordered
  completion uses a monotonic state machine. Successive overlapping regions
  may complete at the same path position after the semantic state advances;
- forbidden regions report violations and approximate distance travelled
  inside;
- terminal distance is measured to the usable terminal region, not merely an
  object's centre;
- timing records parsing, reasoning, planning, optional execution, and total
  episode completion time.

Instruction tasks use a transparent six-point diagnostic proxy:

```text
3 * fraction of required regions intersected
+ 1 if required regions are completed in order
+ 1 if no forbidden region is entered
+ 1 if the terminal region is reached
```

Raw metrics are always saved, so this weighting can change without losing
historical evidence.

## Answer Provenance

The released answer PDFs are visualizations. They do not provide structured
object IDs or counts. The harness therefore never compares an oracle result
with itself and never fabricates an expected answer.

When a checked answer JSON is supplied to the existing dataset loader,
numerical and object-reference questions receive ordinary scored metrics.
Otherwise their predictions and candidate traces are retained while the score
is `null` and the diagnostic category is `ANSWER_MAPPING_MISSING`. Instruction
semantics remain measurable because the parser and semantic regions define the
required, forbidden, ordered, and terminal conditions explicitly.

## Regression Modes

From the repository root, run:

```bash
python -m qmapnav.evaluation.benchmark_runner --mode quick
python -m qmapnav.evaluation.benchmark_runner --mode full
```

After installation in a ROS workspace, the equivalent command is:

```bash
ros2 run qmapnav qmapnav_benchmark --mode quick
ros2 run qmapnav qmapnav_benchmark --mode full
```

The quick mode runs six fixed released questions: one numerical, two object
references, and three instructions. They cover colour, structural anchors,
closest ranking, between gates, ordered subgoals, and avoidance. Relation
diagnostics use up to 25 ground-truth positives plus deterministic negative
samples per supported relation.

Full mode runs all 75 released questions and all supported relation positives.
Both modes require local development data. Paths default to the repository's
`questions/`, workspace `simulation/`, and workspace `data/vla3d/` locations.
They can be overridden with command-line options or `QMAPNAV_SIMULATION_ROOT`
and `QMAPNAV_VLA_ROOT`.

## Reports And Exit Status

Reports default to `reports/oracle/<mode>/latest/` and contain:

```text
summary.json
per_question.json
relation_metrics.json
failures.json
routes/<instruction-question-id>.json
```

Writes are atomic. Reports distinguish structural execution failures from
unavailable labels and ambiguity warnings. A run returns a non-zero process
status only when a question fails to parse, resolve, or plan its required
structure; missing external answer labels do not make an otherwise valid
regression run fail.

Failure categories are stable, specific strings such as `PARSE_FAILURE`,
`ENTITY_NOT_FOUND`, `AMBIGUOUS_REFERENCE`, `INVALID_GATE`, `PATH_NOT_FOUND`,
`CONSTRAINT_ORDER_FAILURE`, `FORBIDDEN_REGION_VIOLATION`, and
`TERMINAL_GOAL_MISSED`.


## Object-reference benchmark

Date: 4 August 2026
Status: complete
Frozen run: `full_annotated_map_frozen`

## Outcome

All 30 released object-reference questions reached a terminal record and logged
a response. All 30 produced one protocol-valid map-frame marker. The annotated
development proxy selected 24/30 targets and passed the marker proxy for 23/30,
for 46/60 proxy points. Mean controlled episode time was 0.440 seconds.

The frozen run is in
`/home/abdul/cmu-vln/data/object_reference/benchmark/full_annotated_map_frozen`. It contains the exact manifest,
run metadata, per-case records, parser audit, scene/tag aggregates, diagnostic
evidence, fix ranking, and human-readable report.

These correctness labels are development proxies derived from the same released
VLA3D annotations used to construct the scene map. The controlled run injects
those boxes through the production persistent `ObjectMap`; it therefore verifies
parsing, fusion, joint relation solving, ranking, OBB adaptation, marker protocol,
failure classification, and bounded logging, but is not an independent estimate
of detector or LiDAR accuracy.

## Baseline And Final Comparison

| Metric | Baseline | Frozen | Change |
| --- | ---: | ---: | ---: |
| Terminal logged responses | 30/30 | 30/30 | 0 |
| Protocol-valid markers | 28/30 | 30/30 | +2 |
| Proxy-correct target selections | 22/30 | 24/30 | +2 |
| Proxy marker successes | 21/28 | 23/30 | +2 |
| Proxy score | 42/60 | 46/60 | +4 |

Two bounded fixes were made from baseline evidence:

- canonical class aliases now cover `desk_light`/`lamp`,
  `flowers`/`flower`, `nightstand`/`night_stand`, and
  `zen_stone_decoration`/`stone_decoration`;
- repeated textual anchor roles may bind the same physical persistent instance,
  while true self-relations still reject an identical participant.

Neither fix contains a scene coordinate, released object ID, trajectory, or
question-specific answer.

## Pipeline Verification

- Parser: 30/30 used the full deterministic parse. The audit records the target,
  attributes, anchors, relation arguments, and absence of instruction steps.
- Target and anchors: all requested classes were available in the frozen
  annotated control after canonicalisation. The real Office 1 smoke detected two
  potted-plant candidates and one file-cabinet anchor.
- Lifting: every controlled object has explicit stage counts. The perceived smoke
  saved projected-point counts, post-ground/depth/cluster counts, statuses, and
  projection images from the real camera/LiDAR path.
- Fusion: every case records association events and persistent IDs. In the live
  smoke, the file cabinet and two plant candidates retained their IDs across
  three observations before commitment.
- Relations and ranking: every case stores all ranked target hypotheses, score
  decompositions, margins, satisfied/violated constraints, and unresolved terms.
- OBB: the final adapter snapshots the selected persistent instance, canonicalises
  dimensions, applies the low-confidence yaw fallback, and validates finite map
  geometry before publication.
- Protocol and timing: 30/30 frozen markers passed frame, type, action,
  quaternion, and dimension checks and were published exactly once after commit.
  Matching waypoints use the same centre.
- Targeted viewpoint: low target evidence, a missing anchor, a low ranking margin,
  weak geometry, or likely occlusion can request one safe viewpoint. A hard guard
  prevents a second request. The real smoke triggered one lateral viewpoint for
  a 0.016 margin, reranked once, and committed at 24.65 seconds.
- Failure attribution: the deterministic decision tree covers parsing, missed
  target, missed anchor, incorrect colour, bad lifting, duplicate instance, bad
  relation, incorrect OBB, and protocol failure, always choosing the earliest
  causal stage.

## Frozen Failure Counts

| Primary category | Count |
| --- | ---: |
| Parsing | 0 |
| Missed target | 0 |
| Missed anchor | 0 |
| Incorrect colour | 0 |
| Bad lifting | 0 |
| Duplicate instance | 0 |
| Bad relation | 6 |
| Incorrect OBB | 1 |
| Protocol failure | 0 |

The six relation disagreements are in Arabic Room 02, Chinese Room 02,
Home Building 2 02, Hotel Room 1 02, Living Room 2 02, and Living Room 3 01.
Hotel Room 2 02 is the OBB proxy failure: the selected picture and centre are
correct, but the released 8 mm thickness conflicts with the conservative 50 mm
minimum marker thickness, producing 0.161 OBB IoU.

## Three Largest Failure Sources

1. Joint relation/ranking disagreement affected six frozen cases. The complete
   inputs exist, but production scoring disagrees with the annotation-derived
   proxy target.
2. Query-to-map vocabulary and repeated-anchor coreference affected three
   baseline cases. The bounded aliases/coreference fix recovered the missing
   candidates and remains a regression risk even though its frozen count is zero.
3. Thin-object OBB conservatism affects one case. It is a proxy/scoring-adapter
   trade-off rather than a target-selection failure.

## Fix Priority

Priority is expected recovered score divided by implementation effort and risk.

| Rank | Source | Affected | Confidence | Effort | Priority | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Vocabulary/coreference | 3 baseline | 0.90 | 0.5 | 10.80 | Implemented and retained |
| 2 | Bad relation | 6 frozen | 0.70 | 2.0 | 4.20 | Next measured tuning target |
| 3 | Thin-object OBB | 1 frozen | 0.80 | 1.5 | 1.07 | Defer pending scoring clarification |

The relation fix is not applied on object-reference because threshold tuning against only
these released answers would risk benchmark overfitting. The OBB minimum is not
lowered because an 8 mm box is fragile under perceived geometry and the official
IoU convention remains unverified.

## Perceived Smoke

`office_1_object_reference_01` ran against the actual Unity camera, LiDAR, YOLOE
perception baseline, projection/lifting worker, persistent maps, solver, targeted
viewpoint, and ROS publishers. It produced a full parse, three 3D candidates,
three persistent instances, one targeted viewpoint, one final map-frame marker,
one matching waypoint, no marker validation error, and a terminal record in
24.65 seconds. Because no independent selected-instance label is available for
this runtime episode, its semantic correctness remains explicitly `null`.

The detector was the frozen `yoloe-11s-seg.pt` baseline with confidence 0.20,
eight perspective crops, and cross-crop IoU 0.40. The source began from Git commit
`878dc3a2f0c565222c76a4065e1da0f2822e24cf` plus the documented uncommitted object-reference
working-tree changes.

## Reproduction

The installed console entry is `qmapnav_object_benchmark`. Required inputs are
the released `questions.json`, all 15 extracted simulation scenes, and the local
VLA3D `Unity` metadata directory. Use `--mode quick` for the six-case regression
subset or `--mode full` for all 30 cases. A run ID is immutable unless the runner
is explicitly invoked with its resume mode.

The principal artifacts are:

- `manifest.json` and `run_metadata.json`;
- `parser_audit.json` and `grouped_metrics.json`;
- `summary.json`, `per_case.json`, `fix_priorities.json`, and `report.md`;
- one evidence directory per case containing the task specification, lifting and
  fusion diagnostics, complete ranking, marker validation, trace, and result.

The final validation gate is the full package test suite, including lint and
docstring checks. No instruction exploration or instruction-following work is part of
this implementation.


## Counting and submission contracts

Date: 16 August 2026
Configuration: `configs/submission_v1.yaml`
Branch tested: `abdullah/Q-map-nav`
Base revision before counting: `e217a69211d706387ef6ad5ac1cfa13e964290e4`

## Delivered Runtime Behaviour

Numerical tasks now use this bounded path:

```text
latched numerical question
-> deterministic TaskSpecification
-> persistent ObjectMap and StructuralMap candidates
-> complete class, colour, relation, and anchor hypotheses
-> definite/probable/rejected/unresolved persistent-ID partitions
-> count and ID-set stability over independent viewpoints
-> stable answer or deadline-triggered best available answer
-> exactly one std_msgs/msg/Int32 on /numerical_response
```

The solver never consumes or counts raw detections. A repeated observation can
update one persistent instance, but cannot create another count unless mapping
association creates a genuinely distinct persistent ID. Numerical anchor
ambiguity is retained as multiple count hypotheses. Counts are only considered
stable when those hypotheses agree, the qualifying persistent-ID set repeats,
and the evidence comes from the configured number of independent map poses.

Zero is an ordinary answer. It is not used as an unset sentinel. Strong zero
confidence is only added when every target-specific plausible support surface
has strong negative evidence in the instruction `SupportSearchHistory`.

## Official Protocol

The audited interfaces are:

| Direction | Topic | Type | Frame/commit policy |
|---|---|---|---|
| input | `/challenge_question` | `std_msgs/msg/String` | first valid question latched |
| input | `/state_estimation` | `nav_msgs/msg/Odometry` | `map` parent, `sensor` child |
| input | `/registered_scan` | `sensor_msgs/msg/PointCloud2` | accumulated in `map` |
| input | `/camera/image` | `sensor_msgs/msg/Image` | query-conditioned perception |
| output | `/way_point_with_heading` | `geometry_msgs/msg/Pose2D` | one active goal at a time |
| output | `/selected_object_marker` | `visualization_msgs/msg/Marker` | one final marker commitment |
| output | `/numerical_response` | `std_msgs/msg/Int32` | one final integer commitment |

The numerical reserve is 30 seconds. If the answer is not stable by then, the
strongest observed result is committed rather than allowing silence. The
600-second episode watchdog independently forces the same bounded fallback.

## Numerical Benchmark

The annotated-map control ran every released numerical question through the
production parser and persistent-map numerical solver. It deliberately bypasses
the detector and LiDAR lifting, so it tests reasoning and identity counting—not
perception recall.

| Scene | Predicted | Expected | Stable | Failure category |
|---|---:|---:|---|---|
| `arabic_room` | 3 | 3 | no | `unstable_count` |
| `chinese_room` | 6 | 6 | yes | none |
| `home_building_1` | 11 | 11 | no | `unstable_count` |
| `home_building_2` | 0 | 3 | no | `incorrect_colour` |
| `hotel_room_1` | 4 | 4 | yes | none |
| `hotel_room_2` | 3 | 3 | yes | none |
| `japanese_room` | 3 | 3 | yes | none |
| `livingroom_1` | 8 | 8 | yes | none |
| `livingroom_2` | 1 | 1 | yes | none |
| `livingroom_3` | 2 | 2 | yes | none |
| `livingroom_4` | 6 | 6 | no | `unstable_count` |
| `loft` | 0 | 2 | no | `incorrect_colour` |
| `office_1` | 6 | 6 | no | `unstable_count` |
| `office_2` | 1 | 1 | no | `unstable_count` |
| `studio` | 3 | 3 | yes | none |

Result: 15/15 terminal responses, 13/15 exact counts, and 8/15 strict stable
counts. The two incorrect proxy cases are annotation-domain colour mismatches:
released VLA-3D annotations call the requested red pillows `maroon` and the
requested black pillows `grey`. Runtime colour evidence comes from the colour
pixel classifier's canonical `red` and `black` distributions, so broadening the
production solver to equate those annotation labels would be an unjustified
scene-proxy special case.

Evidence is stored outside the submission repository at:

```text
/home/abdul/cmu-vln/data/counting/benchmark/annotated_map_v1/summary.json
/home/abdul/cmu-vln/data/counting/benchmark/annotated_map_v1/per_case.json
/home/abdul/cmu-vln/data/counting/benchmark/annotated_map_v1/cases/*/
```

## Regression Evidence

- Pre-Day-12 baseline: 858 passed in 24.40 seconds.
- Final source-mounted suite: 887 passed in 23.76 seconds.
- Final installed-image suite with networking disabled: 887 passed in 21.48
  seconds.
- counting focused and style suite: 31 passed.
- evaluation oracle quick: 6/6 structural, 3/3 instructions, 18/18 proxy points.
- evaluation oracle full: 75/75 structural, 30/30 instructions, 180/180 proxy points.
- object-reference object-reference quick: 6/6 terminal records, markers, protocol, and
  target selections; 12/12 proxy points.
- instruction replay: occluded target found; oracle and perceived routes completed
  in order; stage order enforced.

The full saved evidence root is `/home/abdul/cmu-vln/data/counting/regression/` in the parent project
workspace. It is intentionally not copied into `ai_module/` or submitted.

## Clean Image And Offline Evidence

The image was built once with no cache using:

```bash
cd /home/abdul/cmu-vln/CMU-VLN-Challenge-2026-private
docker compose -f docker/compose_gpu.yml build --no-cache ai_module
```

BuildKit step time was approximately 727 seconds. The final image, rebuilt after
the bounded shutdown fix while reusing those clean dependency/model layers, is:

```text
tag: docker-ai_module:latest
digest: sha256:571a1e11c6adb7b182f4caaed1a1cf852d5ba30180f979ca7bb56f6d5a990636
size: 7,194,837,842 bytes
```

Pinned runtime versions:

```text
Python 3.12.3
torch 2.7.1+cu128
torchvision 0.22.1+cu128
ultralytics 8.3.162
transformers 4.53.2
opencv-python 4.10.0.84
```

The network-disabled smoke loaded both model assets, parsed a numerical query,
ran all eight YOLOE panorama crops on the RTX 4060 Ti, and published exactly one
correct zero for a synthetic empty scene. Measured warm-filesystem cold start to
answer was 7.28 seconds: 0.28 seconds parse/import, 2.78 seconds model load, 4.17
seconds detector, and 0.04 seconds answer construction/publication. Its complete
log contains no attempted HTTP, Hugging Face, GitHub, pip, or download call.

Required packaged assets:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `yoloe-11s-seg.pt` | 27,803,986 | `8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3` |
| `mobileclip_blt.ts` | 599,764,649 | `a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54` |
| `colour_prototypes.json` | 8,311 | `f00df4208e88544c209000db3d8c13cd51f13e0f5efc297400eba9d46b094c66` |
| `submission_v1.yaml` | 6,492 | `a8c9ab51b3bafa05073228333e5790069fe91ee11eb5b59ee3c19d209c64c8e6` |

All four exist in the clean image. The model checkpoint, MobileCLIP TorchScript,
and colour JSON each passed an offline load check.

## Timeout Audit

| Subsystem | Bound |
|---|---:|
| episode watchdog / question wait | 600 s |
| numerical verification | 180 s |
| numerical final-response reserve | 30 s |
| object-reference final reserve | 30 s |
| waypoint no-progress | 12 s |
| targeted viewpoint minimum remaining time | 45 s |
| projection association buffer | 5 s |
| projection worker shutdown | 2 s |
| trace flush | 1 s |
| watchdog tick | 0.25 s |

Detector work runs on a bounded queue of two frames. It does not block the ROS
watchdog thread, so the numerical reserve and episode watchdog can publish from
the current persistent map even if a detector call is still completing.

## Reproduction From A Clean Checkout

1. Clone the repository and check out the submitted revision. Do not copy any
   `data/day*`, VLA-3D annotations, simulator ground truth, or developer trace
   files into `ai_module/`.
2. On Windows/WSL2, start Docker Desktop, enable WSL integration and host
   networking, and verify `docker version` and `nvidia-smi`.
3. Obtain the released Unity scene and place it as described in the challenge
   README. Development-only VLA-3D metadata is not needed at evaluation runtime.
4. Build the AI image from the repository root:

   ```bash
   docker compose -f docker/compose_gpu.yml build ai_module
   ```

5. Start the system and AI containers using the challenge GPU Compose file plus
   the machine-specific scene override:

   ```bash
   docker compose \
     -f docker/compose_gpu.yml \
     -f /home/abdul/cmu-vln/compose.office1.yml \
     up -d
   ```

6. In the system container, launch base autonomy and Unity:

   ```bash
   docker exec -it iros2026_system bash
   /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
   ```

7. In the AI container, launch the installed package. The launch file loads
   `submission_v1.yaml` automatically:

   ```bash
   docker exec -it iros2026_ai_module bash
   source /opt/ros/jazzy/setup.bash
   source /home/docker/ai_module/install/setup.bash
   ros2 launch qmapnav qmapnav.launch.py
   ```

8. Verify interfaces from another container shell:

   ```bash
   ros2 topic info /challenge_question
   ros2 topic info /way_point_with_heading
   ros2 topic info /selected_object_marker
   ros2 topic info /numerical_response
   ros2 param get /qmapnav detector_checkpoint
   ros2 param get /qmapnav episode_time_limit
   ```

9. For a development smoke, publish one question once. The real evaluator
   publishes it at 1 Hz and the latch ignores repeats:

   ```bash
   ros2 topic pub --once /challenge_question std_msgs/msg/String \
     "data: 'How many cups are on the coffee table?'"
   ```

WSL2 GPU containers also require the already-documented `/usr/lib/wsl` mount and
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`. The local Compose override supplies these.
On a native Linux evaluator, the NVIDIA container runtime supplies the driver
libraries directly.

## Validation Commands

From `ai_module/src/qmapnav` inside the AI image:

```bash
pytest -q
python3 tools/tools/protocol_audit.py \
  --output /tmp/qmapnav/protocol_audit.json
python3 tools/tools/submission_assets.py \
  --output /tmp/qmapnav/asset_inventory.json
```

The explicit network-disabled detector smoke is:

```bash
docker run --rm --network none --gpus all \
  -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  docker-ai_module:latest bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /home/docker/ai_module/install/setup.bash && \
   cd /home/docker/ai_module/src/qmapnav && \
   python3 tools/tools/offline_smoke.py \
     --output /tmp/qmapnav/offline_smoke.json --device cuda:0'
```

No API key, internet connection, or secret environment variable is required for
baseline operation. `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are defensive
smoke-test flags, not runtime secrets or mandatory configuration.

## Manual Submission Checklist

Repository publication is intentionally owner-controlled. Codex must never run
`git add`, `git commit`, or `git push` for this project.

The repository owner must:

1. Inspect `git status`, `git diff`, and every untracked path.
2. Confirm that only `ai_module/` differs from the upstream challenge tree.
3. Confirm no credentials, released answers, VLA-3D metadata, generated
   `data/counting` evidence, model weights, or `__pycache__` directories are staged.
4. Stage and commit the intended `ai_module/` changes manually.
5. Push the tested branch manually.
6. Make the submitted GitHub repository public. An anonymous request must return
   HTTP 200; it returned HTTP 404 before publication on 16 August 2026.
7. If publishing a replacement Docker image, use the final image built from the
   same committed revision and record its immutable digest.
8. Fill the competition Google Form linked from the challenge README with the
   public GitHub repository URL.
9. Record the final commit, public URL, form timestamp, and confirmation in
   `docs/submissions/submission_01.md`.

Until steps 4–9 are completed by the owner, the first submission is prepared but
not made, and counting's external exit criterion remains open.
