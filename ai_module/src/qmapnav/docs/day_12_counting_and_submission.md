# Day 12 Counting And Submission Runbook

Date: 16 August 2026  
Configuration: `configs/submission_v1.yaml`  
Branch tested: `abdullah/Q-map-nav`  
Base revision before Day 12: `e217a69211d706387ef6ad5ac1cfa13e964290e4`

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
update one persistent instance, but cannot create another count unless Day 7
association creates a genuinely distinct persistent ID. Numerical anchor
ambiguity is retained as multiple count hypotheses. Counts are only considered
stable when those hypotheses agree, the qualifying persistent-ID set repeats,
and the evidence comes from the configured number of independent map poses.

Zero is an ordinary answer. It is not used as an unset sentinel. Strong zero
confidence is only added when every target-specific plausible support surface
has strong negative evidence in the Day 11 `SupportSearchHistory`.

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
requested black pillows `grey`. Runtime colour evidence comes from the Day 8
pixel classifier's canonical `red` and `black` distributions, so broadening the
production solver to equate those annotation labels would be an unjustified
scene-proxy special case.

Evidence is stored outside the submission repository at:

```text
data/day12/benchmark/annotated_map_v1/summary.json
data/day12/benchmark/annotated_map_v1/per_case.json
data/day12/benchmark/annotated_map_v1/cases/*/
```

## Regression Evidence

- Pre-Day-12 baseline: 858 passed in 24.40 seconds.
- Final source-mounted suite: 887 passed in 23.76 seconds.
- Final installed-image suite with networking disabled: 887 passed in 21.48
  seconds.
- Day 12 focused and style suite: 31 passed.
- Day 3 oracle quick: 6/6 structural, 3/3 instructions, 18/18 proxy points.
- Day 3 oracle full: 75/75 structural, 30/30 instructions, 180/180 proxy points.
- Day 10 object-reference quick: 6/6 terminal records, markers, protocol, and
  target selections; 12/12 proxy points.
- Day 11 replay: occluded target found; oracle and perceived routes completed
  in order; stage order enforced.

The full saved evidence root is `data/day12/regression/` in the parent project
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
| `day8_colour_prototypes.json` | 8,311 | `f00df4208e88544c209000db3d8c13cd51f13e0f5efc297400eba9d46b094c66` |
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
python3 tools/day12_protocol_audit.py \
  --output /tmp/qmapnav/protocol_audit.json
python3 tools/day12_asset_inventory.py \
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
   python3 tools/day12_offline_smoke.py \
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
   `data/day12` evidence, model weights, or `__pycache__` directories are staged.
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
not made, and Day 12's external exit criterion remains open.
