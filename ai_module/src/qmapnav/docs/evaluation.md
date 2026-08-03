# Oracle Evaluation And Regression Harness

The Day 3 evaluator is an offline, ROS-independent development tool. It checks
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
