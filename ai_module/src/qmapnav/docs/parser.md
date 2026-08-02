# Deterministic Language Parser

The language subsystem translates a challenge question into the frozen
`TaskSpecification` contract. It does not ground language into detected object
instances, map regions, or waypoint coordinates.

## Public API

- `parse_question_full(question)` accepts only the explicitly supported grammar
  and raises `FullParseError` when a complete specification cannot be built.
- `parse_question_degraded(question)` deterministically returns a partial valid
  specification for any non-empty input.
- `parse_question(question)` tries the full parser first and invokes the
  degraded parser only after `FullParseError`.

Classification and lexical extraction remain available separately through
`classify_task_type` and `extract_language_features`.

## Full Parse

The full parser performs these steps:

1. classify the task as numerical, object reference, or instruction following;
2. extract normalized, span-aware language features;
3. assign a unique sentence-local ID to every entity mention;
4. attach preceding colours, normalized size/shape/material attributes, and
   explicit cardinalities to the relevant entity;
5. bind descriptive spatial relations to subject and anchor entity IDs;
6. distinguish descriptive `near`/`between` relations from route semantics;
7. build ordered route steps and forbidden constraints;
8. bind the terminal target to an entity already in the specification;
9. construct the validated frozen `TaskSpecification`.

Repeated mentions receive distinct parser-local IDs. Coreference beyond the
explicit `it`/`them` support patterns is intentionally deferred rather than
silently merging potentially different objects.

The normalized route actions currently emitted are:

| Language form | Route action |
|---|---|
| `go to` | `go_to` |
| `go near` | `go_near` |
| `take the path between`, `pass between`, `go between` | `pass_between` |
| `take the path near` | `pass_near` |
| `pass by` | `pass_by` |
| `stop at`, `stop by`, `finish at` | `stop_at` |
| `stop near`, `finish near`, `finish beside` | `stop_near` |

When a path clause ends with a destination but has no repeated motion verb, the
terminal entity is appended as a final `go_to` step. Avoidance phrases never
become positive route steps.

## Degraded Parse

The degraded parser uses the same normalized evidence but relaxes binding:

- it infers a likely task family when strict classification fails;
- it retains every recognized entity, colour, size/shape/material attribute,
  and explicit cardinality;
- it skips relations or forbidden constraints that cannot be attached safely;
- for instructions, it uses the last recognized entity as a fallback terminal
  target and emits a minimal terminal step when necessary;
- it reports `parse_mode='degraded'` and a deterministic confidence below the
  full parser's `1.0` confidence.

An unrecognized non-empty statement still returns a contract-valid degraded
object-reference specification, possibly with no entities. Downstream code can
therefore trace missing evidence without confusing it with a parser crash.

## Regression Corpus

The tests contain an exact snapshot of the 75 released questions because the
competition Docker build context contains `ai_module` but not the repository's
top-level `questions` directory. Tests assert full parsing for every released
question and exact route actions and terminal targets for all 30 released
instruction questions.
