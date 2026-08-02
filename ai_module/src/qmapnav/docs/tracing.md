# JSON Decision Traces

Q-MapNav records versioned JSON Lines using schema `1.0`. Every record contains
the episode ID and sequence plus:

- elapsed episode time, mission state, parser mode/confidence, and time remaining;
- raw and normalized question text plus ignored-publication count;
- known object and structure counts and unresolved entity IDs;
- candidate action, selected action, and selection reason;
- active route index and direct-republish/recovery counters;
- terminal status and event-specific details.

Parser summaries, question-latch outcomes, executor transitions, periodic scan
map statistics, and clean shutdown are observed by the mission node.

`JsonlDecisionTraceRecorder` performs directory creation, JSON serialization,
and file writes on a daemon worker. Production callbacks only attempt a
non-blocking queue insertion. The default queue holds at most `512` events and
the default file is capped at `4 MiB`; overflow is counted and dropped. JSON
serialization and write failures are contained inside the recorder and cannot
change a control decision. Clean shutdown attempts to flush for at most one
second.

The default output is `/tmp/qmapnav/decision_trace.jsonl`. Each process episode
uses a generated identifier even when the configured file already exists.
