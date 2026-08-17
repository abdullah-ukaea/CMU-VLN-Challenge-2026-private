# Question And Waypoint Execution

The protocol protocol layer owns episode input and robust single-active-waypoint
execution. Its state and safety decisions are deterministic and independent of
ROS transport.

## Question Latch

`QuestionLatch` accepts the first non-empty question after trimming outer
whitespace. Later publications produce one of two ignored outcomes:

- `duplicate` when the text matches the active question;
- `conflict` when a different question arrives during the active process
  episode.

Neither outcome replaces the active question or invokes the parser again. The
competition restarts the process for each evaluated question, so the latch has
no in-process episode-reset operation.

## Sequential Waypoint Executor

`SequentialWaypointExecutor` owns an immutable copy of the semantic route. It
returns only a goal that the ROS adapter should publish, and never sends the
next semantic waypoint until map-frame XY odometry is at or inside the
configurable arrival radius.

```text
idle -> active -> complete
          |
          +-> recovering -> active
          |
          +-> failed
          +-> cancelled
```

The initial measurement-driven settings are:

| Setting | Default | Meaning |
|---|---:|---|
| Arrival radius | `0.75 m` | Pose-based waypoint arrival |
| Progress epsilon | `0.15 m` | Required decrease in best distance |
| No-progress timeout | `12 s` | Time before one bounded action |
| Direct republish limit | `1` | Republishes per semantic waypoint |
| Safe-offset limit | `1` | Recovery attempts per semantic waypoint |
| Episode deadline | `600 s` | Terminal bound for every non-terminal state |

Distance must improve relative to the best distance seen for the current
target. Small oscillations therefore cannot postpone the watchdog forever.
After one direct republish, the executor asks an injected map policy for a safe
offset. A missing, unknown, occupied, non-finite, or out-of-bounds candidate is
rejected and the route fails; the executor never invents an unchecked offset.
After reaching an accepted offset it retries the interrupted semantic waypoint
without resetting either retry budget.

The mission watchdog also enforces the episode deadline rather than merely
reporting it. Expiry moves idle, active, or recovery execution to `failed`.
When an active route has a known robot pose, expiry replaces the base goal with
that pose as a hold command. Terminal states are never rewritten by a later
deadline tick.

Every start, meaningful progress update, arrival, republish, recovery,
completion, failure, and cancellation creates an immutable `ExecutorEvent`.
Tracing observes those events but does not control state transitions.

## Cancellation

The challenge exposes no cancel topic. baseline established that a newly published
waypoint immediately replaces the base's current goal. Cancellation therefore:

1. clears the executor's active and recovery targets;
2. enters the terminal `cancelled` state;
3. publishes the latest map-frame robot pose as a hold goal when a pose is
   available;
4. emits no further semantic route waypoint.

With no known pose, cancellation remains terminal but deliberately publishes
nothing rather than inventing a hold location. Repeated cancellation and
cancellation after another terminal state are no-ops.

## ROS Adapter

The mission node uses the official interfaces:

- `/challenge_question` (`std_msgs/msg/String`);
- `/state_estimation` (`nav_msgs/msg/Odometry`);
- `/way_point_with_heading` (`geometry_msgs/msg/Pose2D`).

A `0.25 s` timer invokes the pure watchdog. Heading is sent to the base and
retained for pose-hold cancellation, but it does not gate protocol arrival.
