# Question And Waypoint Protocol Primitives

The Day 2 protocol layer keeps episode input and basic route execution
deterministic and independent from perception or semantic planning.

## Question Latch

`QuestionLatch` accepts the first non-empty question after trimming outer
whitespace. Later publications produce one of two ignored outcomes:

- `duplicate` when the text matches the active question;
- `conflict` when a different question arrives during the active process
  episode.

Neither outcome replaces the active question or reparses it. The ROS node
subscribes to `/challenge_question` with `std_msgs/msg/String` and invokes the
deterministic parser exactly once for the accepted question.

The competition restarts the process for each evaluated question, so this
initial latch deliberately has no in-process episode-reset operation.

## Sequential Waypoint Executor

`SequentialWaypointExecutor` owns an immutable copy of an ordered route of
`Waypoint2D` values. It is ROS-independent and has three basic states:

```text
idle -> active -> complete
```

Starting a route returns only waypoint zero. The ROS adapter publishes that goal
as `geometry_msgs/msg/Pose2D` on `/way_point_with_heading`. Each
`nav_msgs/msg/Odometry` update from `/state_estimation` supplies the robot's map
XY position to the executor.

The executor computes planar distance to the one active goal. At or inside the
provisional `0.75 m` arrival radius, it activates and returns the next waypoint.
Arrival at the last waypoint moves the route to `complete` without emitting
another goal. Heading is transported to the base but is not part of this year's
arrival gate.

An active route cannot be replaced. This protects the measured Day 1 protocol:
the base itself replaces a goal immediately and does not maintain a waypoint
queue, so Q-MapNav must be the sole owner of route ordering.

## Current Boundary

This implementation intentionally does not yet include progress monitoring,
no-progress timeouts, republishing, safe-offset recovery, cancellation, or a
mission deadline. Those remain separate requested tasks and will extend the
executor state machine without changing the current one-goal and pose-arrival
contracts.
