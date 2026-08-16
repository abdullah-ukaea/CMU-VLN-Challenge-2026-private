"""Compose Day 6 lifting over one immutable Day 5 projection frame."""

from dataclasses import dataclass

import numpy as np

from qmapnav.mapping.ground_filter import estimate_local_ground_plane
from qmapnav.mapping.ground_filter import GroundEstimate
from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import LiftingResult
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.projection_pipeline import ProjectionFrame


@dataclass(frozen=True)
class LiftingFrame:
    """Independent candidates lifted from one panorama observation."""

    image_id: str
    image_timestamp_ns: int
    source: GeometrySource
    ground_estimate: GroundEstimate
    results: tuple[LiftingResult, ...]

    @property
    def candidates(self):
        """Return only successfully fitted single-observation candidates."""
        return tuple(
            result.candidate
            for result in self.results
            if result.candidate is not None
        )


class LiftingPipeline:
    """Lift all detections without persistence, fusion, or final selection."""

    def __init__(
        self,
        lifter: ObjectLifter | None = None,
        *,
        source: GeometrySource = GeometrySource.ACCUMULATED,
        use_masks: bool = False,
    ) -> None:
        self.lifter = lifter or ObjectLifter()
        self.source = source
        self.use_masks = bool(use_masks)

    def process(self, frame: ProjectionFrame) -> LiftingFrame:
        """Estimate ground and lift every detection from one selected cloud."""
        if self.source is GeometrySource.CURRENT:
            projection = frame.current
        elif self.source is GeometrySource.ACCUMULATED:
            projection = frame.accumulated
        else:
            projection = combine_projection_results(
                frame.current,
                frame.accumulated,
            )
        ground = estimate_local_ground_plane(
            frame.accumulated_snapshot.points_xyz,
            timestamp_ns=frame.panorama.timestamp_ns,
            sensor_position_xyz=frame.association.pose.position_xyz,
        )
        ages = None
        if self.source is GeometrySource.ACCUMULATED:
            source_stamps = frame.accumulated_snapshot.last_seen_timestamp_ns[
                projection.source_point_indices
            ]
            ages = (
                frame.panorama.timestamp_ns - source_stamps
            ).astype(np.float64) / 1_000_000_000.0
        elif self.source is GeometrySource.COMBINED:
            accumulated_stamps = frame.accumulated_snapshot.last_seen_timestamp_ns[
                frame.accumulated.source_point_indices
            ]
            accumulated_ages = (
                frame.panorama.timestamp_ns - accumulated_stamps
            ).astype(np.float64) / 1_000_000_000.0
            ages = np.concatenate(
                (np.zeros(frame.current.point_count), accumulated_ages)
            )
        results = tuple(
            self.lifter.lift(
                detection,
                projection,
                source=self.source,
                ground_plane=ground.plane,
                use_mask=self.use_masks,
                scan_age_seconds=ages,
            )
            for detection in frame.detections
        )
        return LiftingFrame(
            image_id=frame.panorama.image_id,
            image_timestamp_ns=frame.panorama.timestamp_ns,
            source=self.source,
            ground_estimate=ground,
            results=results,
        )


def combine_projection_results(
    current: ProjectionResult,
    accumulated: ProjectionResult,
) -> ProjectionResult:
    """Combine current and accumulated support with explicit combined source."""
    if current.image_id != accumulated.image_id or (
        current.image_timestamp_ns != accumulated.image_timestamp_ns
    ):
        raise ValueError('combined projections must represent the same image')
    if not np.allclose(
        current.transform_camera_internal_from_map,
        accumulated.transform_camera_internal_from_map,
        atol=1e-9,
    ):
        raise ValueError('combined projections must use the same camera transform')
    count = current.point_count + accumulated.point_count
    diagnostics = ProjectionDiagnostics(
        input_point_count=(
            current.diagnostics.input_point_count
            + accumulated.diagnostics.input_point_count
        ),
        range_valid_count=(
            current.diagnostics.range_valid_count
            + accumulated.diagnostics.range_valid_count
        ),
        vertical_valid_count=(
            current.diagnostics.vertical_valid_count
            + accumulated.diagnostics.vertical_valid_count
        ),
        projected_point_count=count,
        image_scan_delta_ms=max(
            abs(current.diagnostics.image_scan_delta_ms),
            abs(accumulated.diagnostics.image_scan_delta_ms),
        ),
        pose_mode=current.diagnostics.pose_mode,
        pose_before_delta_ms=current.diagnostics.pose_before_delta_ms,
        pose_after_delta_ms=current.diagnostics.pose_after_delta_ms,
        timing_warning=(
            current.diagnostics.timing_warning
            or accumulated.diagnostics.timing_warning
        ),
    )
    return ProjectionResult(
        image_id=current.image_id,
        image_timestamp_ns=current.image_timestamp_ns,
        scan_timestamp_ns=max(
            current.scan_timestamp_ns,
            accumulated.scan_timestamp_ns,
        ),
        transform_camera_internal_from_map=(
            current.transform_camera_internal_from_map
        ),
        source_valid_mask=np.ones(count, dtype=np.bool_),
        source_point_indices=np.arange(count, dtype=np.int64),
        points_map_xyz=np.vstack(
            (current.points_map_xyz, accumulated.points_map_xyz)
        ),
        points_camera_xyz=np.vstack(
            (current.points_camera_xyz, accumulated.points_camera_xyz)
        ),
        panorama_uv=np.vstack((current.panorama_uv, accumulated.panorama_uv)),
        euclidean_range_m=np.concatenate(
            (current.euclidean_range_m, accumulated.euclidean_range_m)
        ),
        forward_depth_m=np.concatenate(
            (current.forward_depth_m, accumulated.forward_depth_m)
        ),
        intensity=None,
        diagnostics=diagnostics,
    )


__all__ = ['combine_projection_results', 'LiftingPipeline', 'LiftingFrame']
