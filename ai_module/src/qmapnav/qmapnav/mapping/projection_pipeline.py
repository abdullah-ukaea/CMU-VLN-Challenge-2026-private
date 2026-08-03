"""Pure Day 5 association, current projection, and accumulated projection flow."""

from dataclasses import dataclass

import numpy as np

from qmapnav.mapping.dense_scan_accumulator import DenseRegisteredScanAccumulator
from qmapnav.mapping.dense_scan_accumulator import DenseScanSnapshot
from qmapnav.mapping.lidar_camera_projection import project_association
from qmapnav.mapping.lidar_camera_projection import project_map_points
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.projection_quality import DetectionProjection
from qmapnav.mapping.projection_quality import ProjectionQualityConfig
from qmapnav.mapping.projection_quality import summarize_detections
from qmapnav.mapping.timed_buffers import AssociationFailure
from qmapnav.mapping.timed_buffers import AssociationResult
from qmapnav.mapping.timed_buffers import ProjectionSynchronizer
from qmapnav.mapping.timed_buffers import TimedPanorama
from qmapnav.mapping.timed_buffers import TimedPose
from qmapnav.mapping.timed_buffers import TimedRegisteredScan
from qmapnav.mapping.transforms import transform_from_pose
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.panorama_projection import PanoramaCameraModel


@dataclass(frozen=True)
class ProjectionFrame:
    """Current and accumulated geometry aligned to one panorama keyframe."""

    panorama: TimedPanorama
    association: AssociationResult
    current: ProjectionResult
    accumulated: ProjectionResult
    accumulated_snapshot: DenseScanSnapshot
    detections: tuple[Detection2D, ...]
    current_detection_support: tuple[DetectionProjection, ...]
    accumulated_detection_support: tuple[DetectionProjection, ...]


class Day5ProjectionPipeline:
    """Compose pure synchronization, projection, and dense map snapshots."""

    def __init__(
        self,
        *,
        synchronizer: ProjectionSynchronizer,
        dense_accumulator: DenseRegisteredScanAccumulator,
        transform_sensor_from_camera_optical: np.ndarray,
        panorama_model: PanoramaCameraModel,
        projection_config: ProjectionConfig | None = None,
        quality_config: ProjectionQualityConfig | None = None,
    ) -> None:
        self.synchronizer = synchronizer
        self.dense_accumulator = dense_accumulator
        self.transform_sensor_from_camera_optical = np.asarray(
            transform_sensor_from_camera_optical,
            dtype=np.float64,
        ).copy()
        self.panorama_model = panorama_model
        self.projection_config = projection_config or ProjectionConfig()
        self.quality_config = quality_config or ProjectionQualityConfig()

    def add_pose(self, pose: TimedPose) -> None:
        """Buffer one full map-from-sensor pose."""
        self.synchronizer.poses.add(pose)

    def add_scan(
        self,
        scan: TimedRegisteredScan,
        *,
        sensor_origin_xyz: np.ndarray | None,
    ) -> None:
        """Buffer one current scan and add it to the bounded dense map."""
        self.synchronizer.scans.add(scan)
        self.dense_accumulator.add_scan(
            scan.points_xyz,
            frame_id=scan.frame_id,
            timestamp_ns=scan.timestamp_ns,
            sensor_origin_xyz=sensor_origin_xyz,
        )

    def process(
        self,
        panorama: TimedPanorama,
        detections: tuple[Detection2D, ...] = (),
    ) -> ProjectionFrame | AssociationFailure:
        """Associate and project current plus rolling geometry for a keyframe."""
        association = self.synchronizer.associate(panorama)
        if isinstance(association, AssociationFailure):
            return association
        current = project_association(
            association,
            self.transform_sensor_from_camera_optical,
            self.panorama_model,
            self.projection_config,
        )
        snapshot = self.dense_accumulator.snapshot(panorama.timestamp_ns)
        latest_stamp = (
            int(np.max(snapshot.last_seen_timestamp_ns))
            if snapshot.last_seen_timestamp_ns.size
            else panorama.timestamp_ns
        )
        accumulated = project_map_points(
            points_map_xyz=snapshot.points_xyz,
            transform_camera_internal_from_map=(
                current.transform_camera_internal_from_map
            ),
            panorama_model=self.panorama_model,
            image_id=panorama.image_id,
            image_timestamp_ns=panorama.timestamp_ns,
            scan_timestamp_ns=latest_stamp,
            pose_mode=association.pose_mode,
            pose_before_delta_ns=association.pose_before_delta_ns,
            pose_after_delta_ns=association.pose_after_delta_ns,
            config=self.projection_config,
        )
        return ProjectionFrame(
            panorama=panorama,
            association=association,
            current=current,
            accumulated=accumulated,
            accumulated_snapshot=snapshot,
            detections=tuple(detections),
            current_detection_support=summarize_detections(
                tuple(detections),
                current,
                self.quality_config,
            ),
            accumulated_detection_support=summarize_detections(
                tuple(detections),
                accumulated,
                self.quality_config,
            ),
        )

    def reset(self) -> None:
        """Clear source buffers and dense geometry at an episode boundary."""
        self.synchronizer.clear()
        self.dense_accumulator.reset()

    @staticmethod
    def sensor_origin_from_pose(pose: TimedPose) -> np.ndarray:
        """Return the map-frame sensor origin from a timed pose."""
        return transform_from_pose(
            pose.position_xyz,
            pose.orientation_xyzw,
        )[:3, 3]


__all__ = ['Day5ProjectionPipeline', 'ProjectionFrame']
