"""Tests for pure current/accumulated projection composition."""

import numpy as np

from qmapnav.mapping import AssociationConfig
from qmapnav.mapping import AssociationFailure
from qmapnav.mapping import DenseRegisteredScanAccumulator
from qmapnav.mapping import DenseScanAccumulatorConfig
from qmapnav.mapping import ProjectionPipeline
from qmapnav.mapping import ProjectionSynchronizer
from qmapnav.mapping import TimedPanorama
from qmapnav.mapping import TimedPose
from qmapnav.mapping import TimedRegisteredScan
from qmapnav.mapping.transforms import make_transform
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def _pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        synchronizer=ProjectionSynchronizer(
            AssociationConfig(max_pose_delta_ns=100, max_scan_delta_ns=100)
        ),
        dense_accumulator=DenseRegisteredScanAccumulator(
            DenseScanAccumulatorConfig(voxel_size_m=0.04)
        ),
        transform_sensor_from_camera_optical=make_transform(
            quaternion_xyzw_to_rotation(np.array([-0.5, 0.5, -0.5, 0.5])),
            np.array([0.0, 0.0, 0.1]),
        ),
        panorama_model=PanoramaCameraModel(width=360, height=120),
    )


def _pose(timestamp: int) -> TimedPose:
    return TimedPose(
        timestamp_ns=timestamp,
        parent_frame_id='map',
        child_frame_id='sensor',
        position_xyz=np.array([0.0, 0.0, 0.75]),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )


def _panorama(timestamp: int) -> TimedPanorama:
    return TimedPanorama(
        image_id='image',
        timestamp_ns=timestamp,
        frame_id='camera',
        image_rgb=np.zeros((120, 360, 3), dtype=np.uint8),
    )


def test_pipeline_projects_current_and_accumulated_geometry() -> None:
    pipeline = _pipeline()
    pipeline.add_pose(_pose(0))
    pipeline.add_pose(_pose(100))
    scan = TimedRegisteredScan(
        timestamp_ns=50,
        frame_id='map',
        points_xyz=np.array([[1.0, 0.0, 0.85], [1.01, 0.0, 0.85]]),
    )
    pipeline.add_scan(scan, sensor_origin_xyz=np.array([0.0, 0.0, 0.75]))

    result = pipeline.process(_panorama(50))

    assert not isinstance(result, AssociationFailure)
    assert result.current.point_count == 2
    assert result.accumulated.point_count == 1
    assert result.current.panorama_uv[0, 0] == 180.0
    assert result.accumulated_snapshot.observation_count[0] == 2


def test_pipeline_reports_association_failure_and_resets() -> None:
    pipeline = _pipeline()

    assert isinstance(pipeline.process(_panorama(50)), AssociationFailure)
    pipeline.add_pose(_pose(50))
    pipeline.add_scan(
        TimedRegisteredScan(
            timestamp_ns=50,
            frame_id='map',
            points_xyz=np.array([[1.0, 0.0, 0.85]]),
        ),
        sensor_origin_xyz=np.array([0.0, 0.0, 0.75]),
    )
    pipeline.reset()
    assert pipeline.dense_accumulator.stats().raw_point_count == 0
    assert isinstance(pipeline.process(_panorama(50)), AssociationFailure)
