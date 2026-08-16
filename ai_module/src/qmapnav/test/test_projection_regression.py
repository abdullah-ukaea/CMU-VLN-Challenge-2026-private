"""Tests for source-complete saved Day 5 projection regressions."""

import numpy as np
import pytest

from qmapnav.mapping import AssociationConfig
from qmapnav.mapping import AssociationFailure
from qmapnav.mapping import DAY5_REGRESSION_CATEGORIES
from qmapnav.mapping import DenseRegisteredScanAccumulator
from qmapnav.mapping import DenseScanAccumulatorConfig
from qmapnav.mapping import ProjectionPipeline
from qmapnav.mapping import ProjectionSynchronizer
from qmapnav.mapping import TimedPanorama
from qmapnav.mapping import TimedPose
from qmapnav.mapping import TimedRegisteredScan
from qmapnav.mapping.projection_regression import load_projection_regression_case
from qmapnav.mapping.projection_regression import replay_projection_regression_case
from qmapnav.mapping.projection_regression import save_projection_regression_case
from qmapnav.mapping.projection_regression import verify_projection_regression_checksums
from qmapnav.mapping.transforms import compose_transforms
from qmapnav.mapping.transforms import invert_transform
from qmapnav.mapping.transforms import make_transform
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation
from qmapnav.mapping.transforms import transform_from_pose
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def _frame():
    extrinsic = make_transform(
        quaternion_xyzw_to_rotation(np.array([-0.5, 0.5, -0.5, 0.5])),
        np.array([0.0, 0.0, 0.1]),
    )
    model = PanoramaCameraModel(width=360, height=120)
    pipeline = ProjectionPipeline(
        synchronizer=ProjectionSynchronizer(
            AssociationConfig(max_pose_delta_ns=100, max_scan_delta_ns=100)
        ),
        dense_accumulator=DenseRegisteredScanAccumulator(
            DenseScanAccumulatorConfig(voxel_size_m=0.01)
        ),
        transform_sensor_from_camera_optical=extrinsic,
        panorama_model=model,
    )
    pose = TimedPose(
        timestamp_ns=50,
        parent_frame_id='map',
        child_frame_id='sensor',
        position_xyz=np.array([0.0, 0.0, 0.75]),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    scan = TimedRegisteredScan(
        timestamp_ns=50,
        frame_id='map',
        points_xyz=np.array(
            [
                [1.0, 0.7, 0.85],
                [1.5, -0.5, 1.0],
                [-1.0, 0.3, 0.65],
                [0.2, -1.0, 1.2],
            ]
        ),
        intensity=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    pipeline.add_pose(pose)
    pipeline.add_scan(scan, sensor_origin_xyz=pose.position_xyz)
    panorama = TimedPanorama(
        image_id='saved-image',
        timestamp_ns=50,
        frame_id='camera',
        image_rgb=np.zeros((120, 360, 3), dtype=np.uint8),
    )
    frame = pipeline.process(panorama)
    assert not isinstance(frame, AssociationFailure)
    return pipeline, frame, extrinsic, model


@pytest.mark.parametrize('category', DAY5_REGRESSION_CATEGORIES)
def test_saved_case_replays_and_detects_yaw_sign_regression(
    tmp_path,
    category: str,
) -> None:
    pipeline, frame, extrinsic, model = _frame()
    case = save_projection_regression_case(
        tmp_path / category,
        category=category,
        scene_id='synthetic',
        pose_id='pose_0',
        frame=frame,
        transform_sensor_from_camera_optical=extrinsic,
        panorama_model=model,
        projection_config=pipeline.projection_config,
        overlay_rgb=frame.panorama.image_rgb,
        notes='Synthetic frame-direction regression.',
    )

    exact = replay_projection_regression_case(case)
    wrong_sign = replay_projection_regression_case(
        case,
        u_yaw_sign_override=1,
    )

    assert exact.passed
    assert exact.maximum_pixel_error <= 1e-9
    assert not wrong_sign.passed
    assert wrong_sign.maximum_pixel_error > 10.0
    panorama, projection, manifest = load_projection_regression_case(case)
    assert panorama.shape == (120, 360, 3)
    assert projection.point_count == exact.projected_point_count
    assert manifest['category'] == category
    assert verify_projection_regression_checksums(case)


def test_saved_case_detects_inverted_chain_and_omitted_axis_conversion(
    tmp_path,
) -> None:
    pipeline, frame, extrinsic, model = _frame()
    case = save_projection_regression_case(
        tmp_path / 'mutation',
        category='walls',
        scene_id='synthetic',
        pose_id='pose_0',
        frame=frame,
        transform_sensor_from_camera_optical=extrinsic,
        panorama_model=model,
        projection_config=pipeline.projection_config,
        overlay_rgb=frame.panorama.image_rgb,
        notes='Transform mutation regression.',
    )
    correct_transform = frame.current.transform_camera_internal_from_map
    map_from_sensor = transform_from_pose(
        frame.association.pose.position_xyz,
        frame.association.pose.orientation_xyzw,
    )
    optical_from_map_without_internal_basis = compose_transforms(
        invert_transform(extrinsic),
        invert_transform(map_from_sensor),
    )

    inverted = replay_projection_regression_case(
        case,
        transform_camera_from_map_override=invert_transform(correct_transform),
    )
    omitted_basis = replay_projection_regression_case(
        case,
        transform_camera_from_map_override=(
            optical_from_map_without_internal_basis
        ),
    )

    assert not inverted.passed
    assert not omitted_basis.passed
