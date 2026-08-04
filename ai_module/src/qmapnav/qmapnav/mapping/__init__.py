"""Persistent object, structure, and occupancy map components."""

from qmapnav.mapping.bounding_boxes import AxisAlignedBox
from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.bounding_boxes import canonicalize_box
from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.bounding_boxes import UprightOrientedBox
from qmapnav.mapping.cluster_selection import ClusterSelectionConfig
from qmapnav.mapping.cluster_selection import ClusterSelectionResult
from qmapnav.mapping.dense_scan_accumulator import DenseAccumulationResult
from qmapnav.mapping.dense_scan_accumulator import DenseAccumulationStatus
from qmapnav.mapping.dense_scan_accumulator import DenseAccumulatorStats
from qmapnav.mapping.dense_scan_accumulator import DenseRegisteredScanAccumulator
from qmapnav.mapping.dense_scan_accumulator import DenseScanAccumulatorConfig
from qmapnav.mapping.dense_scan_accumulator import DenseScanSnapshot
from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.geometry_evaluation import GeometryEvaluation
from qmapnav.mapping.geometry_evaluation import ReferenceUprightBox
from qmapnav.mapping.geometry_evaluation import upright_box_iou_3d
from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.lidar_camera_projection import CropProjection
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.lifting_pipeline import combine_projection_results
from qmapnav.mapping.lifting_pipeline import Day6LiftingPipeline
from qmapnav.mapping.lifting_pipeline import LiftingFrame
from qmapnav.mapping.lifting_regression import DAY6_REGRESSION_CATEGORIES
from qmapnav.mapping.lifting_regression import LiftingRegressionMetrics
from qmapnav.mapping.lifting_regression import replay_lifting_regression_case
from qmapnav.mapping.lifting_regression import save_lifting_regression_case
from qmapnav.mapping.lifting_regression import verify_lifting_regression_checksums
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import LiftingResult
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.mapping.orientation_confidence import OrientationConfidenceConfig
from qmapnav.mapping.point_selection import PointSelectionConfig
from qmapnav.mapping.point_selection import PointSelectionResult
from qmapnav.mapping.projection_pipeline import Day5ProjectionPipeline
from qmapnav.mapping.projection_pipeline import ProjectionFrame
from qmapnav.mapping.projection_quality import DetectionProjection
from qmapnav.mapping.projection_quality import ProjectionQualityConfig
from qmapnav.mapping.projection_regression import DAY5_REGRESSION_CATEGORIES
from qmapnav.mapping.projection_regression import load_projection_regression_case
from qmapnav.mapping.projection_regression import ProjectionRegressionMetrics
from qmapnav.mapping.projection_regression import replay_projection_regression_case
from qmapnav.mapping.projection_regression import save_projection_regression_case
from qmapnav.mapping.projection_regression import verify_projection_regression_checksums
from qmapnav.mapping.projection_worker import BoundedProjectionWorker
from qmapnav.mapping.scan_accumulator import AccumulationResult
from qmapnav.mapping.scan_accumulator import AccumulationStatus
from qmapnav.mapping.scan_accumulator import RegisteredScanAccumulator
from qmapnav.mapping.scan_accumulator import ScanAccumulatorConfig
from qmapnav.mapping.scan_accumulator import ScanAccumulatorStats
from qmapnav.mapping.timed_buffers import AssociationConfig
from qmapnav.mapping.timed_buffers import AssociationFailure
from qmapnav.mapping.timed_buffers import AssociationResult
from qmapnav.mapping.timed_buffers import ProjectionSynchronizer
from qmapnav.mapping.timed_buffers import TimedPanorama
from qmapnav.mapping.timed_buffers import TimedPose
from qmapnav.mapping.timed_buffers import TimedRegisteredScan


__all__ = [
    'AccumulationResult',
    'AccumulationStatus',
    'AxisAlignedBox',
    'AssociationConfig',
    'AssociationFailure',
    'AssociationResult',
    'BoundedProjectionWorker',
    'BoxEstimationConfig',
    'canonicalize_box',
    'ClusterSelectionConfig',
    'ClusterSelectionResult',
    'combine_projection_results',
    'ConfidenceComponents',
    'CropProjection',
    'DenseAccumulationResult',
    'DenseAccumulationStatus',
    'DenseAccumulatorStats',
    'DenseRegisteredScanAccumulator',
    'DenseScanAccumulatorConfig',
    'DenseScanSnapshot',
    'Day5ProjectionPipeline',
    'Day6LiftingPipeline',
    'DAY5_REGRESSION_CATEGORIES',
    'DAY6_REGRESSION_CATEGORIES',
    'DetectionProjection',
    'DepthFilterConfig',
    'estimate_upright_obb',
    'GeometryEvaluation',
    'GeometrySource',
    'GeometryStatus',
    'GroundPlane',
    'LiftingCounts',
    'LiftingFrame',
    'LiftingResult',
    'LiftingRegressionMetrics',
    'ObjectCandidate3D',
    'ObjectLifter',
    'ObjectLiftingConfig',
    'OrientationConfidenceConfig',
    'PointSelectionConfig',
    'PointSelectionResult',
    'ProjectionConfig',
    'ProjectionDiagnostics',
    'ProjectionFrame',
    'ProjectionQualityConfig',
    'ProjectionRegressionMetrics',
    'ProjectionResult',
    'ProjectionSynchronizer',
    'ReferenceUprightBox',
    'RegisteredScanAccumulator',
    'load_projection_regression_case',
    'replay_projection_regression_case',
    'replay_lifting_regression_case',
    'save_lifting_regression_case',
    'save_projection_regression_case',
    'ScanAccumulatorConfig',
    'ScanAccumulatorStats',
    'TimedPanorama',
    'TimedPose',
    'TimedRegisteredScan',
    'UprightOrientedBox',
    'upright_box_iou_3d',
    'verify_lifting_regression_checksums',
    'verify_projection_regression_checksums',
]
