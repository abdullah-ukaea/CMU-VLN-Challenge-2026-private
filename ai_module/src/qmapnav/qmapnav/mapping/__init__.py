"""Persistent object, structure, and occupancy map components."""

from qmapnav.mapping.bounding_boxes import AxisAlignedBox
from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.bounding_boxes import canonicalize_box
from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.bounding_boxes import UprightOrientedBox
from qmapnav.mapping.box_overlap import GeometryEvaluation
from qmapnav.mapping.box_overlap import ReferenceUprightBox
from qmapnav.mapping.box_overlap import upright_box_iou_3d
from qmapnav.mapping.cluster_selection import ClusterSelectionConfig
from qmapnav.mapping.cluster_selection import ClusterSelectionResult
from qmapnav.mapping.dense_scan_accumulator import DenseAccumulationResult
from qmapnav.mapping.dense_scan_accumulator import DenseAccumulationStatus
from qmapnav.mapping.dense_scan_accumulator import DenseAccumulatorStats
from qmapnav.mapping.dense_scan_accumulator import DenseRegisteredScanAccumulator
from qmapnav.mapping.dense_scan_accumulator import DenseScanAccumulatorConfig
from qmapnav.mapping.dense_scan_accumulator import DenseScanSnapshot
from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.lidar_camera_projection import CropProjection
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.lifting_pipeline import combine_projection_results
from qmapnav.mapping.lifting_pipeline import LiftingFrame
from qmapnav.mapping.lifting_pipeline import LiftingPipeline
from qmapnav.mapping.object_association import AssociationDecision
from qmapnav.mapping.object_association import AssociationScore
from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.mapping.object_association import class_compatibility
from qmapnav.mapping.object_association import score_candidate_instance
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import LiftingResult
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.mapping.object_map import ObjectAssociationEvent
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.object_map import ObjectMapConfig
from qmapnav.mapping.object_map import PersistentObjectRecord
from qmapnav.mapping.orientation_confidence import OrientationConfidenceConfig
from qmapnav.mapping.point_selection import PointSelectionConfig
from qmapnav.mapping.point_selection import PointSelectionResult
from qmapnav.mapping.projection_pipeline import ProjectionFrame
from qmapnav.mapping.projection_pipeline import ProjectionPipeline
from qmapnav.mapping.projection_quality import DetectionProjection
from qmapnav.mapping.projection_quality import ProjectionQualityConfig
from qmapnav.mapping.projection_worker import BoundedProjectionWorker
from qmapnav.mapping.ray_wall_intersection import intersect_ray_with_wall
from qmapnav.mapping.ray_wall_intersection import RayWallIntersection
from qmapnav.mapping.ray_wall_intersection import transform_camera_ray_to_map
from qmapnav.mapping.scan_accumulator import AccumulationResult
from qmapnav.mapping.scan_accumulator import AccumulationStatus
from qmapnav.mapping.scan_accumulator import RegisteredScanAccumulator
from qmapnav.mapping.scan_accumulator import ScanAccumulatorConfig
from qmapnav.mapping.scan_accumulator import ScanAccumulatorStats
from qmapnav.mapping.structural_map import StructuralAnchor
from qmapnav.mapping.structural_map import StructuralAssociationEvent
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.structural_map import StructuralMapConfig
from qmapnav.mapping.structural_map import StructuralRecord
from qmapnav.mapping.timed_buffers import AssociationConfig
from qmapnav.mapping.timed_buffers import AssociationFailure
from qmapnav.mapping.timed_buffers import AssociationResult
from qmapnav.mapping.timed_buffers import ProjectionSynchronizer
from qmapnav.mapping.timed_buffers import TimedPanorama
from qmapnav.mapping.timed_buffers import TimedPose
from qmapnav.mapping.timed_buffers import TimedRegisteredScan
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.mapping.wall_extraction import extract_wall_candidates
from qmapnav.mapping.wall_extraction import merge_wall_candidates
from qmapnav.mapping.wall_extraction import WallCandidate
from qmapnav.mapping.wall_extraction import WallExtractionConfig


__all__ = [
    'AccumulationResult',
    'AccumulationStatus',
    'AxisAlignedBox',
    'AssociationConfig',
    'AssociationDecision',
    'AssociationFailure',
    'AssociationResult',
    'AssociationScore',
    'BoundedProjectionWorker',
    'BoxEstimationConfig',
    'canonicalize_box',
    'canonicalize_class_name',
    'class_compatibility',
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
    'ProjectionPipeline',
    'LiftingPipeline',
    'DetectionProjection',
    'DepthFilterConfig',
    'estimate_upright_obb',
    'extract_wall_candidates',
    'GeometryEvaluation',
    'GeometrySource',
    'GeometryStatus',
    'GroundPlane',
    'LiftingCounts',
    'LiftingFrame',
    'LiftingResult',
    'intersect_ray_with_wall',
    'merge_wall_candidates',
    'ObjectCandidate3D',
    'ObjectAssociationEvent',
    'ObjectLifter',
    'ObjectLiftingConfig',
    'ObjectMap',
    'ObjectMapConfig',
    'OrientationConfidenceConfig',
    'PointSelectionConfig',
    'PointSelectionResult',
    'PersistentObjectRecord',
    'ProjectionConfig',
    'ProjectionDiagnostics',
    'ProjectionFrame',
    'ProjectionQualityConfig',
    'ProjectionResult',
    'ProjectionSynchronizer',
    'RayWallIntersection',
    'ReferenceUprightBox',
    'RegisteredScanAccumulator',
    'ScanAccumulatorConfig',
    'ScanAccumulatorStats',
    'score_candidate_instance',
    'TimedPanorama',
    'TimedPose',
    'TimedRegisteredScan',
    'StructuralAnchor',
    'StructuralAssociationEvent',
    'StructuralMap',
    'StructuralMapConfig',
    'StructuralRecord',
    'transform_camera_ray_to_map',
    'UprightOrientedBox',
    'upright_box_iou_3d',
    'ViewpointObservation',
    'WallCandidate',
    'WallExtractionConfig',
]
