"""Runtime parameter declarations and Q-MapNav subsystem factories."""

from pathlib import Path

import numpy as np

from qmapnav.common.decision_trace import JsonlDecisionTraceRecorder
from qmapnav.mapping import AssociationConfig
from qmapnav.mapping import DenseRegisteredScanAccumulator
from qmapnav.mapping import DenseScanAccumulatorConfig
from qmapnav.mapping import ProjectionConfig
from qmapnav.mapping import ProjectionPipeline
from qmapnav.mapping import ProjectionQualityConfig
from qmapnav.mapping import ProjectionSynchronizer
from qmapnav.mapping import RegisteredScanAccumulator
from qmapnav.mapping import ScanAccumulatorConfig
from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.cluster_selection import ClusterSelectionConfig
from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.lifting_pipeline import LiftingPipeline
from qmapnav.mapping.object_association import (
    AssociationConfig as ObjectAssociationConfig,
)
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.object_map import ObjectMapConfig
from qmapnav.mapping.orientation_confidence import OrientationConfidenceConfig
from qmapnav.mapping.point_selection import PointSelectionConfig
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.structural_map import StructuralMapConfig
from qmapnav.mapping.transforms import make_transform
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation
from qmapnav.mapping.wall_extraction import WallExtractionConfig
from qmapnav.navigation import DEFAULT_ARRIVAL_RADIUS
from qmapnav.navigation import DEFAULT_DIRECT_REPUBLISH_LIMIT
from qmapnav.navigation import DEFAULT_NO_PROGRESS_TIMEOUT
from qmapnav.navigation import DEFAULT_PROGRESS_EPSILON
from qmapnav.navigation import DEFAULT_SAFE_OFFSET_LIMIT
from qmapnav.navigation import TargetedViewpointConfig
from qmapnav.reasoning.ambiguity import AmbiguityConfig
from qmapnav.reasoning.candidate_generation import CandidateGenerationConfig
from qmapnav.reasoning.colour_classifier import ColourClassifierConfig
from qmapnav.reasoning.colour_pixel_filter import ColourSelectionConfig
from qmapnav.reasoning.corridor_evaluation import CorridorConfig
from qmapnav.reasoning.relation_graph import RelationGraph
from qmapnav.reasoning.spatial_relations import SpatialRelationConfig
from qmapnav.reasoning.support_relations import SupportRelationConfig
from qmapnav.reasoning.vertical_relations import VerticalRelationConfig


def declare_parameters(node) -> None:
    """Declare the frozen ROS parameter surface on ``node``."""
    declarations = (
        ('arrival_radius', DEFAULT_ARRIVAL_RADIUS),
        ('progress_epsilon', DEFAULT_PROGRESS_EPSILON),
        ('no_progress_timeout', DEFAULT_NO_PROGRESS_TIMEOUT),
        ('direct_republish_limit', DEFAULT_DIRECT_REPUBLISH_LIMIT),
        ('safe_offset_limit', DEFAULT_SAFE_OFFSET_LIMIT),
        ('watchdog_period', 0.25),
        ('recovery_offset_distance', 0.75),
        ('recovery_clearance', 0.35),
        ('episode_time_limit', 600.0),
        ('detector_checkpoint', '/home/docker/models/yoloe-11s-seg.pt'),
        ('detector_confidence_threshold', 0.20),
        ('detector_cross_crop_iou_threshold', 0.40),
        ('numerical_required_consecutive_updates', 3),
        ('numerical_required_independent_viewpoints', 2),
        ('numerical_minimum_count_confidence', 0.50),
        ('numerical_maximum_unresolved_candidates', 0),
        ('numerical_final_commit_reserve_sec', 30.0),
        ('numerical_maximum_verification_sec', 180.0),
        ('instruction_initial_observations', 3),
        ('instruction_reobservation_observations', 3),
        ('instruction_grid_half_extent_m', 10.0),
        ('instruction_grid_resolution_m', 0.25),
        ('instruction_grid_known_free_clearance_m', 0.25),
        ('instruction_waypoint_spacing_m', 1.0),
        ('instruction_final_route_reserve_sec', 300.0),
        ('publish_object_matching_waypoint', True),
        ('object_reference_initial_observations', 3),
        ('object_reference_final_commit_reserve_sec', 30.0),
        ('object_reference_result_directory', '/tmp/qmapnav/object_reference'),
        ('object_reference_case_id', 'runtime_object_reference'),
        ('object_reference_scene_id', 'unknown'),
        ('object_reference_run_id', 'runtime'),
        ('targeted_viewpoint_minimum_margin', 0.12),
        ('targeted_viewpoint_minimum_geometry_confidence', 0.35),
        ('targeted_viewpoint_minimum_time_remaining', 45.0),
        ('targeted_viewpoint_standoff_m', 2.0),
        ('targeted_viewpoint_lateral_offset_m', 0.8),
        ('targeted_viewpoint_maximum_travel_m', 4.0),
        ('targeted_viewpoint_clearance_m', 0.35),
        ('scan_frame', 'map'),
        ('scan_voxel_size', 0.20),
        ('scan_max_range', 30.0),
        ('scan_max_age_seconds', 120.0),
        ('scan_max_voxels', 200_000),
        ('scan_max_views', 16),
        ('pose_parent_frame', 'map'),
        ('pose_child_frame', 'sensor'),
        ('panorama_width', 1920),
        ('panorama_height', 640),
        ('camera_translation_sensor_xyz', [0.0, 0.0, 0.1]),
        ('camera_orientation_sensor_xyzw', [-0.5, 0.5, -0.5, 0.5]),
        ('projection_max_pose_delta_ms', 50.0),
        ('projection_max_scan_delta_ms', 150.0),
        ('projection_buffer_seconds', 5.0),
        ('projection_max_pose_items', 2_000),
        ('projection_max_scan_items', 64),
        ('projection_min_range', 0.30),
        ('projection_max_range', 30.0),
        ('projection_timing_warning_ms', 100.0),
        ('projection_sparse_point_threshold', 8),
        ('projection_high_depth_iqr', 2.0),
        ('dense_scan_voxel_size', 0.04),
        ('dense_scan_max_age_seconds', 15.0),
        ('dense_scan_max_radius', 12.0),
        ('dense_scan_max_points', 1_000_000),
        ('projection_worker_queue_size', 2),
        ('projection_shutdown_timeout', 2.0),
        ('projection_debug_directory', ''),
        ('projection_max_saved_frames', 5),
        ('lifting_source', 'accumulated'),
        ('lifting_use_masks', False),
        ('lifting_bbox_inner_margin', 0.05),
        ('lifting_ground_clearance', 0.07),
        ('lifting_floor_standing_clearance', 0.02),
        ('lifting_depth_bin_width', 0.15),
        ('lifting_depth_minimum_mode_points', 5),
        ('lifting_depth_maximum_band', 1.5),
        ('lifting_dbscan_base_epsilon', 0.07),
        ('lifting_dbscan_range_slope', 0.015),
        ('lifting_dbscan_minimum_samples', 5),
        ('lifting_box_lower_percentile', 2.5),
        ('lifting_box_upper_percentile', 97.5),
        ('lifting_orientation_low_confidence', 0.40),
        ('lifting_orientation_high_confidence', 0.70),
        ('lifting_sparse_point_threshold', 8),
        ('object_map_accept_threshold', 0.62),
        ('object_map_uncertain_threshold', 0.55),
        ('object_map_yaw_confidence_threshold', 0.50),
        ('object_map_same_keyframe_distance', 0.30),
        ('object_map_same_keyframe_overlap', 0.60),
        ('object_map_fused_voxel_size', 0.03),
        ('object_map_max_points_per_instance', 50_000),
        ('object_map_max_total_points', 500_000),
        ('object_map_max_observation_history', 100),
        ('colour_prototype_path', ''),
        ('colour_min_valid_pixels', 50),
        ('colour_mask_erosion_px', 3),
        ('colour_small_object_mask_erosion_px', 1),
        ('colour_geometry_support_dilation_px', 3),
        ('colour_contracted_box_margin_fraction', 0.08),
        ('colour_low_saturation_threshold', 0.15),
        ('colour_shadow_lower_percentile', 5.0),
        ('colour_highlight_value_threshold', 0.92),
        ('colour_highlight_saturation_threshold', 0.12),
        ('colour_probability_temperature', 1.0),
        ('colour_ambiguous_margin', 0.12),
        ('colour_max_fused_evidence', 12.0),
        ('colour_max_observation_history', 32),
        ('relation_above_vertical_tolerance', 0.08),
        ('relation_maximum_support_gap', 0.15),
        ('relation_penetration_tolerance', 0.08),
        ('relation_minimum_subject_support_overlap', 0.50),
        ('relation_support_search_radius', 2.0),
        ('relation_minimum_geometry_confidence', 0.25),
        ('relation_accept_on_confidence', 0.70),
        ('relation_uncertain_on_confidence', 0.40),
        ('relation_include_floor_supports', False),
        ('reasoning_minimum_class_probability', 0.15),
        ('reasoning_minimum_colour_probability', 0.10),
        ('reasoning_minimum_geometry_confidence', 0.20),
        ('reasoning_near_base_margin_m', 0.40),
        ('reasoning_near_size_scale', 0.75),
        ('reasoning_between_projection_tolerance', 0.05),
        ('reasoning_between_max_relative_perpendicular_distance', 0.35),
        ('reasoning_between_min_anchor_separation_m', 0.30),
        ('reasoning_resolved_minimum_score', 0.65),
        ('reasoning_resolved_minimum_margin', 0.12),
        ('reasoning_ambiguous_margin', 0.08),
        ('robot_footprint_width_m', 0.55),
        ('corridor_safety_clearance_m', 0.15),
        ('corridor_minimum_depth_m', 0.60),
        ('corridor_occupancy_free_fraction', 0.90),
        ('corridor_maximum_anchor_separation_m', 5.0),
        ('structural_wall_update_interval', 10),
        ('wall_min_height_above_ground', 0.20),
        ('wall_min_segment_length', 0.50),
        ('wall_max_line_residual', 0.08),
        ('wall_merge_angle_deg', 5.0),
        ('wall_merge_perpendicular_distance', 0.12),
        ('wall_preserve_opening_width', 0.60),
        ('wall_max_candidate_points', 50_000),
        ('structural_ray_parallel_epsilon', 1.0e-6),
        ('structural_max_wall_extent_margin', 0.25),
        ('structural_anchor_merge_distance', 0.55),
        ('structural_ambiguous_wall_distance_margin', 0.10),
        ('trace_path', '/tmp/qmapnav/decision_trace.jsonl'),
        ('trace_max_queue_size', 512),
        ('trace_max_file_bytes', 4 * 1024 * 1024),
        ('trace_flush_timeout', 1.0),
    )
    for name, default in declarations:
        node.declare_parameter(name, default)


def make_accumulator(node) -> RegisteredScanAccumulator:
    """Build the bounded registered-scan accumulator."""
    return RegisteredScanAccumulator(ScanAccumulatorConfig(
        frame_id=str(node.get_parameter('scan_frame').value),
        voxel_size=float(node.get_parameter('scan_voxel_size').value),
        max_range=float(node.get_parameter('scan_max_range').value),
        max_age_seconds=float(
            node.get_parameter('scan_max_age_seconds').value
        ),
        max_voxels=int(node.get_parameter('scan_max_voxels').value),
        max_scan_views=int(node.get_parameter('scan_max_views').value),
    ))


def make_projection_pipeline(node) -> ProjectionPipeline:
    """Build the synchronized projection and dense accumulation pipeline."""
    translation = np.asarray(
        node.get_parameter('camera_translation_sensor_xyz').value,
        dtype=np.float64,
    )
    quaternion = np.asarray(
        node.get_parameter('camera_orientation_sensor_xyzw').value,
        dtype=np.float64,
    )
    transform_sensor_from_camera = make_transform(
        quaternion_xyzw_to_rotation(quaternion), translation
    )
    association = AssociationConfig(
        max_pose_delta_ns=int(
            float(node.get_parameter('projection_max_pose_delta_ms').value)
            * 1_000_000
        ),
        max_scan_delta_ns=int(
            float(node.get_parameter('projection_max_scan_delta_ms').value)
            * 1_000_000
        ),
        buffer_duration_ns=int(
            float(node.get_parameter('projection_buffer_seconds').value)
            * 1_000_000_000
        ),
        max_pose_items=int(node.get_parameter('projection_max_pose_items').value),
        max_scan_items=int(node.get_parameter('projection_max_scan_items').value),
    )
    dense_config = DenseScanAccumulatorConfig(
        frame_id=str(node.get_parameter('scan_frame').value),
        voxel_size_m=float(node.get_parameter('dense_scan_voxel_size').value),
        max_age_seconds=float(
            node.get_parameter('dense_scan_max_age_seconds').value
        ),
        max_radius_m=float(node.get_parameter('dense_scan_max_radius').value),
        max_points=int(node.get_parameter('dense_scan_max_points').value),
    )
    projection_config = ProjectionConfig(
        expected_scan_frame=str(node.get_parameter('scan_frame').value),
        expected_pose_parent_frame=str(
            node.get_parameter('pose_parent_frame').value
        ),
        expected_pose_child_frame=str(
            node.get_parameter('pose_child_frame').value
        ),
        min_range_m=float(node.get_parameter('projection_min_range').value),
        max_range_m=float(node.get_parameter('projection_max_range').value),
        timing_warning_ms=float(
            node.get_parameter('projection_timing_warning_ms').value
        ),
    )
    quality_config = ProjectionQualityConfig(
        sparse_point_threshold=int(
            node.get_parameter('projection_sparse_point_threshold').value
        ),
        high_depth_iqr_m=float(
            node.get_parameter('projection_high_depth_iqr').value
        ),
        timing_warning_ms=projection_config.timing_warning_ms,
    )
    from qmapnav.perception.panorama_projection import PanoramaCameraModel

    return ProjectionPipeline(
        synchronizer=ProjectionSynchronizer(association),
        dense_accumulator=DenseRegisteredScanAccumulator(dense_config),
        transform_sensor_from_camera_optical=transform_sensor_from_camera,
        panorama_model=PanoramaCameraModel(
            int(node.get_parameter('panorama_width').value),
            int(node.get_parameter('panorama_height').value),
        ),
        projection_config=projection_config,
        quality_config=quality_config,
    )


def make_lifting_pipeline(node) -> LiftingPipeline:
    """Build the single-observation object lifting pipeline."""
    source_text = str(node.get_parameter('lifting_source').value)
    try:
        source = GeometrySource(source_text)
    except ValueError as error:
        raise ValueError('lifting_source must be current or accumulated') from error
    config = ObjectLiftingConfig(
        selection=PointSelectionConfig(
            bbox_inner_margin_fraction=float(
                node.get_parameter('lifting_bbox_inner_margin').value
            )
        ),
        depth=DepthFilterConfig(
            bin_width_m=float(node.get_parameter('lifting_depth_bin_width').value),
            minimum_mode_points=int(
                node.get_parameter('lifting_depth_minimum_mode_points').value
            ),
            maximum_band_width_m=float(
                node.get_parameter('lifting_depth_maximum_band').value
            ),
        ),
        clustering=ClusterSelectionConfig(
            base_epsilon_m=float(
                node.get_parameter('lifting_dbscan_base_epsilon').value
            ),
            range_epsilon_slope=float(
                node.get_parameter('lifting_dbscan_range_slope').value
            ),
            minimum_samples=int(
                node.get_parameter('lifting_dbscan_minimum_samples').value
            ),
        ),
        boxes=BoxEstimationConfig(
            lower_percentile=float(
                node.get_parameter('lifting_box_lower_percentile').value
            ),
            upper_percentile=float(
                node.get_parameter('lifting_box_upper_percentile').value
            ),
        ),
        orientation=OrientationConfidenceConfig(
            low_confidence=float(
                node.get_parameter('lifting_orientation_low_confidence').value
            ),
            high_confidence=float(
                node.get_parameter('lifting_orientation_high_confidence').value
            ),
        ),
        ground_clearance_m=float(
            node.get_parameter('lifting_ground_clearance').value
        ),
        floor_standing_clearance_m=float(
            node.get_parameter('lifting_floor_standing_clearance').value
        ),
        sparse_point_threshold=int(
            node.get_parameter('lifting_sparse_point_threshold').value
        ),
    )
    return LiftingPipeline(
        ObjectLifter(config),
        source=source,
        use_masks=bool(node.get_parameter('lifting_use_masks').value),
    )


def make_object_map(node) -> ObjectMap:
    """Build the bounded persistent object map."""
    return ObjectMap(ObjectMapConfig(
        association=ObjectAssociationConfig(
            accept_threshold=float(
                node.get_parameter('object_map_accept_threshold').value
            ),
            uncertain_threshold=float(
                node.get_parameter('object_map_uncertain_threshold').value
            ),
            yaw_confidence_threshold=float(
                node.get_parameter('object_map_yaw_confidence_threshold').value
            ),
        ),
        same_keyframe_distance_m=float(
            node.get_parameter('object_map_same_keyframe_distance').value
        ),
        same_keyframe_overlap_threshold=float(
            node.get_parameter('object_map_same_keyframe_overlap').value
        ),
        fused_voxel_size_m=float(
            node.get_parameter('object_map_fused_voxel_size').value
        ),
        max_fused_points_per_instance=int(
            node.get_parameter('object_map_max_points_per_instance').value
        ),
        max_total_fused_points=int(
            node.get_parameter('object_map_max_total_points').value
        ),
        max_observation_history=int(
            node.get_parameter('object_map_max_observation_history').value
        ),
        max_colour_history=int(
            node.get_parameter('colour_max_observation_history').value
        ),
        max_colour_evidence=float(
            node.get_parameter('colour_max_fused_evidence').value
        ),
    ))


def colour_prototype_path(node) -> Path:
    """Resolve the configured source-tree or installed colour asset."""
    value = str(node.get_parameter('colour_prototype_path').value)
    if value:
        return Path(value)
    source_path = Path(__file__).resolve().parents[2] / (
        'data/colour_prototypes.json'
    )
    if source_path.exists():
        return source_path
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory('qmapnav')) / (
        'data/colour_prototypes.json'
    )


def make_colour_selection_config(node) -> ColourSelectionConfig:
    """Build the image/geometry colour-pixel selection policy."""
    return ColourSelectionConfig(
        min_valid_pixels=int(node.get_parameter('colour_min_valid_pixels').value),
        mask_erosion_px=int(node.get_parameter('colour_mask_erosion_px').value),
        small_object_mask_erosion_px=int(
            node.get_parameter('colour_small_object_mask_erosion_px').value
        ),
        geometry_support_dilation_px=int(
            node.get_parameter('colour_geometry_support_dilation_px').value
        ),
        contracted_box_margin_fraction=float(
            node.get_parameter('colour_contracted_box_margin_fraction').value
        ),
        low_saturation_threshold=float(
            node.get_parameter('colour_low_saturation_threshold').value
        ),
        shadow_lower_percentile=float(
            node.get_parameter('colour_shadow_lower_percentile').value
        ),
        highlight_value_threshold=float(
            node.get_parameter('colour_highlight_value_threshold').value
        ),
        highlight_saturation_threshold=float(
            node.get_parameter('colour_highlight_saturation_threshold').value
        ),
    )


def make_colour_classifier_config(node) -> ColourClassifierConfig:
    """Build the colour vocabulary classifier policy."""
    return ColourClassifierConfig(
        probability_temperature=float(
            node.get_parameter('colour_probability_temperature').value
        ),
        ambiguous_margin=float(
            node.get_parameter('colour_ambiguous_margin').value
        ),
        min_valid_pixels=int(node.get_parameter('colour_min_valid_pixels').value),
        low_saturation_threshold=float(
            node.get_parameter('colour_low_saturation_threshold').value
        ),
    )


def make_relation_graph(node) -> RelationGraph:
    """Build the derived support-relation graph policy."""
    return RelationGraph(
        VerticalRelationConfig(
            vertical_tolerance_m=float(
                node.get_parameter('relation_above_vertical_tolerance').value
            ),
        ),
        SupportRelationConfig(
            maximum_support_gap_m=float(
                node.get_parameter('relation_maximum_support_gap').value
            ),
            penetration_tolerance_m=float(
                node.get_parameter('relation_penetration_tolerance').value
            ),
            minimum_subject_support_overlap=float(
                node.get_parameter(
                    'relation_minimum_subject_support_overlap'
                ).value
            ),
            support_search_radius_m=float(
                node.get_parameter('relation_support_search_radius').value
            ),
            minimum_geometry_confidence=float(
                node.get_parameter('relation_minimum_geometry_confidence').value
            ),
            accept_on_confidence=float(
                node.get_parameter('relation_accept_on_confidence').value
            ),
            uncertain_on_confidence=float(
                node.get_parameter('relation_uncertain_on_confidence').value
            ),
            include_floor_supports=bool(
                node.get_parameter('relation_include_floor_supports').value
            ),
        ),
    )


def make_reasoning_configs(node):
    """Build candidate, spatial, ambiguity, and corridor reasoning policies."""
    candidate = CandidateGenerationConfig(
        minimum_class_probability=float(
            node.get_parameter('reasoning_minimum_class_probability').value
        ),
        minimum_colour_probability=float(
            node.get_parameter('reasoning_minimum_colour_probability').value
        ),
        minimum_geometry_confidence=float(
            node.get_parameter('reasoning_minimum_geometry_confidence').value
        ),
    )
    spatial = SpatialRelationConfig(
        minimum_geometry_confidence=candidate.minimum_geometry_confidence,
        near_base_margin_m=float(
            node.get_parameter('reasoning_near_base_margin_m').value
        ),
        near_size_scale=float(
            node.get_parameter('reasoning_near_size_scale').value
        ),
        between_projection_tolerance=float(
            node.get_parameter('reasoning_between_projection_tolerance').value
        ),
        between_max_relative_perpendicular_distance=float(
            node.get_parameter(
                'reasoning_between_max_relative_perpendicular_distance'
            ).value
        ),
        between_min_anchor_separation_m=float(
            node.get_parameter(
                'reasoning_between_min_anchor_separation_m'
            ).value
        ),
    )
    ambiguity = AmbiguityConfig(
        resolved_minimum_score=float(
            node.get_parameter('reasoning_resolved_minimum_score').value
        ),
        resolved_minimum_margin=float(
            node.get_parameter('reasoning_resolved_minimum_margin').value
        ),
        ambiguous_margin=float(
            node.get_parameter('reasoning_ambiguous_margin').value
        ),
    )
    corridor = CorridorConfig(
        robot_width_m=float(
            node.get_parameter('robot_footprint_width_m').value
        ),
        safety_clearance_m=float(
            node.get_parameter('corridor_safety_clearance_m').value
        ),
        minimum_depth_m=float(
            node.get_parameter('corridor_minimum_depth_m').value
        ),
        occupancy_free_fraction=float(
            node.get_parameter('corridor_occupancy_free_fraction').value
        ),
        maximum_anchor_separation_m=float(
            node.get_parameter('corridor_maximum_anchor_separation_m').value
        ),
    )
    return candidate, spatial, ambiguity, corridor


def make_structural_map(node) -> StructuralMap:
    """Build the bounded structural wall and anchor map."""
    wall_config = WallExtractionConfig(
        min_height_above_ground_m=float(
            node.get_parameter('wall_min_height_above_ground').value
        ),
        min_segment_length_m=float(
            node.get_parameter('wall_min_segment_length').value
        ),
        max_line_residual_m=float(
            node.get_parameter('wall_max_line_residual').value
        ),
        merge_angle_deg=float(node.get_parameter('wall_merge_angle_deg').value),
        merge_perpendicular_distance_m=float(
            node.get_parameter('wall_merge_perpendicular_distance').value
        ),
        preserve_opening_width_m=float(
            node.get_parameter('wall_preserve_opening_width').value
        ),
        max_candidate_points=int(
            node.get_parameter('wall_max_candidate_points').value
        ),
    )
    return StructuralMap(StructuralMapConfig(
        wall_extraction=wall_config,
        ray_parallel_epsilon=float(
            node.get_parameter('structural_ray_parallel_epsilon').value
        ),
        max_wall_extent_margin_m=float(
            node.get_parameter('structural_max_wall_extent_margin').value
        ),
        anchor_merge_distance_m=float(
            node.get_parameter('structural_anchor_merge_distance').value
        ),
        ambiguous_wall_distance_margin_m=float(
            node.get_parameter(
                'structural_ambiguous_wall_distance_margin'
            ).value
        ),
    ))


def make_targeted_viewpoint_config(node) -> TargetedViewpointConfig:
    """Build the bounded targeted-viewpoint policy."""
    return TargetedViewpointConfig(
        minimum_confidence_margin=float(
            node.get_parameter('targeted_viewpoint_minimum_margin').value
        ),
        minimum_geometry_confidence=float(
            node.get_parameter(
                'targeted_viewpoint_minimum_geometry_confidence'
            ).value
        ),
        minimum_time_remaining_sec=float(
            node.get_parameter('targeted_viewpoint_minimum_time_remaining').value
        ),
        preferred_standoff_m=float(
            node.get_parameter('targeted_viewpoint_standoff_m').value
        ),
        lateral_offset_m=float(
            node.get_parameter('targeted_viewpoint_lateral_offset_m').value
        ),
        maximum_travel_m=float(
            node.get_parameter('targeted_viewpoint_maximum_travel_m').value
        ),
    )


def make_trace_recorder(node) -> JsonlDecisionTraceRecorder:
    """Build the bounded asynchronous decision trace recorder."""
    return JsonlDecisionTraceRecorder(
        str(node.get_parameter('trace_path').value),
        max_queue_size=int(node.get_parameter('trace_max_queue_size').value),
        max_file_bytes=int(node.get_parameter('trace_max_file_bytes').value),
    )


__all__ = [
    'colour_prototype_path',
    'declare_parameters',
    'make_accumulator',
    'make_colour_classifier_config',
    'make_colour_selection_config',
    'make_lifting_pipeline',
    'make_object_map',
    'make_projection_pipeline',
    'make_reasoning_configs',
    'make_relation_graph',
    'make_structural_map',
    'make_targeted_viewpoint_config',
    'make_trace_recorder',
]
