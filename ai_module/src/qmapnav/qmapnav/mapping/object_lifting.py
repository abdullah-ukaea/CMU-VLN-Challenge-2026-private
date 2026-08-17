"""Pure single-observation detection-to-3D lifting pipeline."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.bounding_boxes import robust_aabb
from qmapnav.mapping.cluster_selection import cluster_and_select
from qmapnav.mapping.cluster_selection import ClusterSelectionConfig
from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.depth_filter import select_foreground_depth_layer
from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.ground_filter import remove_ground_points
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import LiftingResult
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.orientation_confidence import conservative_orientation
from qmapnav.mapping.orientation_confidence import estimate_orientation_confidence
from qmapnav.mapping.orientation_confidence import OrientationConfidenceConfig
from qmapnav.mapping.point_selection import PointSelectionConfig
from qmapnav.mapping.point_selection import select_detection_points
from qmapnav.mapping.point_selection import SelectionMode
from qmapnav.perception.contracts import Detection2D


FLOOR_STANDING_CLASSES = frozenset(
    {
        'trash can',
        'trash_can',
        'garbage bin',
        'potted plant',
        'plant pot',
        'flower pot',
    }
)


@dataclass(frozen=True)
class ObjectLiftingConfig:
    """Validated policy for one lifting lifting observation."""

    selection: PointSelectionConfig = PointSelectionConfig()
    depth: DepthFilterConfig = DepthFilterConfig()
    clustering: ClusterSelectionConfig = ClusterSelectionConfig()
    boxes: BoxEstimationConfig = BoxEstimationConfig()
    orientation: OrientationConfidenceConfig = OrientationConfidenceConfig()
    ground_clearance_m: float = 0.07
    floor_standing_clearance_m: float = 0.02
    minimum_geometry_points: int = 3
    sparse_point_threshold: int = 8
    background_contamination_threshold: float = 0.85
    partial_boundary_fraction: float = 0.30

    def __post_init__(self) -> None:
        for name in ('ground_clearance_m', 'floor_standing_clearance_m'):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        if self.minimum_geometry_points < 3:
            raise ValueError('minimum_geometry_points must be at least three')
        if self.sparse_point_threshold < self.minimum_geometry_points:
            raise ValueError('sparse threshold must cover minimum geometry points')
        if not 0.0 <= self.background_contamination_threshold <= 1.0:
            raise ValueError('background contamination threshold is invalid')
        if not 0.0 <= self.partial_boundary_fraction <= 1.0:
            raise ValueError('partial boundary fraction is invalid')


class ObjectLifter:
    """Lift one perception detection from one projection projected cloud."""

    def __init__(self, config: ObjectLiftingConfig | None = None) -> None:
        self.config = config or ObjectLiftingConfig()

    def lift(
        self,
        detection: Detection2D,
        projection: ProjectionResult,
        *,
        source: GeometrySource,
        ground_plane: GroundPlane | None,
        use_mask: bool = True,
        scan_age_seconds: np.ndarray | None = None,
    ) -> LiftingResult:
        """Run selection, cleaning, clustering, box fitting and confidence."""
        started = perf_counter()
        selection = select_detection_points(
            detection,
            projection,
            use_mask=use_mask,
            config=self.config.selection,
        )
        selected = selection.selected_projection_indices
        base_counts = {
            'projected': projection.point_count,
            'box_selected': int(selection.contracted_box_indices.size),
            'mask_selected': (
                int(selected.size) if selection.mode is SelectionMode.MASK else 0
            ),
        }
        stages = {
            'original_box': selection.original_box_indices,
            'contracted_box': selection.contracted_box_indices,
            'selected': selected,
        }
        if selected.size == 0:
            return self._failure(
                detection,
                GeometryStatus.NO_POINTS,
                'no projected points in selected image region',
                base_counts,
                stages,
                started,
                {'selection_mode': selection.mode.value},
            )

        selected_points = projection.points_map_xyz[selected]
        clearance = (
            self.config.floor_standing_clearance_m
            if detection.class_name.casefold() in FLOOR_STANDING_CLASSES
            else self.config.ground_clearance_m
        )
        ground = remove_ground_points(
            selected_points,
            ground_plane,
            clearance_m=clearance,
        )
        post_ground = selected[ground.kept_indices]
        stages['post_ground'] = post_ground
        stages['ground_removed'] = selected[ground.removed_indices]
        if post_ground.size < self.config.minimum_geometry_points:
            return self._failure(
                detection,
                GeometryStatus.GROUND_DOMINATED,
                'too few points remain after ground filtering',
                {**base_counts, 'post_ground': int(post_ground.size)},
                stages,
                started,
                {
                    'ground_warning': ground.warning,
                    'ground_clearance_m': clearance,
                },
            )

        depth = select_foreground_depth_layer(
            projection.euclidean_range_m[post_ground],
            self.config.depth,
        )
        post_depth = post_ground[depth.kept_indices]
        stages['post_depth'] = post_depth
        stages['depth_removed'] = post_ground[depth.removed_indices]
        if post_depth.size < self.config.minimum_geometry_points:
            return self._failure(
                detection,
                GeometryStatus.SPARSE,
                'too few points remain after foreground depth filtering',
                {
                    **base_counts,
                    'post_ground': int(post_ground.size),
                    'post_depth': int(post_depth.size),
                },
                stages,
                started,
                {'depth_reason': depth.reason},
            )

        clustering = cluster_and_select(
            projection.points_map_xyz[post_depth],
            projection.euclidean_range_m[post_depth],
            projection.panorama_uv[post_depth],
            detection,
            self.config.clustering,
        )
        if clustering.selected_indices.size:
            final_indices = post_depth[clustering.selected_indices]
        elif post_depth.size < self.config.clustering.minimum_samples:
            # Sparse points cannot satisfy DBSCAN density, but may still define
            # a conservative debug candidate when they span valid 3D geometry.
            final_indices = post_depth
        else:
            return self._failure(
                detection,
                GeometryStatus.BACKGROUND_CONTAMINATED,
                'no coherent spatial cluster survived',
                {
                    **base_counts,
                    'post_ground': int(post_ground.size),
                    'post_depth': int(post_depth.size),
                },
                stages,
                started,
                {'cluster_reason': clustering.reason},
            )
        stages['clustered'] = final_indices
        stages['cluster_noise'] = post_depth[clustering.noise_indices]
        final_points = projection.points_map_xyz[final_indices]
        try:
            aabb = robust_aabb(final_points, self.config.boxes)
            estimated_box = estimate_upright_obb(final_points, self.config.boxes)
        except ValueError as error:
            return self._failure(
                detection,
                GeometryStatus.INVALID_GEOMETRY,
                f'box estimation failed: {error}',
                {
                    **base_counts,
                    'post_ground': int(post_ground.size),
                    'post_depth': int(post_depth.size),
                    'clustered': int(final_indices.size),
                },
                stages,
                started,
                {'cluster_reason': clustering.reason},
            )

        depths = projection.euclidean_range_m[final_indices]
        depth_iqr = (
            float(np.percentile(depths, 75) - np.percentile(depths, 25))
            if depths.size > 1
            else 0.0
        )
        cluster_purity = float(final_indices.size / max(post_depth.size, 1))
        image_coverage = float(
            min(1.0, final_indices.size / max(selection.original_box_indices.size, 1))
        )
        timing_quality = (
            0.5 if projection.diagnostics.timing_warning else 1.0
        )
        orientation = estimate_orientation_confidence(
            final_points,
            estimated_box,
            depth_iqr_m=depth_iqr,
            cluster_purity=cluster_purity,
            image_coverage=image_coverage,
            timing_quality=timing_quality,
            boundary_fraction=selection.boundary_fraction,
            config=self.config.orientation,
        )
        marker_yaw, marker_dimensions, fallback, orientation_state = (
            conservative_orientation(
                estimated_box,
                aabb.dimensions_xyz,
                orientation.confidence,
                self.config.orientation,
            )
        )
        components = orientation.components
        geometry_confidence = float(
            np.clip(
                (
                    max(components.point_support, 1e-6)
                    * max(components.depth_consistency, 1e-6)
                    * max(components.cluster_purity, 1e-6)
                    * max(components.timing_quality, 1e-6)
                ) ** 0.25
                * detection.confidence,
                0.0,
                1.0,
            )
        )
        partial = (
            selection.boundary_fraction >= self.config.partial_boundary_fraction
            or final_indices.size < self.config.sparse_point_threshold
            or np.any(
                estimated_box.raw_dimensions_xyz
                > estimated_box.dimensions_xyz * 1.5
            )
        )
        if clustering.ambiguous:
            status = GeometryStatus.MULTIPLE_CLUSTERS
        elif final_indices.size < self.config.sparse_point_threshold:
            status = GeometryStatus.SPARSE
        elif orientation.confidence < self.config.orientation.low_confidence:
            status = GeometryStatus.UNSTABLE_ORIENTATION
        elif (
            depth.contamination_fraction
            >= self.config.background_contamination_threshold
            and cluster_purity < 0.35
        ):
            status = GeometryStatus.BACKGROUND_CONTAMINATED
        else:
            status = GeometryStatus.GOOD
        counts = LiftingCounts(
            projected=projection.point_count,
            box_selected=base_counts['box_selected'],
            mask_selected=base_counts['mask_selected'],
            post_ground=int(post_ground.size),
            post_depth=int(post_depth.size),
            clustered=int(final_indices.size),
            final=int(final_indices.size),
        )
        age_spread = _selected_age_spread(scan_age_seconds, final_indices)
        candidate = ObjectCandidate3D(
            candidate_id=f'{projection.image_id}:{detection.detection_id}',
            detection_id=detection.detection_id,
            class_name=detection.class_name,
            detection_confidence=detection.confidence,
            source=source,
            source_timestamp_ns=projection.scan_timestamp_ns,
            image_timestamp_ns=projection.image_timestamp_ns,
            scan_timestamp_ns=projection.scan_timestamp_ns,
            pose_timestamp_ns=projection.image_timestamp_ns,
            pose_mode=projection.diagnostics.pose_mode,
            image_scan_delta_ms=projection.diagnostics.image_scan_delta_ms,
            pose_before_delta_ms=(
                projection.diagnostics.pose_before_delta_ms
            ),
            pose_after_delta_ms=projection.diagnostics.pose_after_delta_ms,
            timing_warning=projection.diagnostics.timing_warning,
            points_map_xyz=final_points,
            source_projection_indices=final_indices,
            point_centroid_xyz=np.median(final_points, axis=0),
            aabb_min_xyz=aabb.minimum_xyz,
            aabb_max_xyz=aabb.maximum_xyz,
            obb_centre_xyz=(
                aabb.centre_xyz if fallback else estimated_box.centre_xyz
            ),
            obb_dimensions_xyz=marker_dimensions,
            obb_yaw_rad=marker_yaw,
            estimated_yaw_rad=estimated_box.yaw_rad,
            orientation_confidence=orientation.confidence,
            geometry_confidence=geometry_confidence,
            geometry_status=status,
            partial_geometry=partial,
            low_orientation_fallback=fallback,
            counts=counts,
            confidence_components=components,
            diagnostics={
                'selection_mode': selection.mode.value,
                'selection_reason': selection.reason,
                'ground_warning': ground.warning,
                'ground_clearance_m': clearance,
                'depth_reason': depth.reason,
                'depth_band_m': (depth.lower_depth_m, depth.upper_depth_m),
                'depth_contamination_fraction': depth.contamination_fraction,
                'depth_iqr_m': depth_iqr,
                'cluster_epsilon_m': clustering.epsilon_m,
                'cluster_reason': clustering.reason,
                'cluster_count': len(clustering.summaries),
                'cluster_scores': tuple(
                    (summary.cluster_id, summary.total_score)
                    for summary in clustering.summaries
                ),
                'orientation_state': orientation_state,
                'pca_yaw_rad': estimated_box.pca_yaw_rad,
                'estimator_disagreement_rad': (
                    orientation.estimator_disagreement_rad
                ),
                'maximum_subset_error_rad': orientation.maximum_subset_error_rad,
                'scan_age_spread_seconds': age_spread,
            },
        )
        return LiftingResult(
            detection_id=detection.detection_id,
            status=status,
            candidate=candidate,
            counts=counts,
            reason=_status_reason(status, orientation_state),
            processing_time_ms=(perf_counter() - started) * 1000.0,
            stage_indices=stages,
            diagnostics=dict(candidate.diagnostics),
        )

    def _failure(
        self,
        detection: Detection2D,
        status: GeometryStatus,
        reason: str,
        partial_counts: dict[str, int],
        stages: dict[str, np.ndarray],
        started: float,
        diagnostics: dict[str, object],
    ) -> LiftingResult:
        counts = LiftingCounts(
            projected=partial_counts.get('projected', 0),
            box_selected=partial_counts.get('box_selected', 0),
            mask_selected=partial_counts.get('mask_selected', 0),
            post_ground=partial_counts.get('post_ground', 0),
            post_depth=partial_counts.get('post_depth', 0),
            clustered=partial_counts.get('clustered', 0),
            final=0,
        )
        return LiftingResult(
            detection_id=detection.detection_id,
            status=status,
            candidate=None,
            counts=counts,
            reason=reason,
            processing_time_ms=(perf_counter() - started) * 1000.0,
            stage_indices=stages,
            diagnostics=diagnostics,
        )


def _selected_age_spread(
    scan_age_seconds: np.ndarray | None,
    projection_indices: np.ndarray,
) -> float | None:
    if scan_age_seconds is None or projection_indices.size == 0:
        return None
    ages = np.asarray(scan_age_seconds, dtype=np.float64)
    if ages.ndim != 1 or np.max(projection_indices) >= ages.size:
        raise ValueError('scan_age_seconds must align with projection points')
    selected = ages[projection_indices]
    return float(np.max(selected) - np.min(selected))


def _status_reason(status: GeometryStatus, orientation_state: str) -> str:
    reasons = {
        GeometryStatus.GOOD: 'coherent foreground geometry',
        GeometryStatus.SPARSE: 'valid but sparse foreground geometry',
        GeometryStatus.MULTIPLE_CLUSTERS: 'primary cluster selected with alternatives',
        GeometryStatus.UNSTABLE_ORIENTATION: (
            f'credible geometry with {orientation_state} orientation'
        ),
        GeometryStatus.BACKGROUND_CONTAMINATED: (
            'foreground selected but background contamination remains high'
        ),
    }
    return reasons.get(status, status.value)


__all__ = ['FLOOR_STANDING_CLASSES', 'ObjectLifter', 'ObjectLiftingConfig']
