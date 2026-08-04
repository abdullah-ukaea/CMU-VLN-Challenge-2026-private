"""Episode traces, proxy metrics, and regression evaluation."""

from qmapnav.evaluation.dataset_loader import DatasetLoadError
from qmapnav.evaluation.dataset_loader import load_ascii_trajectory_ply
from qmapnav.evaluation.dataset_loader import load_development_scenes
from qmapnav.evaluation.dataset_loader import load_oracle_scene
from qmapnav.evaluation.dataset_loader import load_questions
from qmapnav.evaluation.dataset_loader import load_unity_object_list
from qmapnav.evaluation.dataset_loader import load_unity_scene_objects
from qmapnav.evaluation.dataset_loader import load_vla3d_objects
from qmapnav.evaluation.dataset_loader import load_vla3d_regions
from qmapnav.evaluation.dataset_loader import load_vla3d_relations
from qmapnav.evaluation.dataset_loader import merge_unity_and_vla_objects
from qmapnav.evaluation.detector_benchmark import CandidatePredictions
from qmapnav.evaluation.detector_benchmark import DetectorBenchmarkCase
from qmapnav.evaluation.detector_benchmark import TwoCandidateDetectorBenchmark
from qmapnav.evaluation.detector_dataset import DetectorDataset
from qmapnav.evaluation.detector_dataset import DetectorDatasetCase
from qmapnav.evaluation.detector_dataset import load_detector_dataset
from qmapnav.evaluation.detector_dataset import roll_visible_instance
from qmapnav.evaluation.detector_metrics import DetectionMetricCounts
from qmapnav.evaluation.detector_metrics import empty_metric_counts
from qmapnav.evaluation.detector_metrics import score_panorama_detections
from qmapnav.evaluation.detector_metrics import VisibleInstance
from qmapnav.evaluation.ground_truth import ColourAttribute
from qmapnav.evaluation.ground_truth import ground_truth_to_data
from qmapnav.evaluation.ground_truth import ground_truth_to_json
from qmapnav.evaluation.ground_truth import OracleObject
from qmapnav.evaluation.ground_truth import OracleRegion
from qmapnav.evaluation.ground_truth import OracleRelation
from qmapnav.evaluation.ground_truth import OracleScene
from qmapnav.evaluation.ground_truth import OracleTrajectory
from qmapnav.evaluation.ground_truth import QuestionRecord
from qmapnav.evaluation.instance_fusion import AnchorStabilityMetrics
from qmapnav.evaluation.instance_fusion import evaluate_anchor_stability
from qmapnav.evaluation.instance_fusion import evaluate_fusion_geometry
from qmapnav.evaluation.instance_fusion import evaluate_identity_assignments
from qmapnav.evaluation.instance_fusion import FusionGeometryMetrics
from qmapnav.evaluation.instance_fusion import IdentityAssignment
from qmapnav.evaluation.instance_fusion import IdentityMetrics
from qmapnav.evaluation.metrics import count_accuracy_metric
from qmapnav.evaluation.metrics import CountAccuracyMetric
from qmapnav.evaluation.metrics import forbidden_region_metrics
from qmapnav.evaluation.metrics import ForbiddenRegionMetric
from qmapnav.evaluation.metrics import object_selection_metric
from qmapnav.evaluation.metrics import ObjectSelectionMetric
from qmapnav.evaluation.metrics import relation_metrics
from qmapnav.evaluation.metrics import RelationMetric
from qmapnav.evaluation.metrics import required_region_metrics
from qmapnav.evaluation.metrics import RequiredRegionMetric
from qmapnav.evaluation.metrics import semantic_route_metric
from qmapnav.evaluation.metrics import SemanticRouteMetric
from qmapnav.evaluation.metrics import terminal_goal_distance
from qmapnav.evaluation.metrics import TimingMetric
from qmapnav.evaluation.object_reference_contracts import (
    build_object_reference_manifest,
)
from qmapnav.evaluation.object_reference_contracts import manifest_digest
from qmapnav.evaluation.object_reference_contracts import ObjectReferenceCase
from qmapnav.evaluation.object_reference_contracts import (
    ObjectReferenceEpisodeResult,
)
from qmapnav.evaluation.object_reference_contracts import StageEvidence
from qmapnav.evaluation.object_reference_failures import (
    classify_primary_failure,
)
from qmapnav.evaluation.object_reference_failures import FailureClassification
from qmapnav.evaluation.object_reference_failures import FixCandidate
from qmapnav.evaluation.object_reference_failures import rank_fix_candidates
from qmapnav.evaluation.object_reference_runner import (
    ObjectReferenceBenchmarkRun,
)
from qmapnav.evaluation.object_reference_runner import (
    ObjectReferenceBenchmarkRunner,
)
from qmapnav.evaluation.object_reference_runner import (
    ObjectReferenceBenchmarkSummary,
)
from qmapnav.evaluation.object_reference_runner import write_episode_result
from qmapnav.evaluation.trace import DecisionTraceEvent
from qmapnav.evaluation.trace import InMemoryTraceRecorder
from qmapnav.evaluation.trace import JsonlDecisionTraceRecorder
from qmapnav.evaluation.trace import TRACE_SCHEMA_VERSION
from qmapnav.evaluation.trace import TraceRecorder
from qmapnav.evaluation.trace import TraceRecorderStats


__all__ = [
    'AnchorStabilityMetrics',
    'ColourAttribute',
    'CandidatePredictions',
    'CountAccuracyMetric',
    'DatasetLoadError',
    'DecisionTraceEvent',
    'DetectorBenchmarkCase',
    'InMemoryTraceRecorder',
    'JsonlDecisionTraceRecorder',
    'ForbiddenRegionMetric',
    'ObjectSelectionMetric',
    'ObjectReferenceBenchmarkRun',
    'ObjectReferenceBenchmarkRunner',
    'ObjectReferenceBenchmarkSummary',
    'ObjectReferenceCase',
    'ObjectReferenceEpisodeResult',
    'OracleObject',
    'OracleRegion',
    'OracleRelation',
    'OracleScene',
    'OracleTrajectory',
    'QuestionRecord',
    'RelationMetric',
    'RequiredRegionMetric',
    'SemanticRouteMetric',
    'TRACE_SCHEMA_VERSION',
    'TraceRecorder',
    'TraceRecorderStats',
    'TimingMetric',
    'TwoCandidateDetectorBenchmark',
    'DetectionMetricCounts',
    'DetectorDataset',
    'DetectorDatasetCase',
    'VisibleInstance',
    'empty_metric_counts',
    'evaluate_anchor_stability',
    'evaluate_fusion_geometry',
    'evaluate_identity_assignments',
    'FailureClassification',
    'FixCandidate',
    'FusionGeometryMetrics',
    'load_detector_dataset',
    'roll_visible_instance',
    'score_panorama_detections',
    'count_accuracy_metric',
    'forbidden_region_metrics',
    'ground_truth_to_data',
    'ground_truth_to_json',
    'IdentityAssignment',
    'IdentityMetrics',
    'load_ascii_trajectory_ply',
    'load_development_scenes',
    'load_oracle_scene',
    'load_questions',
    'load_unity_object_list',
    'load_unity_scene_objects',
    'load_vla3d_objects',
    'load_vla3d_regions',
    'load_vla3d_relations',
    'build_object_reference_manifest',
    'classify_primary_failure',
    'manifest_digest',
    'merge_unity_and_vla_objects',
    'object_selection_metric',
    'relation_metrics',
    'required_region_metrics',
    'rank_fix_candidates',
    'StageEvidence',
    'write_episode_result',
    'semantic_route_metric',
    'terminal_goal_distance',
]
