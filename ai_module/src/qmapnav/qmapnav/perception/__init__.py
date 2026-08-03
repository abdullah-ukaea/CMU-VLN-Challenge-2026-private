"""Query-conditioned image perception and observation processing."""

from qmapnav.perception.baseline import DAY4_BASELINE_CHECKPOINT
from qmapnav.perception.baseline import DAY4_BASELINE_CONFIDENCE
from qmapnav.perception.baseline import DAY4_BASELINE_CROSS_CROP_IOU
from qmapnav.perception.baseline import make_day4_baseline_worker
from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PanoramaBox
from qmapnav.perception.contracts import PerceptionRequest
from qmapnav.perception.contracts import PerceptionResult
from qmapnav.perception.contracts import PerspectiveGeometry
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.crop_generator import eight_view_layout
from qmapnav.perception.crop_generator import PerspectiveCropGenerator
from qmapnav.perception.crop_generator import PerspectiveCropLayout
from qmapnav.perception.cross_crop_nms import cross_crop_nms
from qmapnav.perception.cross_crop_nms import panorama_box_intersection_over_smaller
from qmapnav.perception.cross_crop_nms import panorama_box_iou
from qmapnav.perception.cross_crop_nms import project_crop_detections
from qmapnav.perception.detector_interface import DetectorIdentity
from qmapnav.perception.detector_interface import OpenVocabularyDetector
from qmapnav.perception.grounding_dino_detector import GroundingDinoTinyDetector
from qmapnav.perception.panorama_projection import camera_rays_to_crop_pixels
from qmapnav.perception.panorama_projection import camera_rays_to_panorama_pixels
from qmapnav.perception.panorama_projection import crop_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import crop_pixels_to_panorama_pixels
from qmapnav.perception.panorama_projection import make_perspective_geometry
from qmapnav.perception.panorama_projection import panorama_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import panorama_pixels_to_crop_pixels
from qmapnav.perception.panorama_projection import PanoramaCameraModel
from qmapnav.perception.panorama_projection import project_crop_box_to_panorama
from qmapnav.perception.perception_worker import PerceptionWorker
from qmapnav.perception.visualisation import draw_crop_layout
from qmapnav.perception.visualisation import save_debug_bundle
from qmapnav.perception.vocabulary import detector_classes_from_task_specification
from qmapnav.perception.yoloe_detector import DetectorDependencyError
from qmapnav.perception.yoloe_detector import YOLOEDetector


__all__ = [
    'CropDetection',
    'DAY4_BASELINE_CHECKPOINT',
    'DAY4_BASELINE_CONFIDENCE',
    'DAY4_BASELINE_CROSS_CROP_IOU',
    'Detection2D',
    'DetectorClass',
    'DetectorDependencyError',
    'DetectorIdentity',
    'GroundingDinoTinyDetector',
    'OpenVocabularyDetector',
    'PanoramaBox',
    'PanoramaCameraModel',
    'PerceptionRequest',
    'PerceptionResult',
    'PerceptionWorker',
    'PerspectiveCropGenerator',
    'PerspectiveCropLayout',
    'PerspectiveGeometry',
    'PerspectiveView',
    'YOLOEDetector',
    'cross_crop_nms',
    'camera_rays_to_crop_pixels',
    'camera_rays_to_panorama_pixels',
    'crop_pixels_to_camera_rays',
    'crop_pixels_to_panorama_pixels',
    'detector_classes_from_task_specification',
    'draw_crop_layout',
    'eight_view_layout',
    'make_perspective_geometry',
    'make_day4_baseline_worker',
    'panorama_pixels_to_camera_rays',
    'panorama_pixels_to_crop_pixels',
    'panorama_box_iou',
    'panorama_box_intersection_over_smaller',
    'project_crop_detections',
    'project_crop_box_to_panorama',
    'save_debug_bundle',
]
