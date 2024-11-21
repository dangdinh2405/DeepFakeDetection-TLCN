import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR
from .utils import slowfast

from .facexray_detector import FaceXrayDetector
from .capsule_net_detector import CapsuleNetDetector
from .ucf_detector import UCFDetector
from .videomae_detector import VideoMAEDetector
