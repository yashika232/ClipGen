"""
Video Synthesis Pipeline Utilities Package
"""

from .logger import setup_logger, TimedLogger
from .file_utils import (
    ensure_directory, 
    check_file_exists, 
    get_file_hash,
    copy_file,
    load_json,
    save_json
)

# Only import video_utils when explicitly needed to avoid cv2 dependency
# from .video_utils import (
#     get_video_info,
#     extract_frames,
#     create_video_from_frames
# )

__version__ = "1.0.0"
__all__ = [
    'setup_logger',
    'TimedLogger', 
    'ensure_directory',
    'check_file_exists',
    'get_file_hash',
    'copy_file',
    'load_json',
    'save_json',
    # Video utils available by explicit import: from utils.video_utils import ...
]
