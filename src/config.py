"""Central configuration for the pothole detection pipeline.

Every tunable lives here — modules must import from config and never hardcode
values (owner convention). Paths are pathlib.Path derived from this file's
location, so CLIs work no matter which working directory they run from.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_VIDEOS_DIR: Path = DATA_DIR / "sample_videos"
GPS_LOGS_DIR: Path = DATA_DIR / "gps_logs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Pipeline (shared by baseline.py now, detect.py later)
# ---------------------------------------------------------------------------
RESIZE_WIDTH: int = 640       # fixed processing width; keeps area thresholds stable
FRAME_SKIP: int = 2           # process every Nth frame (CPU speed)
CONF_THRESHOLD: float = 0.25  # YOLO confidence cutoff (detect.py; baseline has none)

# ---------------------------------------------------------------------------
# Severity (used by severity.py in a later step; bbox area / frame area.
# Initial guesses — calibrate visually in a notebook before trusting them.)
# ---------------------------------------------------------------------------
SEVERITY_SMALL_FRAC: float = 0.004  # below this fraction of frame area -> small
SEVERITY_LARGE_FRAC: float = 0.015  # above this fraction -> large; medium = in between

# ---------------------------------------------------------------------------
# Classical baseline detector parameters
# ---------------------------------------------------------------------------
# Region of interest: the road region starts at this fraction of frame height.
# 0.5 = lower half only (dashcam horizon sits mid-frame; potholes are on the road).
ROI_TOP_FRAC: float = 0.5

BLUR_KERNEL: tuple[int, int] = (5, 5)   # Gaussian kernel; sizes must be odd
CANNY_LOW: int = 50                     # weak-edge threshold
CANNY_HIGH: int = 150                   # strong-edge threshold (classic 1:3 ratio)
CLOSE_KERNEL: tuple[int, int] = (7, 7)  # ellipse kernel for morphological close

MIN_AREA_FRAC: float = 0.001            # min contour area as a fraction of frame area
MAX_AREA_FRAC: float = 0.25             # reject huge blobs (shadows, road texture)
ASPECT_MIN: float = 0.33                # bbox aspect ratio (w/h) range to keep
ASPECT_MAX: float = 3.0
MIN_EXTENT: float = 0.3                 # contour area / bbox area; rejects spindly shapes
