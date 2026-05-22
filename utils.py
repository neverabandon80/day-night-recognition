"""
Utility functions for configuration loading, input path resolution,
debug visualization, statistics computation, and TTF text rendering.
"""

import os
import glob
import yaml
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def load_config(config_path: str) -> dict:
    """
    Load algorithm parameters from a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_input_files(cfg: dict) -> List[str]:
    """
    Resolve input video file list from config settings.

    Behavior:
        - If input_path is a file: return it as a single-element list.
        - If input_path is a directory: glob for files matching video_extension.
        - Supports both absolute and relative paths (relative to CWD).

    Args:
        cfg: Configuration dictionary containing 'input_path' and 'video_extension'.

    Returns:
        Sorted list of video file paths.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If no matching video files are found in the directory.
    """
    input_path = cfg.get("input_path", "")
    ext = cfg.get("video_extension", ".mp4")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Single file mode
    if os.path.isfile(input_path):
        print(f"[Input] Single file mode: {input_path}")
        return [input_path]

    # Directory mode: search for matching extensions
    pattern = os.path.join(input_path, f"*{ext}")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(
            f"No '{ext}' files found in directory: {input_path}\n"
            f"Checked pattern: {pattern}"
        )

    print(f"[Input] Directory mode: {len(files)} '{ext}' file(s) found in {input_path}")
    return files


# Mapping from internal debug image keys to display window names
DEBUG_WINDOW_MAP = {
    "sky_color": "Sky_Color_Video",
    "sky_y": "Sky_Video",
    "sky_resized": "Sky_Resized_Video",
    "edge": "Sky_Resized_Edge_Video",
    "dilated": "Sky_Resized_Dilated_Video",
    "candidate": "Sky_Resized_Candidate_Video",
}


def show_debug_images(debug_images: Optional[Dict[str, object]]) -> None:
    """
    Display intermediate processing stage images in separate OpenCV windows.

    Only images present in the provided dictionary are displayed.
    Missing keys (e.g., Step 2 images when Step 2 was skipped) are silently ignored.

    Args:
        debug_images: Dictionary mapping stage names to numpy arrays,
                      or None if debug mode is disabled.
    """
    if debug_images is None:
        return

    for key, window_name in DEBUG_WINDOW_MAP.items():
        img = debug_images.get(key)
        if img is not None:
            cv2.imshow(window_name, img)


def destroy_debug_windows() -> None:
    """
    Close all debug visualization windows.

    Safely handles cases where windows were never created or already destroyed.
    """
    for window_name in DEBUG_WINDOW_MAP.values():
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def compute_final_statistics(stats: Dict[str, int]) -> str:
    """
    Compute and format final classification statistics.

    Metrics:
        - True Rate (TR): Percentage of dominant class among classified frames.
        - False Rate (FR): Complement of True Rate.
        - Miss Rate (MR): Percentage of UNKNOWN frames among all frames.

    Args:
        stats: Dictionary with keys 'day', 'night', 'unknown' and integer counts.

    Returns:
        Formatted string containing all statistics.
    """
    day_cnt = stats["day"]
    night_cnt = stats["night"]
    unknown_cnt = stats["unknown"]
    total = day_cnt + night_cnt + unknown_cnt
    candidate = day_cnt + night_cnt

    if candidate > 0:
        dominant = max(day_cnt, night_cnt)
        true_rate = (dominant / candidate) * 100.0
        false_rate = 100.0 - true_rate
        miss_rate = (unknown_cnt / total * 100.0) if total > 0 else 0.0
    else:
        true_rate = 0.0
        false_rate = 0.0
        miss_rate = 0.0

    return (
        f"\nDayCnt: {day_cnt}, NightCnt: {night_cnt}, "
        f"UnknownCnt: {unknown_cnt}, CandidateCnt: {candidate}, "
        f"TotalCnt: {total}, TR: {true_rate:.2f}, FR: {false_rate:.2f}, "
        f"MR: {miss_rate:.2f}"
    )


class TTFTextRenderer:
    """
    Renders anti-aliased text using TrueType fonts via PIL/Pillow.
    Falls back to OpenCV's built-in Hershey font if Pillow is unavailable.
    Provides significantly cleaner and more readable OSD text.
    """

    def __init__(self, font_size: int = 28):
        """
        Initialize renderer with preferred system TTF font.

        Args:
            font_size: Pixel height of rendered text.
        """
        self.font_size = font_size
        self.pil_available = False
        self.pil_font = None

        try:
            from PIL import ImageFont

            # Cross-platform clean sans-serif font candidates
            font_candidates = [
                "C:/Windows/Fonts/segoeui.ttf",      # Windows
                "C:/Windows/Fonts/arial.ttf",         # Windows fallback
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Ubuntu
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc", # macOS
            ]

            for font_path in font_candidates:
                if os.path.isfile(font_path):
                    self.pil_font = ImageFont.truetype(font_path, font_size)
                    self.pil_available = True
                    print(f"[TTFRenderer] Loaded font: {font_path}")
                    break

            if not self.pil_available:
                print("[TTFRenderer] No TTF font found. Falling back to OpenCV Hershey.")
        except ImportError:
            print("[TTFRenderer] Pillow not installed. Falling back to OpenCV Hershey.")
            print("              Install via: pip install Pillow")

    def put_text(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """
        Draw text on image at specified position.

        Uses PIL for anti-aliased TTF rendering when available,
        otherwise falls back to cv2.putText with Hershey font.

        Args:
            img: BGR numpy array to draw on (modified in-place).
            text: String to render.
            pos: (x, y) top-left position of text.
            color: BGR color tuple.
            thickness: Stroke thickness (used for fallback; PIL uses font weight).
        """
        if self.pil_available and self.pil_font is not None:
            from PIL import Image, ImageDraw

            # Convert BGR → RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            # PIL uses RGB color order
            rgb_color = (color[2], color[1], color[0])
            draw.text(pos, text, font=self.pil_font, fill=rgb_color)

            # Convert back RGB → BGR in-place
            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # Fallback: OpenCV Hershey font
            scale = self.font_size / 30.0
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)