"""
Day/Night Scene Recognition Algorithm

Analyzes brightness distribution and edge characteristics within a sky ROI
to classify driving scenes as Day, Night, or Unknown.
Includes a temporal stabilization layer to overcome single-frame limitations.

Pipeline:
    1. ROI extraction and Luminance (Y) channel conversion
    2. Nearest Neighbor resizing (computational efficiency)
    3. Step 1: Histogram-based primary classification (Day/Night/Unknown)
    4. Step 2: Sobel edge + morphological dilation based secondary night
               classification (executed only when Step 1 result is Unknown)
    5. Temporal stabilization: Buffer voting + Hysteresis + Minimum duration lock
"""

import numpy as np
import cv2
from collections import deque
from typing import Tuple, Dict, Any, Optional


class SceneAnalyzer:
    """
    Day/Night scene classification engine.

    All thresholds and ROI ratios are injected externally via a configuration
    dictionary. The run() method returns the classification result along with
    optional debug images for intermediate processing stages.

    Attributes:
        current_state: Currently confirmed scene state (DAY / NIGHT / UNKNOWN).
        state_lock_counter: Consecutive frames the current state has been held.
        vote_buffer: Sliding window storing raw results of recent N frames.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize analyzer with configuration parameters.

        Computes pixel-level ROI coordinates from ratio-based config values,
        allocates fixed kernels and weights, and resets all temporal state.

        Args:
            cfg: Parameter dictionary loaded from YAML configuration file.
        """
        self.cfg = cfg

        # Image size and ROI calculation (ratio → pixel)
        vw = cfg["video_width"]
        vh = cfg["video_height"]
        iz = cfg["image_zoom_out_ratio"]

        self.image_width = int(vw / iz)
        self.image_height = int(vh / iz)

        self.roi_x = int(vw * cfg["roi_x_ratio"])
        self.roi_y = int(vh * cfg["roi_y_ratio"])
        self.roi_width = int(vw * cfg["roi_width_ratio"])
        self.roi_height = int(vh * cfg["roi_height_ratio"])

        rz = cfg["roi_zoom_out_ratio"]
        self.sky_resized_w = self.roi_width // rz
        self.sky_resized_h = self.roi_height // rz

        # Fixed kernels and weights (allocated once at init)
        # BT.601 luminance coefficients: Y = 0.299R + 0.587G + 0.114B
        # Ordered as BGR to match OpenCV default channel ordering
        self.y_weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)

        # 5x5 structuring element for morphological dilation
        self.dilate_kernel = np.ones((5, 5), dtype=np.uint8)

        # Temporal stabilization state
        fps = cfg.get("fps", 30.0)
        min_dur = cfg.get("min_duration_sec", 2.0)

        self.min_frames = int(fps * min_dur)
        self.current_state: int = cfg["flag_unknown"]
        self.state_lock_counter: int = 0

        # Hysteresis exit thresholds (stricter than entry to prevent flipping)
        self.day_exit_thresh: float = cfg.get("day_exit_thresh", 0.15)
        self.night_exit_thresh: float = cfg.get("night_exit_thresh", 0.70)

        # Sliding window for majority voting over recent frames
        vote_win = cfg.get("vote_window", 5)
        self.vote_buffer: deque = deque(maxlen=vote_win)

        self.show_debug: bool = cfg.get("show_debug_windows", False)

    @property
    def resized_size(self) -> Tuple[int, int]:
        """Analysis image dimensions after zoom-out resize (width, height)."""
        return self.image_width, self.image_height

    def run(self, frame: np.ndarray) -> Tuple[int, Optional[Dict[str, np.ndarray]]]:
        """
        Perform day/night classification on a single frame.

        Processing sequence:
            A. ROI crop → Y-channel conversion → NN resize
            B. Step 1: Histogram-based primary classification
            C. Step 2: Edge+dilation secondary classification (conditional)
            D. Raw classification from scores
            E. Buffer voting for noise filtering
            F. Hysteresis + Duration Lock for temporal consistency

        Args:
            frame: BGR image pre-resized to resized_size dimensions.

        Returns:
            result: Final state flag (flag_day / flag_night / flag_unknown).
            debug_images: Intermediate images dict when show_debug=True, else None.
        """
        c = self.cfg
        debug_images: Optional[Dict[str, np.ndarray]] = {} if self.show_debug else None

        # A. ROI extraction and Y-channel conversion
        roi = frame[
            self.roi_y:self.roi_y + self.roi_height,
            self.roi_x:self.roi_x + self.roi_width,
        ]
        sky_y = np.dot(roi.astype(np.float32), self.y_weights).astype(np.uint8)

        # B. Nearest Neighbor resize
        sky_resized = cv2.resize(
            sky_y, (self.sky_resized_w, self.sky_resized_h),
            interpolation=cv2.INTER_NEAREST,
        )

        if debug_images is not None:
            debug_images["sky_color"] = roi.copy()
            debug_images["sky_y"] = sky_y.copy()
            debug_images["sky_resized"] = sky_resized.copy()

        # C. Step 1: Histogram scores
        day_score, night_score = self._get_scores_step1(sky_resized)

        # D. Step 2: Conditional secondary classification
        night_score2 = 0.0
        if day_score <= c["min_day_ratio_step1"] and \
           night_score <= c["min_night_ratio_step1"]:
            candidate, edge_binary, dilated = self._make_candidate_step2(sky_resized)
            night_score2 = self._get_night_score_step2(candidate)

            if debug_images is not None:
                debug_images["edge"] = edge_binary
                debug_images["dilated"] = dilated
                debug_images["candidate"] = candidate

        # E. Raw classification
        raw_result = self._classify_raw(day_score, night_score, night_score2)

        # F. Buffer voting
        self.vote_buffer.append(raw_result)
        filtered_result = max(set(self.vote_buffer), key=self.vote_buffer.count)

        # G. Temporal consistency
        final_result = self._apply_temporal_consistency(
            filtered_result, day_score, night_score
        )

        return final_result, debug_images

    # =====================================================================
    # Raw Classification
    # =====================================================================

    def _classify_raw(self, day_s: float, night_s: float, night2_s: float) -> int:
        """
        Score-based raw scene classification without temporal stabilization.

        Priority: DAY(Step1) > NIGHT(Step1) > NIGHT(Step2) > UNKNOWN

        Args:
            day_s: Step 1 day pixel ratio.
            night_s: Step 1 night pixel ratio.
            night2_s: Step 2 night pixel ratio (edge-excluded region).

        Returns:
            Raw classification flag.
        """
        c = self.cfg
        if day_s > c["min_day_ratio_step1"]:
            return c["flag_day"]
        if night_s > c["min_night_ratio_step1"]:
            return c["flag_night"]
        if night2_s > c["min_night_ratio_step2"]:
            return c["flag_night"]
        return c["flag_unknown"]

    def _apply_temporal_consistency(
        self, new_result: int, day_s: float, night_s: float
    ) -> int:
        """
        Enforce temporal consistency via three combined mechanisms:
          1. Duration Lock: Block transitions for min_frames after change.
          2. Hysteresis: Exit threshold stricter than entry threshold.
          3. UNKNOWN bypass: Allow immediate transition from UNKNOWN state.

        Args:
            new_result: Filtered result after buffer voting.
            day_s: Current frame day score for exit evaluation.
            night_s: Current frame night score for exit evaluation.

        Returns:
            Final confirmed state flag.
        """
        c = self.cfg
        self.state_lock_counter += 1

        # Within minimum duration: always maintain previous state
        if self.state_lock_counter < self.min_frames:
            return self.current_state

        # Hysteresis check on transition attempt (skip for UNKNOWN)
        if new_result != self.current_state and \
           self.current_state != c["flag_unknown"]:
            is_day_exiting = (
                self.current_state == c["flag_day"]
                and day_s < self.day_exit_thresh
            )
            is_night_exiting = (
                self.current_state == c["flag_night"]
                and night_s < self.night_exit_thresh
            )
            if not (is_day_exiting or is_night_exiting):
                return self.current_state

        # Transition confirmed: update state and reset counter
        if new_result != self.current_state:
            self.current_state = new_result
            self.state_lock_counter = 0

        return self.current_state

    # =====================================================================
    # Step 1: Histogram-Based Classification
    # =====================================================================

    def _get_scores_step1(self, img: np.ndarray) -> Tuple[float, float]:
        """
        Compute day/night scores from luminance histogram.

        Args:
            img: Resized Y-channel grayscale image.

        Returns:
            Tuple of (day_score, night_score) as pixel ratios in [0.0, 1.0].
        """
        c = self.cfg
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        total = img.size

        night_sum = hist[
            c["night_min_intensity_step1"]:c["night_max_intensity_step1"] + 1
        ].sum()
        day_sum = hist[
            c["day_min_intensity_step1"]:c["day_max_intensity_step1"] + 1
        ].sum()

        return day_sum / total, night_sum / total

    # =====================================================================
    # Step 2: Edge + Morphology Based Secondary Classification
    # =====================================================================

    def _make_candidate_step2(self, gray: np.ndarray):
        """
        Generate candidate image by removing light source regions.

        Pipeline: Sobel → L1 magnitude → binarize → 5x5 dilate → mask out.
        Only background sky pixels (where dilated mask == 0) retain luminance.

        Args:
            gray: Resized Y-channel image.

        Returns:
            candidate: Background-only image for night score recalculation.
            edge_binary: Binary edge map for debug visualization.
            dilated: Dilated edge map for debug visualization.
        """
        c = self.cfg

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.abs(sobel_x) + np.abs(sobel_y)

        edge_binary = (magnitude > c["min_edge_magnitude_step2"]).astype(np.uint8) * 255
        dilated = cv2.dilate(edge_binary, self.dilate_kernel)
        candidate = np.where(dilated == 0, gray, np.uint8(0))

        return candidate, edge_binary, dilated

    def _get_night_score_step2(self, candidate: np.ndarray) -> float:
        """
        Compute night score from candidate (background-only) image.

        Only non-zero pixels are included in histogram calculation.
        Returns 0.0 if no valid background region exists.

        Args:
            candidate: Background sky image from _make_candidate_step2.

        Returns:
            Night pixel ratio in [0.0, 1.0].
        """
        c = self.cfg
        mask = candidate != 0
        object_count = mask.sum()

        if object_count == 0:
            return 0.0

        objects = candidate[mask]
        hist = cv2.calcHist([objects], [0], None, [256], [0, 256]).flatten()

        night_sum = hist[
            c["night_min_intensity_step2"]:c["night_max_intensity_step2"]
        ].sum()

        return night_sum / object_count