"""
Day/Night Scene Recognition Entry Point (Adaptive Display)

Features adaptive font scaling that maintains readability at any display_scale.
Font size uses non-linear scaling with a minimum readable floor to prevent
text from becoming illegible on small output windows.
"""

import sys
import time
import argparse
import cv2

from utils import (
    load_config,
    resolve_input_files,
    show_debug_images,
    destroy_debug_windows,
    compute_final_statistics,
    TTFTextRenderer,
)
from scene_analyzer import SceneAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for config file path only."""
    parser = argparse.ArgumentParser(
        description="Day/Night Scene Recognition (Adaptive Display)"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    return parser.parse_args()


def compute_adaptive_font_size(display_scale: float, base_size: int = 32) -> int:
    """
    Compute font size that adapts to display_scale while preserving readability.

    Uses sqrt-based non-linear scaling so font shrinks slower than the window.
    A minimum floor ensures text remains legible even at very small scales.

    Args:
        display_scale: Current output window scale factor (0.1 ~ 2.0).
        base_size: Reference font size at scale 1.0.

    Returns:
        Adaptively scaled font size in pixels (always >= 18).
    """
    import math
    # Non-linear: sqrt makes font shrink slower than window
    # At scale=0.5 → sqrt(0.5)=0.707 → font is 70% not 50% of base
    # At scale=0.25 → sqrt(0.25)=0.5 → font is 50% not 25% of base
    scaled = base_size * math.sqrt(max(display_scale, 0.1))
    # Minimum readable floor: 18px ensures TTF anti-aliasing still works
    return max(18, int(round(scaled)))


def main():
    """Main processing loop driven entirely by YAML configuration."""
    args = parse_args()
    cfg = load_config(args.config)

    try:
        video_files = resolve_input_files(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}")
        sys.exit(-1)

    # Display scale clamped to valid range
    display_scale = max(0.1, min(2.0, cfg.get("display_scale", 1.0)))
    print(f"[Display] Output scale: {display_scale:.2f}x")

    # ★ Adaptive font sizing: readable at any scale
    adaptive_font_size = compute_adaptive_font_size(display_scale)
    renderer = TTFTextRenderer(font_size=adaptive_font_size)
    print(f"[Display] Adaptive font size: {adaptive_font_size}px "
          f"(base=32, scale={display_scale:.2f})")

    stats = {"day": 0, "night": 0, "unknown": 0}
    flag_map = {
        cfg["flag_day"]:     ("SCENE: DAY",     (0, 0, 255),   "day"),
        cfg["flag_night"]:   ("SCENE: NIGHT",   (0, 255, 0),   "night"),
        cfg["flag_unknown"]: ("SCENE: UNKNOWN", (255, 0, 0),   "unknown"),
    }

    show_main = cfg.get("show_process", True)
    show_debug = cfg.get("show_debug_windows", False)
    draw_roi = cfg.get("draw_roi_on_main", True)

    for filepath in video_files:
        analyzer = SceneAnalyzer(cfg)
        w, h = analyzer.resized_size
        algo_fps = 0.0

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"[Warning] Cannot open: {filepath}")
            continue

        filename = filepath.replace("\\", "/").split("/")[-1]
        print(f"\nProcessing: {filename}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            # Pure algorithm timing (unaffected by display_scale)
            t_start = time.perf_counter()
            result, debug_images = analyzer.run(resized)
            t_end = time.perf_counter()

            dt = t_end - t_start
            if dt > 0:
                algo_fps = 0.9 * algo_fps + 0.1 * (1.0 / dt)

            label, color, key = flag_map[result]
            stats[key] += 1

            if show_main:
                # Draw OSD and ROI on ORIGINAL resolution frame
                display_frame = frame.copy()
                orig_h, orig_w = frame.shape[:2]

                if draw_roi:
                    scale_x = orig_w / w
                    scale_y = orig_h / h
                    rx = int(analyzer.roi_x * scale_x)
                    ry = int(analyzer.roi_y * scale_y)
                    rw = int(analyzer.roi_width * scale_x)
                    rh = int(analyzer.roi_height * scale_y)
                    cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

                # Text positions proportional to original resolution
                line1_y = int(25 * (orig_h / 720.0))
                line2_y = int(65 * (orig_h / 720.0))

                renderer.put_text(display_frame, f"Algo FPS: {algo_fps:.1f}", (30, line1_y), (255, 0, 255))
                renderer.put_text(display_frame, label, (30, line2_y), color)

                # Apply display_scale ONLY for window output
                if abs(display_scale - 1.0) > 0.01:
                    out_w = int(orig_w * display_scale)
                    out_h = int(orig_h * display_scale)
                    output_frame = cv2.resize(display_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    output_frame = display_frame

                cv2.imshow("Result Video", output_frame)

            if show_debug:
                show_debug_images(debug_images)

            if show_main or show_debug:
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif k == 13:
                    break
                elif k == 32:
                    cv2.waitKey(0)

        cap.release()
        if show_debug:
            destroy_debug_windows()

        if show_main:
            print(f"  Day:{stats['day']} Night:{stats['night']} Unknown:{stats['unknown']}")

    cv2.destroyAllWindows()
    print(compute_final_statistics(stats))


if __name__ == "__main__":
    main()