import os
import sys
import cv2
import numpy as np
from pathlib import Path

"""
This module contains reusable inference components:
- OpenPose initialization
- Image preprocessing
- Keypoint-based bbox extraction
- Frame processing (OpenPose -> tracker -> drawing)
- Visualization loop
"""

op = None  # will be set in init_openpose_from_config()


def init_openpose_from_config(openpose_cfg: dict):
    """
    Initialize OpenPose wrapper from YAML config.
    Returns (op_module, opWrapper).
    """
    global op

    root = os.path.expanduser(openpose_cfg["root"])
    py_path = os.path.join(root, "build", "python")

    if py_path not in sys.path:
        sys.path.append(py_path)

    try:
        from openpose import pyopenpose as op_module
    except Exception as e:
        raise RuntimeError(
            "Failed to import pyopenpose. Check OpenPose installation and the 'root' path."
        ) from e

    op = op_module

    params = dict()
    params["model_folder"] = os.path.join(root, "models")
    params["hand"] = openpose_cfg.get("hand", False)
    params["face"] = openpose_cfg.get("face", False)
    params["net_resolution"] = openpose_cfg.get("net_resolution", "-1x256")
    params["num_gpu"] = openpose_cfg.get("num_gpu", 1)
    params["num_gpu_start"] = openpose_cfg.get("num_gpu_start", 0)
    params["render_pose"] = openpose_cfg.get("render_pose", 0)
    params["disable_blending"] = openpose_cfg.get("disable_blending", True)
    params["number_people_max"] = openpose_cfg.get("number_people_max", 5)
    params["disable_multi_thread"] = openpose_cfg.get("disable_multi_thread", True)

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return op_module, opWrapper



def preprocess_image(opWrapper, img_path: Path, save_width: int, return_img=False):
    """
    Read and resize an image, run OpenPose forward pass.
    Returns Datum (and optionally the resized image).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    h, w = img.shape[:2]
    if w > save_width:
        scale = save_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    datum = op.Datum()
    datum.cvInputData = img

    datums = op.VectorDatum()
    datums.append(datum)

    opWrapper.emplaceAndPop(datums)

    if return_img:
        return datums[0], img
    return datums[0]



def keypoints_to_bbox(kps, conf_th=0.1):
    """
    Compute bounding box over keypoints with confidence > conf_th.
    Returns None if no valid keypoints found.
    """
    valid = kps[:, 2] > conf_th
    if not valid.any():
        return None

    xs, ys = kps[valid, 0], kps[valid, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())



def process_frame(result, tracker, original_img, conf_th):
    """
    Convert OpenPose output to tracker input, update tracker and draw results.
    Returns processed_frame, log_dict.
    """
    frame = original_img.copy()

    if result.poseKeypoints is None:
        return frame, {"matches": [], "cost_matrix": np.zeros((0, 0)), "active_tracks": []}

    kps = result.poseKeypoints
    people = [{"keypoints": kps[i]} for i in range(len(kps))]

    log = tracker.update_with_openpose(people)

    # Precompute all bounding boxes
    bboxes = [keypoints_to_bbox(kps[i], conf_th) for i in range(len(kps))]

    # Draw tracks
    for track_id, det_idx in log["matches"]:
        bb = bboxes[det_idx]
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Draw OpenPose det indices
    for det_idx, bb in enumerate(bboxes):
        if bb is None:
            continue
        x1, y1, _, _ = bb
        cv2.putText(frame, f"OP {det_idx}", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame, log



def visualize_sequence(opWrapper, tracker, images, save_width, merge_n):
    """
    Iterate through images, run inference, print debug info, and visualize batches.
    """
    frames = []
    count = 0

    # Resolve debug level
    show_level = getattr(tracker, "show_level", 1)

    from src.boxing_project.tracking.tracking_debug import (
        print_pre_tracking_results,
        print_tracking_results,
    )

    for idx, path in enumerate(images):
        # use 1-based index for logging: 1, 2, 3, ...
        frame_idx = idx + 1

        if show_level >= 2:
            print_pre_tracking_results(frame_idx)

        result, img = preprocess_image(opWrapper, path, save_width, return_img=True)
        frame, log = process_frame(result, tracker, img, tracker.cfg.min_kp_conf)

        # ---- draw "Frame N" label in the top-right corner ----
        h, w = frame.shape[:2]
        text = f"Frame {frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)
        thickness = 2

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        org = (w - text_w - 10, text_h + 10)

        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        # ------------------------------------------------------

        if show_level >= 1:
            print_tracking_results(log, frame_idx)

        frames.append(frame)
        count += 1

        if count == merge_n and show_level >= 1:
            _show_merged(frames, merge_n)
            frames = []
            count = 0




def _show_merged(frames, n):
    """
    Merge multiple frames horizontally (aligning by height) and show via cv2.imshow.
    """
    max_h = max(f.shape[0] for f in frames)

    aligned = []
    for f in frames:
        h, w = f.shape[:2]
        if h < max_h:
            top = (max_h - h) // 2
            bottom = max_h - h - top
            f = cv2.copyMakeBorder(f, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        aligned.append(f)

    combined = cv2.hconcat(aligned)

    cv2.imshow(f"Tracking ({n} frames)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
