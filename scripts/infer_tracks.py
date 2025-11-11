import sys
import os
import cv2
from pathlib import Path
import numpy as np

from src.boxing_project.tracking.tracking_debug import print_tracking_results
from src.boxing_project.tracking.tracker import MultiObjectTracker
from src.boxing_project.tracking import DEFAULT_TRACKING_CONFIG_PATH  # ← беремо шлях до YAML




OPENPOSE_ROOT = '/home/bohdan/openpose'
SAVE_WIDTH = 700
IMAGE_DIR_PATH = '/home/bohdan/Документи/Samples/Images/boxing/Processed_data/output/GH098416_00051'

OPENPOSE_PY = os.path.join(OPENPOSE_ROOT, 'build', 'python')
if OPENPOSE_PY not in sys.path:
    sys.path.append(OPENPOSE_PY)

try:
    from openpose import pyopenpose as op
except ImportError:
    print("Please be sure that you installed openpose correctly and set a correct path to it")


# ---- OpenPose params (можна теж колись винести в YAML, але поки лишаємо тут) ----
params = dict()
params['model_folder'] = os.path.join(OPENPOSE_ROOT, 'models')
params['hand'] = False
params['face'] = False
params['net_resolution'] = '-1x256'
params['num_gpu'] = 1
params['num_gpu_start'] = 0
params['render_pose'] = 0        # do not render (saves VRAM)
params['disable_blending'] = True
params['number_people_max'] = 5
params['disable_multi_thread'] = True

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def _preprocess_image(opWrapper, img_path, conf_threshold=0.1, return_img=False):
    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f'Failed to open image {img_path}')
    h, w = img.shape[:2]
    if w > SAVE_WIDTH:
        scale = SAVE_WIDTH / w
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    datum = op.Datum()
    datum.cvInputData = img
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)

    result = datums[0]

    if return_img:
        return result, img
    else:
        return result


def bbox(kps, conf_threshold=0.1):
    """
    Compute bbox over keypoints with confidence > conf_threshold.
    """
    valid = kps[:, 2] > conf_threshold
    if not valid.any():
        return None

    xs = kps[valid, 0]
    ys = kps[valid, 1]

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return x1, y1, x2, y2


def return_processed_frame(result, tracker, unprocessed_img=None):
    if unprocessed_img is not None:
        frame = unprocessed_img
    else:
        frame = result.cvOutputData

    if result.poseKeypoints is None:
        print('No person detected')
        return frame, {"matches": [], "cost_matrix": np.zeros((0, 0)), "active_tracks": []}

    # (N, 25, 3)
    kps = result.poseKeypoints

    # adapt OpenPose output to tracker API
    people = [{"keypoints": kps[i]} for i in range(len(kps))]
    log = tracker.update_with_openpose(people)

    # use threshold from tracker config instead of hardcoded value
    conf_th = tracker.cfg.min_kp_conf

    for track_id, det_idx in log["matches"]:
        bb = bbox(kps[det_idx], conf_threshold=conf_th)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return frame, log  # return frame and log


def display(opWrapper, images_path, tracker, num=1):
    frames = []
    count = 0

    # get "show" flag from YAML
    raw_cfg = tracker.get_config_dict() or {}
    show_flag = raw_cfg.get("tracking", {}).get("show", True)

    for i, img_path in enumerate(images_path):

        if show_flag:

            # to clearly divide a different frames
            print("=" * 140)
            print(f"         PRE TRACKING RESULTS: {i+1}")
            print("=" * 140)

        result, img = _preprocess_image(opWrapper, img_path, return_img=True)
        frame, log = return_processed_frame(result, tracker, unprocessed_img=img)

        if show_flag:  # only print logs and show images if show=true
            from src.boxing_project.tracking.tracking_debug import print_tracking_results
            print_tracking_results(log, i)

        frames.append(frame)
        count += 1

        if count == num and show_flag:
            try:
                max_h = max(f.shape[0] for f in frames)
                aligned_frames = []
                for f in frames:
                    h, w = f.shape[:2]
                    if h < max_h:
                        pad_top = (max_h - h) // 2
                        pad_bottom = max_h - h - pad_top
                        f_aligned = cv2.copyMakeBorder(f, pad_top, pad_bottom, 0, 0,
                                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        aligned_frames.append(f_aligned)
                    else:
                        aligned_frames.append(f)

                combined_frame = cv2.hconcat(aligned_frames)

                cv2.imshow(f"Tracking ({num} frames)", combined_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Error while frame merging: {e}")
                print("Skip this frame.")

            frames = []
            count = 0


# --------- CREATE TRACKER FROM YAML CONFIG ---------

# Just use YAML-based tracker config (tracking.yaml via DEFAULT_TRACKING_CONFIG_PATH)
tracker = MultiObjectTracker(config_path=DEFAULT_TRACKING_CONFIG_PATH)

images_path = sorted(
    [p for p in Path(IMAGE_DIR_PATH).iterdir()
     if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
)

display(opWrapper, images_path, tracker, num=1)
