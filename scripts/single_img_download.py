# single_img_download.py

# this script is used to process and download sample data, and some of this functions are used
# to preprocess data with opens pose

import os
import sys
import cv2
import numpy as np

# ---------- CONFIG ----------
OPENPOSE_ROOT = '/home/bohdan/openpose'
CONF_THRESHOLD = 0.1
NUM_PEOPLE_MAX = 5
SAVE_WIDTH = 900
# ----------------------------

# --- Import OpenPose (once) ---
OPENPOSE_PY = os.path.join(OPENPOSE_ROOT, 'build', 'python')
if OPENPOSE_PY not in sys.path:
    sys.path.append(OPENPOSE_PY)

try:
    from openpose import pyopenpose as op

except Exception as e:
    raise RuntimeError(
        'Failed to import OpenPose. '
        'Make sure OPENPOSE_ROOT points to the OpenPose build root and that the Python API is installed.'
    ) from e


BODY25_HEAD = {
    "nose": 0,
    "reye": 15, "leye": 16,
    "rear": 17, "lear": 18,
    "neck": 1
}


# ---------- Geometry / OpenPose utilities ----------
def _compute_bbox(single_kps, conf_thresh):
    valid = single_kps[:, 2] > conf_thresh
    if not valid.any():
        return None
    xs = single_kps[valid, 0]
    ys = single_kps[valid, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_bbox_and_labels(img, kps, conf_thresh):
    bboxes = []
    if kps is None or len(kps) == 0:
        return bboxes
    for i in range(kps.shape[0]):
        bb = _compute_bbox(kps[i], conf_thresh)
        bboxes.append(bb)
        if bb is not None:
            x1, y1, x2, y2 = bb
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(img, f'Person {i+1}', (x1, max(10, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return bboxes


def _person_center(single_kps, img_w, img_h, conf_thresh, fallback_bbox=None):
    conf = single_kps[:, 2]
    valid = conf > conf_thresh
    if valid.any():
        cx = float(single_kps[valid, 0].mean())
        cy = float(single_kps[valid, 1].mean())
        return cx, cy
    if fallback_bbox is not None:
        x1, y1, x2, y2 = fallback_bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return img_w / 2.0, img_h / 2.0


def _head_abs_xy(single_kps, W, H, conf_th):
    conf = single_kps[:, 2] > conf_th

    if conf[BODY25_HEAD["nose"]]:
        j = BODY25_HEAD["nose"]
        return float(single_kps[j, 0]), float(single_kps[j, 1])

    head_candidates = [BODY25_HEAD["reye"], BODY25_HEAD["leye"],
                       BODY25_HEAD["rear"], BODY25_HEAD["lear"]]
    pts = [(float(single_kps[j, 0]), float(single_kps[j, 1])) for j in head_candidates if conf[j]]
    if pts:
        xs, ys = zip(*pts)
        return float(np.mean(xs)), float(np.mean(ys))

    if conf[BODY25_HEAD["neck"]]:
        j = BODY25_HEAD["neck"]
        return float(single_kps[j, 0]), float(single_kps[j, 1])

    return None


def _nearest_head_dists_abs(kps_abs, W, H, conf_th):
    """
    kps_abs: (num_people, 25, 3) in pixels
    Returns np.array (NUM_PEOPLE_MAX,) — distances to the nearest head,
    normalized by the frame diagonal. For empty/padded people -> 0.
    """
    diag = np.hypot(W, H)
    dists = np.zeros((NUM_PEOPLE_MAX,), dtype=np.float32)

    if kps_abs is None or len(kps_abs) == 0:
        return dists

    P = min(kps_abs.shape[0], NUM_PEOPLE_MAX)
    heads = [_head_abs_xy(kps_abs[i], W, H, conf_th) for i in range(P)]

    for i in range(P):
        hi = heads[i]
        if hi is None:
            dists[i] = 0.0
            continue
        xi, yi = hi
        best = None
        for j in range(P):
            if j == i:
                continue
            hj = heads[j]
            if hj is None:
                continue
            xj, yj = hj
            d = np.hypot(xi - xj, yi - yj)
            if best is None or d < best:
                best = d
        dists[i] = (best / diag) if best is not None else 0.0

    # if there are fewer than NUM_PEOPLE_MAX people — the rest are zeros (already so)
    return dists


def _normalize_xy(x, y, cx, cy, W, H):
    return (x - cx) / float(W), (y - cy) / float(H)


def _preprocess_image(opWrapper, img_path, conf_threshold=0.1):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f'Failed to open image {img_path}')
    h, w = img.shape[:2]
    if w > SAVE_WIDTH:
        scale = SAVE_WIDTH / w
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    datum = op.Datum()
    datum.cvInputData = img
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)

    result = datums[0]
    out = result.cvOutputData.copy() if hasattr(result, 'cvOutputData') else img.copy()
    kps = result.poseKeypoints  # shape: (num_people, 25, 3) or None

    kps5 = np.zeros((NUM_PEOPLE_MAX, 25, 5), dtype=np.float32)

    if kps is None or len(kps) == 0:
        # Nobody detected → return empty annotation (zeros)
        return out, kps5

    bboxes = _draw_bbox_and_labels(out, kps, conf_threshold)

    dists = _nearest_head_dists_abs(kps, W, H, conf_threshold)

    N = min(kps.shape[0], NUM_PEOPLE_MAX)
    for i in range(N):
        single = kps[i]
        bbox = bboxes[i] if i < len(bboxes) else None
        cx, cy = _person_center(single, W, H, conf_threshold, fallback_bbox=bbox)
        for j in range(25):
            x, y, c = single[j]
            if c <= conf_threshold:
                kps5[i, j, :4] = (0, 0, 0, 0)
            else:
                nx, ny = _normalize_xy(x, y, cx, cy, W, H)
                kps5[i, j, 0] = nx  # nx
                kps5[i, j, 1] = ny  # ny
                kps5[i, j, 2] = c   # conf
                kps5[i, j, 3] = 1.0 # vis_flag

            # 5th channel — same scalar for all 25 keypoints of this person
        kps5[i, :, 4] = dists[i]
    return out, kps5


# ---------- Public function for import ----------
def process_image(input_path):
    """
    Processes a single image with OpenPose and returns:
      - annotated_img: numpy.ndarray (BGR), annotated image
      - kps4: np.ndarray of shape (NUM_PEOPLE_MAX, 25, 4): (nx, ny, conf, vis_flag)

    input_path can be:
      - full path to an image file, or
      - path to a directory that contains ONE image (the first .jpg/.jpeg/.png will be taken)
    """
    # 1) If a directory is passed — find an image file inside it
    img_path = input_path
    if os.path.isdir(input_path):
        imgs = [f for f in os.listdir(input_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            raise RuntimeError(f'No .jpg/.jpeg/.png found in directory {input_path}')
        # take the first found
        img_path = os.path.join(input_path, sorted(imgs)[0])

    if not os.path.isfile(img_path):
        raise RuntimeError(f'Path is not an image file: {img_path}')

    # 2) OpenPose settings (without saving to disk)
    params = dict()
    params['model_folder'] = os.path.join(OPENPOSE_ROOT, 'models')
    params['hand'] = False
    params['face'] = False
    params['net_resolution'] = '-1x144'
    params['num_gpu'] = 1
    params['num_gpu_start'] = 0
    params['number_people_max'] = NUM_PEOPLE_MAX
    params['disable_multi_thread'] = True

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    annotated, kps4 = _preprocess_image(opWrapper, img_path, conf_threshold=CONF_THRESHOLD)
    return annotated, kps4


# ---------- CLI (not executed on import) ----------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to an image or a directory with a single image')
    args = parser.parse_args()

    img, npz = process_image(args.input_path)
    print('Done. img shape:', img.shape, '| kps5 shape:', npz.shape)
