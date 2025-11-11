import sys
import os
import cv2
from pathlib import Path
import numpy as np

from src.boxing_project.tracking.tracking_debug import print_tracking_results
from src.boxing_project.tracking.tracker import MultiObjectTracker, TrackerConfig
from src.boxing_project.tracking.matcher import MatchConfig



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


params = dict()
params['model_folder'] = os.path.join(OPENPOSE_ROOT, 'models')
params['hand'] = False
params['face'] = False
params['net_resolution'] = '-1x256'
params['num_gpu'] = 1
params['num_gpu_start'] = 0
params['render_pose'] = 0        # не рендерити (суттєво економить VRAM)
params['disable_blending'] = True
params['number_people_max'] = 5
params['disable_multi_thread'] = True

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def _preprocess_image(opWrapper, img_path, conf_threshold=0.1, return_img=False):
    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f'Не вдалося відкрити зображення {img_path}')
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

    valid = kps[:, 2] > conf_threshold

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

    else:
        kps = result.poseKeypoints # shape (N, 25, 3)


        people = [{"keypoints": kps[i]} for i in range(len(kps))]
        log = tracker.update_with_openpose(people)



        for track_id, det_idx in log["matches"]:
            x1, y1, x2, y2 = bbox(kps[det_idx])
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return frame, log # return frame, and log



def display(opWrapper, images_path, tracker, num=1):
    frames = []
    count = 0


    for i, img_path in enumerate(images_path):

        print("=" * 140)
        print(f"         РЕЗУЛЬТАТИ ПРЕТРЕКІНГУ: {i+1}")
        print("=" * 140)

        result, img = _preprocess_image(opWrapper, img_path, return_img=True)
        
        frame, log = return_processed_frame(result, tracker, unprocessed_img=img)
        frame, log = return_processed_frame(result, tracker, unprocessed_img=img)




        print_tracking_results(log, i)


        frames.append(frame)
        count += 1

        if count == num:
            # --- ОСЬ ГОЛОВНІ ЗМІНИ ---

            # 1. Вирівнюємо висоту всіх кадрів (додаємо чорні смуги, якщо треба)
            # hconcat вимагає, щоб усі зображення мали однакову висоту
            try:
                max_h = max(f.shape[0] for f in frames)
                aligned_frames = []
                for f in frames:
                    h, w = f.shape[:2]
                    if h < max_h:
                        pad_top = (max_h - h) // 2
                        pad_bottom = max_h - h - pad_top
                        f_aligned = cv2.copyMakeBorder(f, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT,
                                                       value=(0, 0, 0))
                        aligned_frames.append(f_aligned)
                    else:
                        aligned_frames.append(f)

                # 2. "Склеюємо" всі кадри в один горизонтально
                combined_frame = cv2.hconcat(aligned_frames)

                # 3. Показуємо ОДНЕ об'єднане вікно
                cv2.imshow(f"Tracking ({num} frames)", combined_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Помилка при об'єднанні кадрів: {e}")
                print("Пропускаємо цю пачку.")

            frames = []
            count = 0

# параметри як у tracking.yaml (по-старинці)
FPS = 60.0
DT = 1.0 / FPS
PROCESS_VAR = 3.0
MEASURE_VAR = 9.0
GATE_CHI2 = 9.21
ALPHA_MOTION = 0.2
COST_MAX = 880.0
MAX_AGE = 10
MIN_HITS = 3

def make_tracker():
    match_cfg = MatchConfig(
        alpha=ALPHA_MOTION,
        chi2_gating=GATE_CHI2,
        large_cost=COST_MAX,
        min_kp_conf=0.05,
        keypoint_weights=None
    )
    tcfg = TrackerConfig(
        dt=DT,
        process_var=PROCESS_VAR,
        measure_var=MEASURE_VAR,
        p0=1e3,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        match=match_cfg,
        min_kp_conf=0.05,
        expect_body25=True
    )
    return MultiObjectTracker(tcfg)


tracker = make_tracker()


images_path = sorted([p for p in Path(IMAGE_DIR_PATH).iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


display(opWrapper, images_path, tracker, num=1)





