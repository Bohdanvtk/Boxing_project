import sys
import os
import cv2
from pathlib import Path
import numpy as np

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


def print_tracking_results(log: dict, iteration: int, show_pose_tables: bool = False):
    """
    Красивий вивід результатів трекінгу за кадр:
      - Summary active tracks
      - Cost matrix з коректними Track IDs (row_track_ids)
      - Парні логи по кожному активному треку (motion/pose/final)
      - (опційно) повні таблиці по 25 точках для кожної пари
    """
    print("=" * 40)
    print(f"         РЕЗУЛЬТАТИ ТРЕКІНГУ: {iteration + 1}")
    print("=" * 40)



    active_tracks = log.get("active_tracks", [])
    C = np.array(log.get("cost_matrix", []))
    row_track_ids = log.get("row_track_ids", [])  # <— те, що ми додали в update()

    # 1) ACTIVE TRACKS — коротко
    print(f"\n{'active_tracks':<25}: [{len(active_tracks)} активних треків]")
    for t in active_tracks:
        track_id = t.get('track_id', 'N/A')
        confirmed = t.get('confirmed', False)
        hits = t.get('hits', 0)
        age = t.get('age', 0)
        pos = t.get('pos', (0.0, 0.0))
        x_pos = round(pos[0], 2) if pos else 0.0
        y_pos = round(pos[1], 2) if pos else 0.0
        print(f"  |-> ID {track_id} (H:{hits}/A:{age}): "
              f"CONFIRMED={str(confirmed):<5} | POS=({x_pos}, {y_pos})")

    # 2) COST MATRIX — з коректними row IDs
    print()
    if C.size == 0:
        print(f"{'cost_matrix':<25}: Порожня (не було збігів для порівняння)")
    else:
        rows, cols = C.shape
        print(f"{'cost_matrix':<25}: size={rows}x{cols}")
        print("  | Рядки = Track ID (до апдейту), Стовпці = Detection Index")

        # заголовок
        max_id_len = max((len(str(tid)) for tid in row_track_ids), default=3)
        column_width = 8
        header = "  | " + " " * (max_id_len + 9)
        for j in range(cols):
            header += f"Det {j:^{column_width - 4}}"
        print(header)
        print("  |" + "-" * (len(header) - 3))

        # рядки
        for i in range(rows):
            tid = row_track_ids[i] if i < len(row_track_ids) else "N/A"
            row_str = f"  | Track {str(tid):>{max_id_len}}: "
            for j in range(cols):
                row_str += f"{C[i, j]:{column_width}.2f} "
            print(row_str)

    # 3) ПАРНІ ЛОГИ ПО КОЖНОМУ АКТИВНОМУ ТРЕКУ
    print("\nPAIR LOGS (за треками):")
    for t in active_tracks:
        tid = t["track_id"]
        pairs = t.get("match_log", [])
        if not pairs:
            print(f"  - Track {tid}: (логів немає)")
            continue

        print(f"  - Track {tid}: {len(pairs)} пар(и)")
        for p in pairs:
            reason = p.get("final", {}).get("reason", "ok")
            cost = p.get("final", {}).get("cost", None)
            d_motion = p.get("final", {}).get("components", {}).get("d_motion", None)
            d_pose = p.get("final", {}).get("components", {}).get("d_pose", None)
            det_j = p.get("det_index", None)

            motion = p.get("motion")
            if motion is not None:
                d2 = motion.get("d2", None)
                allowed = motion.get("allowed", None)
            else:
                d2 = None
                allowed = None

            pose = p.get("pose") or {}
            used_count = pose.get("used_count", 0)
            D_pose = pose.get("D_pose", 0.0)

            # короткий рядок по парі
            print(f"     • det={det_j}, reason={reason}, "
                  f"d2={None if d2 is None else round(d2, 4)}, "
                  f"allowed={allowed}, "
                  f"d_motion={None if d_motion is None else round(d_motion, 4)}, "
                  f"used_kps={used_count}, D_pose={round(D_pose, 4)}, "
                  f"final_cost={None if cost is None else round(cost, 4)}")

            # (опційно) повна таблиця по 25 точках
            if show_pose_tables and pose and pose.get("trk_norm") is not None:
                trk_norm = np.array(pose["trk_norm"], dtype=float)
                det_norm = np.array(pose["det_norm"], dtype=float)
                diff = np.array(pose["diff"], dtype=float)
                per_k = np.array(pose["per_k"], dtype=float)
                w_eff = np.array(pose["w_eff"], dtype=float)
                good_mask = np.array(pose["good_mask"], dtype=bool)

                print("       k |    trk_x    trk_y |    det_x    det_y |      dx       dy |   ||d||  |  w_eff | used")
                print("      ---+--------------------+--------------------+------------------+---------+--------+------")
                K = trk_norm.shape[0]
                for k in range(K):
                    tx, ty = trk_norm[k]
                    dx_, dy_ = det_norm[k]
                    ddx, ddy = diff[k]
                    perk = per_k[k]
                    ww = w_eff[k]
                    used = bool(good_mask[k])
                    def fmt(v):
                        return "   nan  " if (v is None or np.isnan(v)) else f"{v:7.4f}"
                    print(f"      {k:2d} | "
                          f"{fmt(tx)} {fmt(ty)} | {fmt(dx_)} {fmt(dy_)} | "
                          f"{fmt(ddx)} {fmt(ddy)} | "
                          f"{'   nan  ' if np.isnan(perk) else f'{perk:7.4f}'} | "
                          f"{'   nan ' if np.isnan(ww) else f'{ww:7.4f}'} | "
                          f"{str(used):>5s}")
    print("=" * 40)
    print(f"         РЕЗУЛЬТАТИ ПРЕТРЕКІНГУ: {iteration + 2}")
    print("=" * 40)



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
        result, img = _preprocess_image(opWrapper, img_path, return_img=True)
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

    if frames:
        print("Показуємо залишок...")
        try:
            max_h = max(f.shape[0] for f in frames)
            aligned_frames = []
            for f in frames:
                h, w = f.shape[0:2]
                if h < max_h:
                    pad_top = (max_h - h) // 2
                    pad_bottom = max_h - h - pad_top
                    f_aligned = cv2.copyMakeBorder(f, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    aligned_frames.append(f_aligned)
                else:
                    aligned_frames.append(f)

            combined_frame = cv2.hconcat(aligned_frames)
            cv2.imshow(f"Tracking (Залишок: {len(frames)} frames)", combined_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Помилка при об'єднанні залишку: {e}")



from src.boxing_project.tracking.tracker import MultiObjectTracker, TrackerConfig
from src.boxing_project.tracking.matcher import MatchConfig

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





