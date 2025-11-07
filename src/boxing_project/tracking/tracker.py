from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import copy
import numpy as np

from .kalman import KalmanTracker
from .track import Track, Detection
from .matcher import MatchConfig, match_tracks_and_detections


# ----------------------------- #
#          Конфіг тракера       #
# ----------------------------- #

@dataclass
class TrackerConfig:
    # Параметри Кальмана
    dt: float = 1.0
    process_var: float = 1.0
    measure_var: float = 10.0
    p0: float = 1e3

    # Керування життям треків
    max_age: int = 30          # скільки кадрів можна не оновлювати трек
    min_hits: int = 3          # після скількох хітів трек стає confirmed

    # Матчинг (важливо: окремий екземпляр для кожного TrackerConfig)
    match: MatchConfig = field(default_factory=MatchConfig)

    # OpenPose → Detection
    min_kp_conf: float = 0.05  # фільтр слабких kps
    expect_body25: bool = True # якщо True — чекаємо BODY_25 (75 чисел у flatten)


# ----------------------------- #
#     Адаптер OpenPose → Det    #
# ----------------------------- #

def openpose_people_to_detections(
    people: List[Dict[str, Any]],
    min_kp_conf: float = 0.05,
    expect_body25: bool = True
) -> List[Detection]:
    """
    Перетворює список people з OpenPose JSON у список Detection.
    Підтримує кілька форматів:
      1) 'pose_keypoints_2d' — flatten список довжини 75 (BODY_25: 25*(x,y,conf))
      2) 'keypoints'/'pose' як np.ndarray або список форми (K, 3)
      3) 'pose_2d' з полями 'x', 'y', 'conf' (будь-яка форма, яку можна привести до (K,3))

    Центр детекції = медіана видимих (x,y) з conf >= min_kp_conf (стійко до аутлайєрів).
    """
    dets: List[Detection] = []

    for person in people:
        kps: Optional[np.ndarray] = None

        # варіант 1: стандартний JSON OpenPose
        if 'pose_keypoints_2d' in person and isinstance(person['pose_keypoints_2d'], (list, tuple)):
            arr = np.asarray(person['pose_keypoints_2d'], dtype=float).reshape(-1)
            if arr.size % 3 != 0:
                # некоректний розмір — пропускаємо цю людину
                continue
            K = arr.size // 3
            # якщо очікуємо BODY_25, K має бути 25; але допускаємо інші, аби не падати
            kps = arr.reshape(K, 3)

        # варіант 2: вже наданий (K,3) або (K,2)
        elif 'keypoints' in person:
            arr = np.asarray(person['keypoints'], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    # немає conf -> додамо conf=1
                    ones = np.ones((arr.shape[0], 1), dtype=float)
                    arr = np.concatenate([arr, ones], axis=1)
                kps = arr[:, :3]

        elif 'pose' in person:
            arr = np.asarray(person['pose'], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    ones = np.ones((arr.shape[0], 1), dtype=float)
                    arr = np.concatenate([arr, ones], axis=1)
                kps = arr[:, :3]

        elif 'pose_2d' in person:
            p = person['pose_2d']
            xs = np.asarray(p.get('x', []), dtype=float).reshape(-1, 1)
            ys = np.asarray(p.get('y', []), dtype=float).reshape(-1, 1)
            cs = np.asarray(p.get('conf', []), dtype=float).reshape(-1, 1)
            if xs.shape == ys.shape == cs.shape and xs.size > 0:
                kps = np.concatenate([xs, ys, cs], axis=1)

        if kps is None:
            # немає ключових точок — пропускаємо
            continue

        # Фільтр за conf, сховаємо погані як NaN (щоб не впливали)
        good = kps[:, 2] >= float(min_kp_conf)
        xy = kps[:, :2].copy()
        xy[~good] = np.nan

        # Центр — медіана видимих (робить центр стійким)
        if np.all(~good):
            # якщо жодної якісної точки — пропускаємо
            continue
        cx = np.nanmedian(xy[:, 0])
        cy = np.nanmedian(xy[:, 1])

        dets.append(
            Detection(
                center=(float(cx), float(cy)),
                keypoints=xy,      # (K, 2) з NaN там, де conf низький
                kp_conf=kps[:, 2], # (K,)
                meta={'raw': person}
            )
        )

    return dets


# ----------------------------- #
#            ТРЕКЕР              #
# ----------------------------- #

class MultiObjectTracker:
    """
    Керує списком треків:
      predict → build C → Hungarian → update → керування життям.
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: List[Track] = []
        self._next_id: int = 1

    # ---- службові ---- #

    def _new_track(self, det: Detection) -> Track:
        kf = KalmanTracker(
            x0=[det.center[0], det.center[1], 0.0, 0.0],
            dt=self.cfg.dt,
            process_var=self.cfg.process_var,
            measure_var=self.cfg.measure_var,
            p0=self.cfg.p0
        )
        trk = Track(
            track_id=self._next_id,
            kf=kf,
            min_hits=self.cfg.min_hits
        )
        self._next_id += 1
        # одразу перше оновлення — щоб стан «прибився» до виміру
        trk.update(det)
        return trk

    def _remove_dead(self):
        self.tracks = [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    # ---- API кадру ---- #

    def update_with_openpose(
        self,
        openpose_people: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Повний цикл для одного кадру з сирими people від OpenPose.

        Повертає словник:
          {
            'matches': List[Tuple[track_id, det_index]],
            'unmatched_track_ids': List[int],
            'unmatched_det_indices': List[int],
            'cost_matrix': np.ndarray,
            'active_tracks': List[Dict]  # стислий стан треків після апдейту
          }
        """
        detections = openpose_people_to_detections(
            openpose_people,
            min_kp_conf=self.cfg.min_kp_conf,
            expect_body25=self.cfg.expect_body25
        )
        return self.update(detections)

    def update(
            self,
            detections: List[Detection]
    ) -> Dict[str, Any]:
        # 1) PREDICT
        for trk in self.tracks:
            trk.predict()

        # --- SNAPSHOT: індекс треку → track_id ДО матчингу ---
        idx2tid = {i: t.track_id for i, t in enumerate(self.tracks)}
        # >>> ДОДАТИ ОЦЕ: список track_id рівно у порядку рядків cost_matrix
        row_track_ids = [idx2tid[i] for i in range(len(self.tracks))]

        # 2) MATCH
        matches_idx, um_tr_idx, um_det_idx, C, log_matcher = match_tracks_and_detections(
            tracks=self.tracks,
            detections=detections,
            cfg=self.cfg.match
        )

        # 3) збирання pair_logs_by_tid ...
        pair_logs_by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for p in log_matcher.get("pairs", []):
            i = p.get("track_index")
            tid = idx2tid.get(i)
            if tid is not None:
                pair_logs_by_tid[tid].append(copy.deepcopy(p))

        # 4) UPDATE призначені ...
        id_pairs: List[Tuple[int, int]] = []
        for i_track, j_det in matches_idx:
            trk = self.tracks[i_track]
            det = detections[j_det]
            trk.update(det)
            id_pairs.append((trk.track_id, j_det))

        # 5) нові треки з unmatched детекцій ...
        for j in um_det_idx:
            new_trk = self._new_track(detections[j])
            self.tracks.append(new_trk)
            pair_logs_by_tid[new_trk.track_id].append({
                "track_index": None,
                "det_index": j,
                "motion": None,
                "pose": None,
                "final": {
                    "alpha": self.cfg.match.alpha,
                    "cost": 0.0,
                    "components": {"d_motion": 0.0, "d_pose": 0.0},
                    "reason": "new_track_from_unmatched_detection"
                }
            })

        # 6) прибрати мертві
        self._remove_dead()

        # 7) зібрати вихід
        unmatched_track_ids = [idx2tid[i] for i in um_tr_idx if i in idx2tid]

        active_tracks_summary = [
            {
                "track_id": t.track_id,
                "confirmed": t.confirmed,
                "age": t.age,
                "hits": t.hits,
                "time_since_update": t.time_since_update,
                "state": t.state.tolist(),
                "pos": t.pos(),
                "match_log": pair_logs_by_tid.get(t.track_id, [])
            }
            for t in self.tracks
            if not t.is_dead(self.cfg.max_age)
        ]

        return {
            "matches": id_pairs,
            "unmatched_track_ids": unmatched_track_ids,
            "unmatched_det_indices": um_det_idx,
            "cost_matrix": C,
            "active_tracks": active_tracks_summary,
            "frame_log": log_matcher,
            "row_track_ids": row_track_ids  # ← тепер це визначено вище
        }

    # ---- зручності ---- #

    def get_active_tracks(
        self,
        confirmed_only: bool = True
    ) -> List[Track]:
        """Повернути список активних треків (за замовчуванням лише confirmed)."""
        if confirmed_only:
            return [t for t in self.tracks if t.confirmed and not t.is_dead(self.cfg.max_age)]
        return [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    def reset(self):
        """Скинути усі треки (нове відео)."""
        self.tracks.clear()
        self._next_id = 1
