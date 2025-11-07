from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
from PIL.EpsImagePlugin import has_ghostscript
from scipy.optimize import linear_sum_assignment

from .track import Track, Detection


# --------- Утиліти логування ---------
class DebugLog:
    """
    Збирає структурований лог (dict) + за бажанням друкує; ВСЕГДА буферизує рядки.
    """
    def __init__(
        self,
        enabled_print: bool = False,
        sink: Optional[Callable[[str], None]] = None,
    ):
        self.enabled_print = enabled_print
        self.sink = sink if sink is not None else print
        self.buffer: List[str] = []          # ← буфер усіх рядків

        self.store: Dict[str, Any] = {
            "config": {},
            "shape": [0, 0],
            "pairs": [],
            "print_chunks": self.buffer,     # ← буфер віддамо нагору
        }

    # --- друк / буфер ---
    def _emit(self, line: str):
        self.buffer.append(line)
        if self.enabled_print:
            self.sink(line)

    def _print(self, msg: str):
        self._emit(msg)

    def section(self, title: str):
        title = title
        bar = "—" * max(8, len(title))
        self._emit("")         # порожній рядок перед секцією
        self._emit(title)
        self._emit(bar)

    def table(self, header: str, rows: List[str]):
        if header:
            self._emit(header)
        for r in rows:
            self._emit(r)

    # --- запис у словник ---
    def add_pair(self, pair_obj: Dict[str, Any]):
        self.store["pairs"].append(pair_obj)

    def set_meta(self, config: "MatchConfig", shape: Tuple[int, int]):
        self.store["config"] = {
            "alpha": config.alpha,
            "chi2_gating": config.chi2_gating,
            "large_cost": config.large_cost,
            "min_kp_conf": config.min_kp_conf,
            "has_keypoint_weights": config.keypoint_weights is not None,
        }
        self.store["shape"] = [int(shape[0]), int(shape[1])]


@dataclass
class MatchConfig:
    # Вага між рухом і позою: C = alpha*D_motion + (1-alpha)*D_pose
    alpha: float = 0.8
    # χ^2-поріг (df=2)
    chi2_gating: float = 9.21  # p≈0.99
    # Вартість для заборонених пар
    large_cost: float = 1e6
    # (зарезервовано під альтернативні нормалізації масштабу)
    pose_scale_eps: float = 1e-6
    # Ваги для ключових точок (якщо None — всі 1.0)
    keypoint_weights: Optional[np.ndarray] = None
    # Мінімальний конфіденс точки
    min_kp_conf: float = 0.05


def _normalize_pose(kp: np.ndarray) -> Tuple[np.ndarray, float]:
    vis = np.isfinite(kp).all(axis=1)
    if not np.any(vis):
        return kp * 0.0, 1.0
    pts = kp[vis]
    center = pts.mean(axis=0, keepdims=True)
    centered = kp - center
    d = np.sqrt(((pts - center) ** 2).sum(axis=1) + 1e-12)
    scale = float(np.sqrt((d ** 2).mean()) + 1e-12)
    return centered / max(scale, 1e-12), scale


# --- форматуємо рядки для друку таблиці (не для словника) ---
def _format_pose_rows(
    k_idx: np.ndarray,
    trk_norm: np.ndarray,
    det_norm: np.ndarray,
    diff: np.ndarray,
    per_k: np.ndarray,
    w_eff: np.ndarray,
    used_mask: np.ndarray
) -> List[str]:
    rows = []
    for i, k in enumerate(k_idx):
        tx, ty = trk_norm[k]
        dx_, dy_ = det_norm[k]
        ddx, ddy = diff[i]
        used = bool(used_mask[k])
        rows.append(
            f"{k:2d} | "
            f"{tx:>7.4f} {ty:>7.4f} | "
            f"{dx_:>7.4f} {dy_:>7.4f} | "
            f"{ddx:>7.4f} {ddy:>7.4f} | "
            f"{per_k[i]:>7.4f} | {w_eff[i]:>7.4f} | {str(used):>5s}"
        )
    return rows


def _pose_distance(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = ""
) -> Tuple[float, Dict[str, Any]]:
    """
    Рахує D_pose та збирає детальний словник для логів.
    Повертає (dist, pose_dict).
    """
    pose_dict: Dict[str, Any] = {}

    if track.last_keypoints is None or det.keypoints is None:
        if log:
            log.section(f"[{pair_tag}] POSE")
            log._print("немає поз — D_pose=0.0")
        pose_dict.update({
            "has_pose": False,
            "good_mask": None,
            "trk_norm": None,
            "det_norm": None,
            "diff": None,
            "per_k": None,
            "w_eff": None,
            "used_count": 0,
            "D_pose": 0.0
        })
        return 0.0, pose_dict

    kpt_t = np.asarray(track.last_keypoints, dtype=float)
    kpt_d = np.asarray(det.keypoints, dtype=float)
    n_k = kpt_t.shape[0]

    conf_t = np.asarray(track.last_kp_conf, dtype=float).reshape(-1) if track.last_kp_conf is not None else np.ones((n_k,), dtype=float)
    conf_d = np.asarray(det.kp_conf, dtype=float).reshape(-1) if det.kp_conf is not None else np.ones((n_k,), dtype=float)

    assert kpt_t.shape[0] == kpt_d.shape[0], "Track/Det must have same number of keypoints"

    good_t = np.isfinite(kpt_t).all(axis=1) & (conf_t >= cfg.min_kp_conf)
    good_d = np.isfinite(kpt_d).all(axis=1) & (conf_d >= cfg.min_kp_conf)
    good = (good_t & good_d)

    if not np.any(good):
        if log:
            log.section(f"[{pair_tag}] POSE")
            log._print("немає спільно якісних точок — D_pose=0.0")
        pose_dict.update({
            "has_pose": True,
            "good_mask": good.astype(bool).tolist(),
            "trk_norm": None,
            "det_norm": None,
            "diff": None,
            "per_k": None,
            "w_eff": None,
            "used_count": 0,
            "D_pose": 0.0
        })
        return 0.0, pose_dict

    # Нормалізація
    kpt_tn, _ = _normalize_pose(kpt_t)
    kpt_dn, _ = _normalize_pose(kpt_d)

    diff_used = kpt_tn[good] - kpt_dn[good]
    per_k_used = np.linalg.norm(diff_used, axis=1)

    if cfg.keypoint_weights is not None:
        w_all = np.asarray(cfg.keypoint_weights, dtype=float).reshape(-1)
        w_used = w_all[good]
    else:
        w_all = np.ones((n_k,), dtype=float)
        w_used = np.ones_like(per_k_used)

    w_used = w_used * 0.5 * (conf_t[good] + conf_d[good])
    D_pose = float((w_used * per_k_used).sum() / (w_used.sum() + 1e-12))

    # Розгортка до повної довжини (25) для словника/таблиці
    full_diff = np.full((n_k, 2), np.nan, dtype=float)
    full_perk = np.full((n_k,), np.nan, dtype=float)
    full_w = np.full((n_k,), np.nan, dtype=float)
    used_idx = np.where(good)[0]
    full_diff[used_idx] = diff_used
    full_perk[used_idx] = per_k_used
    # Для w_eff — теж тільки used
    full_w[used_idx] = w_used

    pose_dict.update({
        "has_pose": True,
        "good_mask": good.astype(bool).tolist(),
        "trk_norm": kpt_tn.tolist(),
        "det_norm": kpt_dn.tolist(),
        "diff": full_diff.tolist(),
        "per_k": full_perk.tolist(),
        "w_eff": full_w.tolist(),
        "used_count": int(used_idx.size),
        "D_pose": D_pose
    })

    # ДРУК (опціонально)
    if log and log.enabled_print:
        log.section(f"[{pair_tag}] POSE")
        header = (
            " k |    trk_x    trk_y |    det_x    det_y |      dx       dy |   ||d||  |  w_eff | used\n"
            "---+--------------------+--------------------+------------------+---------+--------+------"
        )
        rows = _format_pose_rows(
            k_idx=np.arange(n_k),
            trk_norm=kpt_tn,
            det_norm=kpt_dn,
            diff=full_diff,
            per_k=full_perk,
            w_eff=full_w,
            used_mask=good
        )
        log.table(header, rows)
        log._print(f"\nD_pose = {D_pose:.6f}  (по {int(used_idx.size)} точках)")

    return D_pose, pose_dict


def _motion_cost_with_gating(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = ""
) -> Tuple[float, bool, float]:
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = (d2 <= cfg.chi2_gating)
    d_motion = float(np.sqrt(max(d2, 0.0)))

    if log and log.enabled_print:
        log.section(f"[{pair_tag}] MOTION")
        check = "✓" if allowed else "✗"
        log._print(f"• d² = {d2:.6f}   |   χ²_gate = {cfg.chi2_gating:.6f}   |   allowed = {check}")
        log._print(f"• d_motion = √(d²) = {d_motion:.6f}")


    return d_motion, allowed, float(d2)


def build_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    cfg: MatchConfig,
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Будує матрицю C (n_tracks x n_dets) і ПОВЕРТАЄ:
      - C
      - log_matcher (великий словник з деталями по кожній парі)

    log_matcher["pairs"] — список таких об’єктів:
      {
        "track_index": i, "det_index": j,
        "motion": {"d2": ..., "d_motion": ..., "allowed": ...},
        "pose": {
           "has_pose": bool,
           "good_mask": [bool]*K,
           "trk_norm": [[x,y]]*K,
           "det_norm": [[x,y]]*K,
           "diff": [[dx,dy]]*K (NaN для не-used),
           "per_k": [float]*K (NaN для не-used),
           "w_eff": [float]*K (NaN для не-used),
           "used_count": int,
           "D_pose": float
        },
        "final": {
           "alpha": cfg.alpha,
           "cost": float,
           "components": {"d_motion": ..., "d_pose": ...},
           "reason": "ok" | "gated_out"
        }
      }
    """
    n_tracks, n_dets = len(tracks), len(detections)
    if n_tracks == 0 or n_dets == 0:
        log = DebugLog(enabled_print=show, sink=sink)
        log.set_meta(cfg, (n_tracks, n_dets))
        return np.zeros((n_tracks, n_dets), dtype=float), log.store

    C = np.zeros((n_tracks, n_dets), dtype=float)
    log = DebugLog(enabled_print=show, sink=sink)
    log.set_meta(cfg, (n_tracks, n_dets))

    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            pair_tag = f"T{i}↔D{j}"
            pair_obj: Dict[str, Any] = {"track_index": i, "det_index": j}

            d_motion, ok, d2 = _motion_cost_with_gating(trk, det, cfg, log=log, pair_tag=pair_tag)
            pair_obj["motion"] = {"d2": d2, "d_motion": d_motion, "allowed": bool(ok)}

            if not ok:
                C[i, j] = cfg.large_cost
                pair_obj["pose"] = {
                    "has_pose": False, "good_mask": None, "trk_norm": None, "det_norm": None,
                    "diff": None, "per_k": None, "w_eff": None, "used_count": 0, "D_pose": 0.0
                }
                pair_obj["final"] = {
                    "alpha": cfg.alpha,
                    "cost": float(cfg.large_cost),
                    "components": {"d_motion": d_motion, "d_pose": 0.0},
                    "reason": "gated_out"
                }
                log.add_pair(pair_obj)

                if show:
                    log.section(f"[pair {pair_tag}] RESULT")
                    log._print(f"Гейтинг відсік пару (d2={d2:.6f} > {cfg.chi2_gating:.6f}). "
                               f"Вартість = LARGE_COST={cfg.large_cost:g}")
                continue

            d_pose, pose_dict = _pose_distance(trk, det, cfg, log=log, pair_tag=pair_tag)

            cost = cfg.alpha * d_motion + (1.0 - cfg.alpha) * d_pose
            C[i, j] = cost

            pair_obj["pose"] = pose_dict
            pair_obj["final"] = {
                "alpha": cfg.alpha,
                "cost": float(cost),
                "components": {"d_motion": float(d_motion), "d_pose": float(d_pose)},
                "reason": "ok"
            }
            log.add_pair(pair_obj)

            if show:
                log.section(f"[pair {pair_tag}] RESULT")
                alpha = cfg.alpha
                log._print(f"α = {alpha:.4f}")
                log._print(
                    f"final = α·d_motion + (1-α)·D_pose = {alpha:.4f}·{d_motion:.6f} + {(1.0 - alpha):.4f}·{d_pose:.6f} = {cost:.6f}")


    return C, log.store


def linear_assignment_with_unmatched(C: np.ndarray, large_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    rows, cols = linear_sum_assignment(C)

    matched = []
    used_tracks = set()
    used_dets = set()

    for r, c in zip(rows, cols):
        if C[r, c] >= large_cost:
            continue
        matched.append((int(r), int(c)))
        used_tracks.add(int(r))
        used_dets.add(int(c))

    n_tracks, n_dets = C.shape
    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]

    return matched, unmatched_tracks, unmatched_dets


def match_tracks_and_detections(
    tracks: List[Track],
    detections: List[Detection],
    cfg: Optional[MatchConfig] = None,
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray, Dict[str, Any]]:
    """
    Тепер повертаємо й словник логу, щоб ти мав його на рівні кадру.
    """
    if cfg is None:
        cfg = MatchConfig()

    C, log_matcher = build_cost_matrix(tracks, detections, cfg, show=show, sink=sink)
    matches, um_tr, um_dt = linear_assignment_with_unmatched(C, cfg.large_cost)
    return matches, um_tr, um_dt, C, log_matcher
