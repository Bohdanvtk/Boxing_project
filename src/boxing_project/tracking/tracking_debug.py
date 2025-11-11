import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple


class DebugLog:

    id: int = 0
    """
    Збирає структурований лог (dict) + за бажанням друкує; ВСЕГДА буферизує рядки.
    """
    def __init__(
        self,
        enabled_print: bool = True,
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


# ---------- ПОЗА КЛАСОМ: хелпери для pose ----------

def set_pose_no_keypoints(
    pose_dict: Dict[str, Any],
    log: Optional[DebugLog],
    pair_tag: str,
) -> None:
    """
    Випадок: немає keypoints взагалі (або в треку, або в детекції).
    Заповнює pose_dict і (опційно) щось друкує в лог.
    """
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


def set_pose_no_good_points(
    pose_dict: Dict[str, Any],
    log: Optional[DebugLog],
    pair_tag: str,
    good_mask: np.ndarray,
) -> None:
    """
    Випадок: keypoints є, але НУЛЬ спільно якісних.
    """
    if log:
        log.section(f"[{pair_tag}] POSE")
        log._print("немає спільно якісних точок — D_pose=0.0")

    pose_dict.update({
        "has_pose": True,
        "good_mask": good_mask.astype(bool).tolist(),
        "trk_norm": None,
        "det_norm": None,
        "diff": None,
        "per_k": None,
        "w_eff": None,
        "used_count": 0,
        "D_pose": 0.0
    })


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


def fill_pose_full_debug(
    pose_dict: Dict[str, Any],
    log: Optional[DebugLog],
    pair_tag: str,
    n_k: int,
    good_mask: np.ndarray,
    kpt_tn: np.ndarray,
    kpt_dn: np.ndarray,
    diff_used: np.ndarray,
    per_k_used: np.ndarray,
    w_used: np.ndarray,
    D_pose: float,
) -> None:
    """
    Випадок: нормальні точки є, порахований D_pose.
    Тут:
      - розгортаємо diff/per_k/w_eff до повної довжини n_k
      - оновлюємо pose_dict
      - (опційно) друкуємо красиву табличку через DebugLog
    """
    full_diff = np.full((n_k, 2), np.nan, dtype=float)
    full_perk = np.full((n_k,), np.nan, dtype=float)
    full_w = np.full((n_k,), np.nan, dtype=float)

    used_idx = np.where(good_mask)[0]
    full_diff[used_idx] = diff_used
    full_perk[used_idx] = per_k_used
    full_w[used_idx] = w_used  # Для w_eff — теж тільки used

    pose_dict.update({
        "has_pose": True,
        "good_mask": good_mask.astype(bool).tolist(),
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
            used_mask=good_mask
        )
        log.table(header, rows)
        log._print(f"\nD_pose = {D_pose:.6f}  (по {int(used_idx.size)} точках)")


# ---------- PRINT TRACKING RESULTS ----------

def print_tracking_results(log: dict, iteration: int, show_pose_tables: bool = False):
    """
    Красивий вивід результатів трекінгу за кадр:
      - Summary active tracks
      - Cost matrix з коректними Track IDs (row_track_ids)
      - Парні логи по кожному активному треку (motion/pose/final)
      - (опційно) повні таблиці по 25 точках для кожної пари
    """
      # DEBUG, можеш потім прибрати

    print("=" * 140)
    print(f"         РЕЗУЛЬТАТИ ТРЕКІНГУ: {iteration + 1}")
    print("=" * 140)

    # --- дістаємо штуки з логу кадру ---
    active_tracks = log.get("active_tracks", [])
    C = np.array(log.get("cost_matrix", []))
    row_track_ids = log.get("row_track_ids", [])

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




# ---------- MATCHER HELPERS ----------

def create_matcher_log(
    cfg: "MatchConfig",
    shape: Tuple[int, int],
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
) -> DebugLog:
    log = DebugLog(enabled_print=show, sink=sink)
    log.set_meta(cfg, shape)
    return log


def make_pair_base(track_index: int, det_index: int) -> Dict[str, Any]:
    return {
        "track_index": track_index,
        "det_index": det_index,
    }


def fill_pair_gated_out(
    pair_obj: Dict[str, Any],
    cfg: "MatchConfig",
    d_motion: float,
) -> None:
    pair_obj["pose"] = {
        "has_pose": False,
        "good_mask": None,
        "trk_norm": None,
        "det_norm": None,
        "diff": None,
        "per_k": None,
        "w_eff": None,
        "used_count": 0,
        "D_pose": 0.0,
    }
    pair_obj["final"] = {
        "alpha": cfg.alpha,
        "cost": float(cfg.large_cost),
        "components": {"d_motion": float(d_motion), "d_pose": 0.0},
        "reason": "gated_out",
    }


def fill_pair_ok(
    pair_obj: Dict[str, Any],
    cfg: "MatchConfig",
    d_motion: float,
    d_pose: float,
    cost: float,
    pose_dict: Dict[str, Any],
) -> None:
    pair_obj["pose"] = pose_dict
    pair_obj["final"] = {
        "alpha": cfg.alpha,
        "cost": float(cost),
        "components": {
            "d_motion": float(d_motion),
            "d_pose": float(d_pose),
        },
        "reason": "ok",
    }


def print_gating_result(
    log: Optional[DebugLog],
    pair_tag: str,
    d2: float,
    cfg: "MatchConfig",
) -> None:
    if not (log and log.enabled_print):
        return

    log.section(f"[pair {pair_tag}] RESULT")
    log._print(
        f"Гейтинг відсік пару (d2={d2:.6f} > {cfg.chi2_gating:.6f}). "
        f"Вартість = LARGE_COST={cfg.large_cost:g}"
    )


def print_pair_result(
    log: Optional[DebugLog],
    pair_tag: str,
    cfg: "MatchConfig",
    d_motion: float,
    d_pose: float,
    cost: float,
) -> None:
    if not (log and log.enabled_print):
        return

    alpha = cfg.alpha
    log.section(f"[pair {pair_tag}] RESULT")
    log._print(f"α = {alpha:.4f}")
    log._print(
        "final = α·d_motion + (1-α)·D_pose = "
        f"{alpha:.4f}·{d_motion:.6f} + {(1.0 - alpha):.4f}·{d_pose:.6f} = {cost:.6f}"
    )
