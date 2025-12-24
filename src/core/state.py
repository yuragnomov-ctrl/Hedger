# -*- coding: utf-8 -*-
"""
State management for hedger
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections import deque
from typing import Any, Deque, Dict

log = logging.getLogger(__name__)


class State:
    """Класс для хранения внутренних состояний между итерациями."""

    def __init__(self) -> None:
        # Dynamic hedge
        self.last_px: Dict[str, float] = {}
        self.last_zone: Dict[str, str] = {}

        # Chase attempts for √P‑grid
        self.chase_attempts: Dict[str, int] = {}

        # Trigger Market history
        self.tm_anchor_price: Dict[str, float] = {}
        self.tm_last_trigger_price: Dict[str, float] = {}
        self.tm_last_trigger_ts: Dict[str, float] = {}
        self.tm_levels_fired: Dict[str, int] = {}
        self.tm_history: Dict[str, Deque[Tuple[float, float]]] = {}

        # √P‑grid state
        self.last_trig_s: Dict[str, float] = {}
        self.dvol_ds: Dict[str, float] = {}
        self.prev_s: Dict[str, float] = {}
        self.prev_vol: Dict[str, float] = {}

        # Runtime-only timers
        self.last_action_ts: Dict[str, float] = {}
        self.last_bleed_ts: Dict[str, float] = {}
        self.last_inv_bleed_ts: Dict[str, float] = {}

        # Zero volume tracking for intelligent flatten
        self.consecutive_zero_vol: Dict[str, int] = {}
        self.vol_zero_timestamps: Dict[str, float] = {}
        self.cached_csv_boundaries: Dict[str, Dict[str, float]] = {}

        # LP Positions tracking v1.8.0
        self.lp_positions: Dict[int, Dict[str, Any]] = {}  # token_id -> position data
        self.active_positions_count: int = 0
        self.archived_positions_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает сериализуемый словарь с существенными полями."""
        return {
            "last_px": self.last_px,
            "last_zone": self.last_zone,
            "chase_attempts": self.chase_attempts,
            "tm_anchor_price": self.tm_anchor_price,
            "tm_last_trigger_price": self.tm_last_trigger_price,
            "tm_last_trigger_ts": self.tm_last_trigger_ts,
            "tm_levels_fired": self.tm_levels_fired,
            "last_trig_s": self.last_trig_s,
            "dvol_ds": self.dvol_ds,
            "prev_s": self.prev_s,
            "prev_vol": self.prev_vol,
            "consecutive_zero_vol": self.consecutive_zero_vol,
            "vol_zero_timestamps": self.vol_zero_timestamps,
            "cached_csv_boundaries": self.cached_csv_boundaries,
            "lp_positions": self.lp_positions,
            "active_positions_count": self.active_positions_count,
            "archived_positions_count": self.archived_positions_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "State":
        s = cls()
        s.last_px = d.get("last_px", {})
        s.last_zone = d.get("last_zone", {})
        s.chase_attempts = d.get("chase_attempts", {})
        s.tm_anchor_price = d.get("tm_anchor_price", {})
        s.tm_last_trigger_price = d.get("tm_last_trigger_price", {})
        s.tm_last_trigger_ts = d.get("tm_last_trigger_ts", {})
        s.tm_levels_fired = d.get("tm_levels_fired", {})
        s.last_trig_s = d.get("last_trig_s", {})
        s.dvol_ds = d.get("dvol_ds", {})
        s.prev_s = d.get("prev_s", {})
        s.prev_vol = d.get("prev_vol", {})
        s.consecutive_zero_vol = d.get("consecutive_zero_vol", {})
        s.vol_zero_timestamps = d.get("vol_zero_timestamps", {})
        s.cached_csv_boundaries = d.get("cached_csv_boundaries", {})
        # JSON сохраняет ключи словаря как строки -> возвращаем обратно int token_id
        raw_lp = d.get("lp_positions", {}) or {}
        if isinstance(raw_lp, dict):
            try:
                s.lp_positions = {int(k): v for k, v in raw_lp.items()}
            except Exception:
                # fallback: как есть
                s.lp_positions = raw_lp
        else:
            s.lp_positions = {}
        s.active_positions_count = d.get("active_positions_count", 0)
        s.archived_positions_count = d.get("archived_positions_count", 0)
        return s


def save_state(state: State, filepath: str, cfg: Dict[str, Any] = None) -> None:
    """Сохраняет текущее состояние в JSON-файл (атомарно через tmp) с ротацией."""
    try:
        # Проверяем размер и количество архивов
        if cfg:
            state_cfg = cfg.get("persistence", {})
            archive_count = int(state_cfg.get("state_archive_count", 5))
            max_size = int(state_cfg.get("state_max_size", 10000))
            archive_path = state_cfg.get("state_archive_path", "logs/state_archive")

            # Создаем директорию для архивов
            os.makedirs(archive_path, exist_ok=True)

            # Проверяем размер текущего state
            state_data = json.dumps(state.to_dict(), indent=2)
            current_size = len(state_data.encode('utf-8'))

            # Если размер превышает лимит, архивируем текущий
            if current_size > max_size:
                # Получаем список существующих архивов
                existing_archives = []
                if os.path.exists(archive_path):
                    for f in os.listdir(archive_path):
                        if f.startswith("state_") and f.endswith(".json"):
                            existing_archives.append(f)

                # Удаляем старые архивы если превышен лимит
                existing_archives.sort()
                while len(existing_archives) >= archive_count:
                    old_archive = existing_archives.pop(0)
                    old_path = os.path.join(archive_path, old_archive)
                    try:
                        os.remove(old_path)
                        log.info(f"Removed old state archive: {old_archive}")
                    except Exception as e:
                        log.warning(f"Failed to remove old state archive {old_archive}: {e}")

                # Создаем новый архив с timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_filename = f"state_{timestamp}.json"
                archive_filepath = os.path.join(archive_path, archive_filename)

                # Копируем текущий state в архив
                if os.path.exists(filepath):
                    shutil.copy2(filepath, archive_filepath)
                    log.info(f"Archived state to: {archive_filename}")
                    state.archived_positions_count += 1

        # Атомарное сохранение через временный файл
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp_path, filepath)
    except Exception as e:
        log.error("Failed to save state: %s", e)


def load_state(filepath: str) -> State:
    """Загружает состояние из файла; если файл отсутствует, возвращает пустое."""
    if not os.path.exists(filepath):
        log.warning("State file not found, starting fresh.")
        return State()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return State.from_dict(json.load(f))
    except Exception as e:
        log.error("Failed to load state: %s, starting fresh.", e)
        return State()