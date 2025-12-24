# -*- coding: utf-8 -*-
"""
Utility helper functions for hedger
"""
from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Constants - will be loaded from config
Q96 = 2 ** 96  # Fallback value
EPS = 1e-9  # Fallback value
MAX_ZERO_VOL_CYCLES_DEFAULT = 6  # Fallback value

def init_constants_from_config(cfg: Dict[str, Any]) -> None:
    """Initialize constants from configuration"""
    global Q96, EPS, MAX_ZERO_VOL_CYCLES_DEFAULT
    try:
        system_cfg = cfg.get("system", {})
        Q96 = int(system_cfg.get("Q96", 2 ** 96))
        EPS = float(system_cfg.get("EPS", 1e-9))
        MAX_ZERO_VOL_CYCLES_DEFAULT = int(system_cfg.get("max_zero_vol_cycles_default", 6))
    except Exception as e:
        log.warning("Failed to initialize constants from config, using fallbacks: %s", e)

# Dependencies for onchain functionality
try:
    from src.dex.uniswap_v3_math import get_amounts_for_liquidity
    from decimal import Decimal
    _ONCHAIN_ENABLED = True
except Exception as e:
    log.warning("Failed to import onchain dependencies: %s. PnL projection disabled.", e)
    get_amounts_for_liquidity = None
    _ONCHAIN_ENABLED = False


def _ema(prev: float, x: float, alpha: float = 0.3) -> float:
    """Экспоненциальное сглаживание."""
    return x if prev == 0.0 else prev * (1 - alpha) + x * alpha


def _update_price_history(state: "State", symbol: str, px: float, lookback_sec: int) -> float:
    """
    Сохраняет цену в историю Trigger‑Market и возвращает самую старую цену в окне
    lookback_sec. Если история пуста, возвращает текущую цену.
    """
    dq = state.tm_history.get(symbol)
    if dq is None:
        dq = deque()
        state.tm_history[symbol] = dq
    now = time.time()
    dq.append((now, px))
    while dq and (now - dq[0][0] > lookback_sec):
        dq.popleft()
    return dq[0][1] if dq else px


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def classify_zone(
    long_q: float,
    short_q: float,
    be_long: float,
    be_short: float,
    px_perp: float,
    tp_long_pct: float,
    tp_short_pct: float,
    sl_long_pct: float,
    sl_short_pct: float,
    long_to_short_threshold_pct: float = 40.0,
    inv_b_mode: str = "both",
    inv_b_long_pct: float = 1.0,
    inv_b_short_pct: float = 1.0,
) -> Tuple[str, bool]:
    """
    Возвращает (zone, inv_b_flag) для текущего состояния позиций.
    zone: "A", "B", "C", "BE+"
    inv_b_flag: True если B-зона в инверсии (обе позиции в плюсе)
    """
    # unrealized PnL на бирже
    pnl_long = (px_perp - be_long) * long_q if be_long > 0 else 0.0
    pnl_short = (be_short - px_perp) * short_q if be_short > 0 else 0.0
    pnl_total = pnl_long + pnl_short

    # Проверяем инверсию B-зоны только если TP/SL не запрещен
    inv_b = False
    if inv_b_mode in ("both", "long") and long_q > 0 and be_long > 0:
        inv_b = inv_b or (pnl_long / (be_long * long_q) * 100 >= inv_b_long_pct)
    if inv_b_mode in ("both", "short") and short_q > 0 and be_short > 0:
        inv_b = inv_b or (pnl_short / (be_short * short_q) * 100 >= inv_b_short_pct)

    # ZONE LOGIC
    if long_q <= EPS and short_q <= EPS:
        return "EMPTY", False

    # TP/SL зоны
    if long_q > EPS and be_long > 0:
        long_pct = (px_perp - be_long) / be_long * 100
        if long_pct >= tp_long_pct:
            return "TP_LONG", inv_b
        if long_pct <= -sl_long_pct:
            return "SL_LONG", inv_b

    if short_q > EPS and be_short > 0:
        short_pct = (be_short - px_perp) / be_short * 100
        if short_pct >= tp_short_pct:
            return "TP_SHORT", inv_b
        if short_pct <= -sl_short_pct:
            return "SL_SHORT", inv_b

    # Основные зоны A/B/C
    if long_q > EPS and short_q > EPS:
        # Обе позиции есть - определяем зону по BE
        if be_long <= EPS or be_short <= EPS:
            return "C", inv_b
        if px_perp < be_long and px_perp < be_short:
            return "A", inv_b
        elif px_perp > be_long and px_perp > be_short:
            return "C", inv_b
        else:
            return "B", inv_b
    elif long_q > EPS:
        # Только long
        if be_long <= EPS:
            return "C", inv_b
        return "C" if px_perp > be_long else "A", inv_b
    elif short_q > EPS:
        # Только short
        if be_short <= EPS:
            return "C", inv_b
        return "C" if px_perp < be_short else "A", inv_b
    else:
        return "EMPTY", False


def get_grid_boundaries(base: str, targets: Dict[str, Dict[str, float]], trig_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Возвращает границы цен для √P-сетки:
    - price_min: lower bound LP + расширение вниз
    - price_max: upper bound LP + расширение вверх
    """
    cfg = trig_cfg.get("lp_boundaries", {})
    expand_down = float(cfg.get("expand_down_pct", 0.0)) / 100.0
    expand_up = float(cfg.get("expand_up_pct", 0.0)) / 100.0

    # Получаем границы LP из targets
    t = targets.get(base.upper(), {})
    lp_price_min = float(t.get("price_min", 0.0))
    lp_price_max = float(t.get("price_max", 0.0))

    if lp_price_min <= 0 or lp_price_max <= 0:
        log.warning("[GRID_BOUNDARIES] %s has invalid LP bounds: min=%.6f, max=%.6f",
                   base, lp_price_min, lp_price_max)
        return 0.0, 0.0

    # Применяем расширение
    price_min = lp_price_min * (1.0 - expand_down)
    price_max = lp_price_max * (1.0 + expand_up)

    log.debug("[GRID_BOUNDARIES] %s: LP[%.6f, %.6f] -> Grid[%.6f, %.6f]",
              base, lp_price_min, lp_price_max, price_min, price_max)

    return price_min, price_max


def _get_lp_info_by_token_id(base: str, targets: Dict[str, Dict[str, float]], state: "State") -> tuple[Optional[int], Optional[Dict[str, Any]]]:
    base = base.upper()
    t = targets.get(base, {}) or {}
    token_id = t.get("token_id")
    if token_id is None:
        return None, None
    try:
        tid = int(token_id)
    except Exception:
        return None, None
    return tid, (state.lp_positions.get(tid, None))


def _get_lp_initial_value_usdt(base: str, targets: Dict[str, Dict[str, float]], state: "State") -> float:
    tid, lp_info = _get_lp_info_by_token_id(base, targets, state)
    if not lp_info:
        return 0.0
    return _safe_float(lp_info.get("initial_value_usdt"), 0.0)


def _get_cex_realized_delta_since_lp_start(
    base: str,
    bybit_symbol: str,
    targets: Dict[str, Dict[str, float]],
    state: "State",
    bb: "BybitClient",
) -> float:
    """
    Возвращает realized_delta = realized_total(now) - realized_baseline(at LP start).
    Baseline хранится в state.lp_positions[token_id]["cex_realized_baseline"].
    """
    tid, lp_info = _get_lp_info_by_token_id(base, targets, state)
    if tid is None or not lp_info:
        return 0.0

    try:
        realized_total = _safe_float(bb.get_realized_pnl(bybit_symbol), 0.0)
    except Exception:
        realized_total = 0.0

    baseline = lp_info.get("cex_realized_baseline", None)
    if baseline is None:
        # привязываем baseline при первом расчёте
        lp_info["cex_symbol"] = bybit_symbol
        lp_info["cex_realized_baseline"] = realized_total
        lp_info["cex_realized_last"] = realized_total
        state.lp_positions[tid] = lp_info
        return 0.0

    baseline_f = _safe_float(baseline, realized_total)
    lp_info["cex_symbol"] = bybit_symbol
    lp_info["cex_realized_last"] = realized_total
    state.lp_positions[tid] = lp_info
    return realized_total - baseline_f


def lp_value_at_vol_price(target: Dict[str, Any], price_vol_in_stable: float) -> Optional[float]:
    """
    Стоимость LP (USDT) на гипотетической цене price_vol_in_stable (USDT за 1 VOL).
    Требует meta из onchain reader: liquidity, sqrt_price_lower_x96, sqrt_price_upper_x96,
    decimals0/1, is0_stable.
    """
    if not _ONCHAIN_ENABLED or get_amounts_for_liquidity is None:
        return None
    if price_vol_in_stable <= 0:
        return None

    L = int(target.get("liquidity", 0) or 0)
    spl = int(target.get("sqrt_price_lower_x96", 0) or 0)
    spu = int(target.get("sqrt_price_upper_x96", 0) or 0)
    if L <= 0 or spl <= 0 or spu <= 0:
        return None

    is0_stable = bool(target.get("is0_stable", False))
    d0 = int(target.get("decimals0", 18) or 18)
    d1 = int(target.get("decimals1", 18) or 18)

    # Перевод vol/stable цены в raw p = token1/token0 (Uniswap v3 sqrtPriceX96)
    # token0 stable, token1 vol => vol/stable = 1/p_raw => p_raw = 1/price
    # token0 vol, token1 stable => p_raw = price
    p_raw = (1.0 / price_vol_in_stable) if is0_stable else price_vol_in_stable
    if p_raw <= 0:
        return None

    sqrt_price_x96 = int((Decimal(p_raw).sqrt()) * Q96)

    amount0_raw, amount1_raw = get_amounts_for_liquidity(L, sqrt_price_x96, spl, spu)
    a0 = float(amount0_raw / (Decimal(10) ** d0))
    a1 = float(amount1_raw / (Decimal(10) ** d1))

    # Стоимость в USDT (stable)
    return (a0 + a1 * price_vol_in_stable) if is0_stable else (a1 + a0 * price_vol_in_stable)


def project_strategy_pnl_at_price(
    base: str,
    price_vol_in_stable: float,
    long_q: float,
    short_q: float,
    be_long: float,
    be_short: float,
    targets: Dict[str, Dict[str, float]],
    state: "State",
    realized_delta_since_lp: float,
) -> Optional[Dict[str, float]]:
    """
    Проекция total PnL стратегии на цене P:
      (LP_value(P) - LP_init) + CEX_unreal(P) + realized_delta_since_lp
    """
    base = base.upper()
    t = targets.get(base, {}) or {}

    lp_init = _get_lp_initial_value_usdt(base, targets, state)
    if lp_init <= 0:
        return None

    lp_val = lp_value_at_vol_price(t, price_vol_in_stable)
    if lp_val is None:
        return None

    lp_delta = lp_val - lp_init

    cex_unreal = 0.0
    if long_q > 0 and be_long > 0:
        cex_unreal += (price_vol_in_stable - be_long) * long_q
    if short_q > 0 and be_short > 0:
        cex_unreal += (be_short - price_vol_in_stable) * short_q

    total = lp_delta + cex_unreal + realized_delta_since_lp
    return {"total": total, "lp_value": lp_val, "lp_delta": lp_delta, "cex_unreal": cex_unreal}


def dd_guard_triggered(total_pnl: Optional[float], max_projected_drawdown_usd: float) -> bool:
    if total_pnl is None:
        return False
    return total_pnl <= -abs(max_projected_drawdown_usd)