# -*- coding: utf-8 -*-
"""
HEDGER DUAL MODE v1.8.2 - Core Module
================================

This module contains the main hedging logic with proper modular structure.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

# Import core modules
from src.core.state import State, save_state, load_state

# Import utility modules
from src.utils.helpers import (
    _ema, _update_price_history, classify_zone, get_grid_boundaries,
    classify_zone, _get_cex_realized_delta_since_lp_start,
    project_strategy_pnl_at_price, dd_guard_triggered
)
from src.utils.lp_reader import OnchainLPReader, read_targets_from_lp

# Import guard functions
from src.guards.pnl_projection import (
    _safe_float, _get_lp_info_by_token_id, _get_lp_initial_value_usdt,
    lp_value_at_vol_price
)

# Import exchange client
try:
    from src.exchange.bybit_client import BybitClient
except ImportError:
    try:
        from bybit_client import BybitClient
    except ImportError:
        log.error("Failed to import BybitClient")
        sys.exit(1)

# Try to import prometheus for metrics
try:
    from prometheus_client import Counter, Gauge, start_http_server
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# Web3 dependencies for onchain reading
try:
    from web3 import Web3
    from src.dex.uniswap_v3_math import get_amounts_for_liquidity, tick_to_price
    _ONCHAIN_ENABLED = True
except Exception as e:
    logging.warning("Failed to import onchain dependencies: %s. Onchain reading disabled.", e)
    _ONCHAIN_ENABLED = False

# Constants - imported from utils
from src.utils.helpers import Q96, EPS, MAX_ZERO_VOL_CYCLES_DEFAULT, init_constants_from_config

# Setup logging
def setup_logging(log_cfg: Dict[str, Any]) -> None:
    """Настраивает ротацию логов и уровень детализации."""
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_dir = log_cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Формат с timestamp
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Обработчик с ротацией
    max_bytes = log_cfg.get("max_file_size_mb", 10) * 1024 * 1024
    backup_count = log_cfg.get("backup_count", 5)
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        os.path.join(log_dir, "hedger.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setFormatter(formatter)

    # Консольный вывод
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # Глобальный конфиг
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.addHandler(console)


def load_config(path: str) -> Dict[str, Any]:
    """Загружает YAML-конфиг с проверкой обязательных секций."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Проверяем наличие обязательных секций
    required = ["trading", "target", "trigger_market"]
    missing = [s for s in required if s not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    return cfg


class CircuitBreaker:
    """Ограничивает суммарный оборот и комиссии за час."""
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.max_volume = float(cfg.get("max_volume_usd_1h", 5000.0))
        self.max_commission = float(cfg.get("max_commission_usd_1h", 10.0))
        self.reset_interval = int(cfg.get("reset_interval_sec", 3600))
        self.last_reset = time.time()
        self.volume_usd = 0.0
        self.commission_usd = 0.0

    def add_trade(self, volume_usd: float, commission_usd: float) -> None:
        now = time.time()
        if now - self.last_reset >= self.reset_interval:
            self.volume_usd = 0.0
            self.commission_usd = 0.0
            self.last_reset = now
        self.volume_usd += volume_usd
        self.commission_usd += commission_usd

    def check(self) -> bool:
        now = time.time()
        if now - self.last_reset >= self.reset_interval:
            self.volume_usd = 0.0
            self.commission_usd = 0.0
            self.last_reset = now
        return (
            self.volume_usd < self.max_volume and
            self.commission_usd < self.max_commission
        )


def run_once(cfg: Dict[str, Any], state: State, bb: BybitClient, cb: CircuitBreaker, lp_reader: Optional[OnchainLPReader] = None) -> None:
    """
    Выполняет одну итерацию цикла хеджирования для всех активов из onchain LP-позиций.
    Каждый актив обрабатывается независимо. Функция предполагает, что
    Circuit Breaker разрешил торговлю.
    """
    # Константы для проверки volume и границ
    VOLUME_THRESHOLD = EPS  # 1e-9
    MAX_ZERO_VOL_CYCLES = int(cfg.get("smart_flatten", {}).get("max_zero_vol_cycles", MAX_ZERO_VOL_CYCLES_DEFAULT))
    BOUNDARY_TOLERANCE = float(cfg.get("smart_flatten", {}).get("boundary_tolerance_pct", 1.0)) / 100.0  # 1% по умолчанию

    # Guard конфигурация для проекции PnL
    proj_cfg = cfg.get("pnl_projection", {}) if isinstance(cfg.get("pnl_projection", {}), dict) else {}
    proj_enabled = bool(proj_cfg.get("enabled", False))
    proj_log_each_cycle = bool(proj_cfg.get("log_each_cycle", False))

    guard_enabled = bool(proj_cfg.get("guard_enabled", False))
    guard_scope = str(proj_cfg.get("guard_scope", "grid_only")).lower()
    guard_mode = str(proj_cfg.get("guard_mode", "block")).lower()  # block|flatten
    max_proj_dd = float(proj_cfg.get("max_projected_drawdown_usd", 0.0))

    log.info("run_once: Starting iteration")
    if not cb.check():
        # если сработал Circuit Breaker, увеличиваем метрику и пропускаем итерацию
        try:
            if METRICS_ENABLED:
                CIRCUIT_BREAKER_EVENTS.labels(symbol="global").inc()
        except Exception as e:
            # Fixed: Log metric errors instead of silent failure
            log.debug("Failed to increment circuit breaker metric: %s", e)
        log.warning("Circuit breaker active. Trading halted.")
        return

    # ЭТАП 1: Emergency Mode - проверка осиротевших позиций (выполняется всегда)
    log.info("run_once: Checking for orphaned positions...")
    try:
        handle_orphaned_positions(cfg, state, bb, cb, lp_reader)
    except Exception as e:
        log.warning("Failed to handle orphaned positions: %s", e)

    # ЭТАП 2: Normal Mode - обычное хеджирование (только при наличии целей)
    log.info("run_once: About to read targets from onchain")
    targets = read_targets_from_lp(cfg, lp_reader, state)
    log.info("run_once: Read %d targets from onchain", len(targets))
    if not targets or all(t['dex_px'] == 0 for t in targets.values()):
        log.info("No valid targets from onchain.")
        return

    # Получаем конфигурации
    t_cfg = cfg["target"]
    trig_cfg = cfg["sqrt_p_grid"]
    tm_cfg = cfg["trigger_market"]
    h_cfg = cfg["hedge"]
    b_cfg = cfg["bubble_bleed"]
    tp_sl_cfg = cfg["tp_sl"]
    smart_flatten_cfg = cfg.get("smart_flatten", {})

    hedge_ratio_max = float(h_cfg.get("hedge_ratio_max", 0.95))
    min_notional = float(t_cfg.get("min_notional_usd", 10))
    fee_rate = float(cfg.get("trading", {}).get("fee_rate_pct", 0.06)) / 100.0

    # Получаем текущие позиции и цены для всех символов
    all_positions = {}
    all_prices = {}

    for base in targets:
        symbol = base + cfg["trading"]["symbol_suffix"]
        try:
            # Получаем размер позиций
            long_q = bb.get_position_size(symbol, "Buy")
            short_q = bb.get_position_size(symbol, "Sell")
            all_positions[symbol] = {"long": long_q, "short": short_q}

            # Получаем текущую цену perpetual
            ticker = bb.get_ticker(symbol)
            if not ticker or not ticker.get("last_price"):
                log.warning(f"[{symbol}] Failed to get ticker price")
                continue
            px_perp = float(ticker["last_price"])
            all_prices[symbol] = px_perp

            log.info(f"[{symbol}] Current positions: long={long_q:.6f}, short={short_q:.6f}, price={px_perp:.6f}")

        except Exception as e:
            log.error(f"[{symbol}] Error getting positions/price: {e}")
            continue

    # Основной цикл по каждому активу
    for base, target in targets.items():
        symbol = base + cfg["trading"]["symbol_suffix"]
        symbol_info = cfg["trading"]["symbols"][base]

        # Проверяем, что у нас есть данные по позициям и ценам
        if symbol not in all_positions or symbol not in all_prices:
            log.warning(f"[{symbol}] Missing position or price data, skipping")
            continue

        long_q = all_positions[symbol]["long"]
        short_q = all_positions[symbol]["short"]
        px_perp = all_prices[symbol]

        # Получаем данные из onchain
        dex_px = float(target["dex_px"]) if target.get("dex_px", 0) > 0 else 0.0
        vol = float(target["vol"])
        value = float(target.get("value", 0.0))
        price_min = float(target.get("price_min", 0.0))
        price_max = float(target.get("price_max", 0.0))

        log.info(f"[{symbol}] Target: vol={vol:.6f}, dex_px={dex_px:.6f}, value=${value:.2f}")

        # Расчет параметров хеджирования
        notional = vol * px_perp
        net_long_short = long_q - short_q

        # Расчет baseline PnL
        try:
            be_long = bb.get_break_even(symbol, "Buy")
            be_short = bb.get_break_even(symbol, "Sell")
        except Exception as e:
            log.warning(f"[{symbol}] Error getting break-even prices: {e}")
            continue

        # Проверяем zone permissions
        zone, inv_b = classify_zone(
            long_q, short_q, be_long, be_short, px_perp,
            float(tp_sl_cfg.get("tp_long_pct", 10.0)),
            float(tp_sl_cfg.get("tp_short_pct", 10.0)),
            float(tp_sl_cfg.get("sl_long_pct", 15.0)),
            float(tp_sl_cfg.get("sl_short_pct", 15.0)),
            float(h_cfg.get("long_to_short_threshold_pct", 40.0)),
            b_cfg.get("inv_b_mode", "both"),
            float(b_cfg.get("inv_b_long_pct", 1.0)),
            float(b_cfg.get("inv_b_short_pct", 1.0)),
        )

        # Получаем разрешения для зоны
        zone_permissions = cfg["zones"].get(zone, {})
        can_long = bool(zone_permissions.get("long", False))
        can_short = bool(zone_permissions.get("short", False))
        can_reduce_long = bool(zone_permissions.get("reduce_long", False))
        can_reduce_short = bool(zone_permissions.get("reduce_short", False))

        log.info(f"[{symbol}] Zone: {zone}, inv_b={inv_b}, permissions: long={can_long}, short={can_short}")

        # Пропускаем, если нет разрешений
        if not (can_long or can_short or can_reduce_long or can_reduce_short):
            log.info(f"[{symbol}] No permissions in zone {zone}, skipping")
            continue

        # ==================== SMART HEDGE v1.8.0 ====================
        # Расчет умного коэффициента хеджирования с учетом LP PnL
        base_hedge_ratio = float(h_cfg.get("hedge_ratio", 0.9))

        # Получаем LP PnL и CEX realized PnL
        try:
            lp_pnl = bb.get_unrealized_pnl(symbol)  # Это может быть комбинированный PnL
            realized_pnl = bb.get_realized_pnl(symbol)
        except Exception as e:
            log.warning(f"[{symbol}] Error getting PnL data: {e}")
            lp_pnl = realized_pnl = 0.0

        # Smart Hedge: корректируем коэффициент при убытках
        smart_hedge_cfg = cfg.get("smart_hedge", {})
        if smart_hedge_cfg.get("enabled", True):
            pnl_adjustment_enabled = smart_hedge_cfg.get("pnl_adjustment_enabled", True)
            pnl_threshold_pct = float(smart_hedge_cfg.get("pnl_threshold_pct", 5.0))
            hedge_reduction_pct = float(smart_hedge_cfg.get("hedge_reduction_pct", 0.3))

            if pnl_adjustment_enabled:
                # Общий PnL (LP + CEX)
                total_pnl = lp_pnl + realized_pnl
                pnl_threshold_usd = notional * (pnl_threshold_pct / 100.0)

                if total_pnl < -pnl_threshold_usd:
                    # Убыток превышает порог - уменьшаем коэффициент хеджирования
                    reduction_factor = max(0.5, 1.0 - hedge_reduction_pct)
                    adjusted_ratio = base_hedge_ratio * reduction_factor

                    log.info(f"[{symbol}] Smart hedge: PnL=${total_pnl:.2f} < -${pnl_threshold_usd:.2f}, "
                           f"ratio {base_hedge_ratio:.2f} -> {adjusted_ratio:.2f}")
                    base_hedge_ratio = adjusted_ratio

        # Проверяем directional hedge
        directional_cfg = cfg.get("directional_hedge", {})
        if directional_cfg.get("enabled", False):
            # Получаем цену из предыдущей итерации
            last_px = state.last_px.get(base, px_perp)
            price_change_pct = ((px_perp - last_px) / last_px) * 100.0 if last_px > 0 else 0.0

            directional_threshold = float(directional_cfg.get("threshold_pct", 2.0))
            ratio_up = float(directional_cfg.get("hedge_ratio_on_price_up", 0.95))
            ratio_down = float(directional_cfg.get("hedge_ratio_on_price_down", 0.85))

            if price_change_pct > directional_threshold:
                # Цена выросла значительно - увеличиваем хеджирование
                base_hedge_ratio = min(base_hedge_ratio, ratio_up)
                log.info(f"[{symbol}] Directional hedge: price +{price_change_pct:.2f}%, ratio <= {ratio_up}")
            elif price_change_pct < -directional_threshold:
                # Цена упала значительно - уменьшаем хеджирование
                base_hedge_ratio = min(base_hedge_ratio, ratio_down)
                log.info(f"[{symbol}] Directional hedge: price {price_change_pct:.2f}%, ratio <= {ratio_down}")

        # Проверяем funding guard
        funding_cfg = cfg.get("funding_guard", {})
        if funding_cfg.get("enabled", False):
            try:
                funding_rate = bb.get_funding_rate(symbol)
                funding_threshold = float(funding_cfg.get("negative_threshold_pct", -0.01))
                funding_reduction = float(funding_cfg.get("hedge_reduction_pct", 0.2))

                if funding_rate < funding_threshold:
                    # Отрицательный funding - уменьшаем хеджирование
                    reduction_factor = 1.0 - funding_reduction
                    base_hedge_ratio = base_hedge_ratio * reduction_factor

                    log.info(f"[{symbol}] Funding guard: rate={funding_rate:.4f} < {funding_threshold:.4f}, "
                           f"ratio reduced by {funding_reduction:.0%}")
            except Exception as e:
                log.debug(f"[{symbol}] Error checking funding rate: {e}")

        # Ограничиваем максимальный коэффициент
        base_hedge_ratio = min(base_hedge_ratio, hedge_ratio_max)

        # ==================== РАСЧЕТ ЦЕЛЕЙ ====================
        # Текущий неттинг (long - short)
        current_net = long_q - short_q

        # Целевой неттинг с учетом умного коэффициента
        target_net = vol * base_hedge_ratio

        # Требуемое изменение
        required_delta = target_net - current_net

        log.info(f"[{symbol}] Hedge calculation: current_net={current_net:.6f}, target_net={target_net:.6f}, "
               f"required_delta={required_delta:.6f}, ratio={base_hedge_ratio:.2f}")

        # ==================== TP/SL FLATTEN v1.8.0 ====================
        # LP Boundary-based TP/SL имеет приоритет
        lp_tp_sl_enabled = tp_sl_cfg.get("lp_boundary_enabled", True)
        flatten_triggered = False

        if lp_tp_sl_enabled:
            # Проверяем нахождение цены за границами LP
            if px_perp <= price_min * (1.0 - BOUNDARY_TOLERANCE):
                log.warning(f"[{symbol}] TP/SL: price {px_perp:.6f} <= LP boundary {price_min:.6f} - FLATTENING")
                flatten_triggered = True
                reason = "price_below_lp_min"
            elif px_perp >= price_max * (1.0 + BOUNDARY_TOLERANCE):
                log.warning(f"[{symbol}] TP/SL: price {px_perp:.6f} >= LP boundary {price_max:.6f} - FLATTENING")
                flatten_triggered = True
                reason = "price_above_lp_max"

        # Если LP boundary не сработал, проверяем обычные TP/SL
        if not flatten_triggered:
            tp_sl_check = check_tp_sl_conditions(
                long_q, short_q, be_long, be_short, px_perp,
                tp_sl_cfg, zone, inv_b
            )
            if tp_sl_check["flatten"]:
                flatten_triggered = True
                reason = tp_sl_check["reason"]

        if flatten_triggered:
            try:
                bb.flatten_both(symbol, float(h_cfg.get("maker_offset_pct", 0.01)))
                log.info(f"[{symbol}] TP/SL FLATTEN executed: {reason}")
                state.last_zone[symbol] = zone  # Сохраняем зону для логики
                state.last_px[symbol] = px_perp
                continue
            except Exception as e:
                log.error(f"[{symbol}] Error executing TP/SL flatten: {e}")
                continue

        # ==================== BUBBLE-BLEED ====================
        # Проверяем условия bubble-bleed
        bubble_bleed_cfg = cfg.get("bubble_bleed", {})
        if (bubble_bleed_cfg.get("enabled", True) and inv_b and
            long_q > 0 and short_q > 0 and
            zone_permissions.get("bubble_bleed", True)):

            # Расчет размеров сокращения
            bleed_ratio = float(bubble_bleed_cfg.get("bleed_ratio_pct", 50.0)) / 100.0

            try:
                bleed_long = bb.flatten_position(symbol, "Buy", bleed_ratio)
                bleed_short = bb.flatten_position(symbol, "Sell", bleed_ratio)

                log.info(f"[{symbol}] Bubble-bleed executed: long={bleed_long:.6f}, short={bleed_short:.6f}")
                state.last_inv_bleed_ts[symbol] = time.time()
                state.last_zone[symbol] = zone  # Сохраняем зону
                state.last_px[symbol] = px_perp
                continue
            except Exception as e:
                log.error(f"[{symbol}] Error executing bubble-bleed: {e}")

        # ==================== EMERGENCY BLEED ====================
        # Проверяем использование маржи
        try:
            margin_usage = bb.get_margin_usage()
            bleed_threshold = float(cfg.get("emergency", {}).get("bleed_margin_pct", 80.0)) / 100.0

            if margin_usage > bleed_threshold:
                log.warning(f"[{symbol}] Emergency: margin usage {margin_usage:.1%} > {bleed_threshold:.1%} - BLEED")

                emergency_cfg = cfg.get("emergency", {})
                bleed_ratio = float(emergency_cfg.get("bleed_ratio_pct", 50.0)) / 100.0

                try:
                    bleed_long = bb.flatten_position(symbol, "Buy", bleed_ratio)
                    bleed_short = bb.flatten_position(symbol, "Sell", bleed_ratio)

                    log.info(f"[{symbol}] Emergency BLEED executed: long={bleed_long:.6f}, short={bleed_short:.6f}")
                    continue
                except Exception as e:
                    log.error(f"[{symbol}] Error executing emergency bleed: {e}")
        except Exception as e:
            log.debug(f"[{symbol}] Error checking margin usage: {e}")

        # ==================== √P-GRID ====================
        # Основная стратегия хеджирования
        target_vol = vol * base_hedge_ratio
        target_net = target_vol
        current_net = long_q - short_q

        # Вычисляем требуемое изменение
        required_delta = target_net - current_net

        log.info(f"[{symbol}] Grid target: target_vol={target_vol:.6f}, required_delta={required_delta:.6f}")

        # Проверяем минимальный размер сделки
        if abs(required_delta) * px_perp < min_notional:
            log.info(f"[{symbol}] Required change too small: ${abs(required_delta) * px_perp:.2f} < ${min_notional}")
            continue

        # Выполняем сделку
        trade_executed = False
        try:
            if required_delta > 0:
                # Нужно увеличить net_short (продать)
                # Приоритет: reduce long -> sell short
                if can_reduce_long and long_q > 0:
                    sell_amount = min(required_delta, long_q)
                    success = bb.create_market_order(symbol, "Sell", sell_amount, True)
                    if success:
                        log.info(f"[{symbol}] Grid: reduced long by {sell_amount:.6f}")
                        trade_executed = True
                elif can_short:
                    success = bb.create_market_order(symbol, "Sell", required_delta, False)
                    if success:
                        log.info(f"[{symbol}] Grid: increased short by {required_delta:.6f}")
                        trade_executed = True
            else:
                # Нужно уменьшить net_short (купить)
                buy_amount = abs(required_delta)
                # Приоритет: reduce short -> buy long
                if can_reduce_short and short_q > 0:
                    buy_amount = min(buy_amount, short_q)
                    success = bb.create_market_order(symbol, "Buy", buy_amount, True)
                    if success:
                        log.info(f"[{symbol}] Grid: reduced short by {buy_amount:.6f}")
                        trade_executed = True
                elif can_long:
                    success = bb.create_market_order(symbol, "Buy", abs(required_delta), False)
                    if success:
                        log.info(f"[{symbol}] Grid: increased long by {abs(required_delta):.6f}")
                        trade_executed = True

            if trade_executed:
                # Обновляем circuit breaker
                cb.add_trade(abs(required_delta) * px_perp, abs(required_delta) * px_perp * fee_rate)

                # Обновляем метрики
                try:
                    if METRICS_ENABLED:
                        GRID_TRADES.labels(symbol=symbol).inc()
                except Exception:
                    pass

                log.info(f"[{symbol}] Grid trade executed: delta={required_delta:.6f}, value=${abs(required_delta) * px_perp:.2f}")

        except Exception as e:
            log.error(f"[{symbol}] Error executing grid trade: {e}")

        # Сохраняем состояние
        state.last_zone[symbol] = zone
        state.last_px[symbol] = px_perp


# Вспомогательные функции
def handle_orphaned_positions(cfg: Dict[str, Any], state: State, bb: BybitClient, cb: CircuitBreaker, lp_reader: Optional[OnchainLPReader] = None) -> None:
    """Обрабатывает осиротевшие позиции (Emergency Mode)"""
    emergency_cfg = cfg.get("emergency", {})
    if not emergency_cfg.get("enabled", True):
        return

    try:
        # Получаем все биржевые позиции
        all_positions = get_all_exchange_positions(bb, cfg["trading"]["symbol_suffix"])

        # Получаем текущие LP цели
        current_targets = {}
        if lp_reader:
            current_targets = read_targets_from_lp(cfg, lp_reader, state)

        # Находим осиротевшие позиции
        orphaned = {}
        for symbol, pos_data in all_positions.items():
            base = symbol.replace(cfg["trading"]["symbol_suffix"], "")
            if base not in current_targets:
                # Позиции есть, а LP цели нет - осиротевшая
                if pos_data["long"] > 0 or pos_data["short"] > 0:
                    orphaned[symbol] = pos_data

        if orphaned:
            log.warning(f"Found {len(orphaned)} orphaned positions: {list(orphaned.keys())}")

            flatten_pct = float(emergency_cfg.get("orphaned_flatten_pct", 100.0)) / 100.0
            max_wait_sec = int(emergency_cfg.get("orphaned_max_wait_sec", 300))

            for symbol, pos_data in orphaned.items():
                # Проверяем, как давно позиция осиротевшая
                first_seen = state.get("orphaned_positions", {}).get(symbol, time.time())
                wait_time = time.time() - first_seen

                if symbol not in state.get("orphaned_positions", {}):
                    # Только что обнаружили
                    if "orphaned_positions" not in state:
                        state["orphaned_positions"] = {}
                    state["orphaned_positions"][symbol] = time.time()
                    log.info(f"[{symbol}] Orphaned position detected, waiting {max_wait_sec}s before flatten")
                    continue

                if wait_time < max_wait_sec:
                    log.info(f"[{symbol}] Orphaned position waiting: {wait_time:.0f}s / {max_wait_sec}s")
                    continue

                # Время вышло - делаем flatten
                try:
                    if flatten_pct >= 1.0:
                        bb.flatten_both(symbol, float(cfg.get("hedge", {}).get("maker_offset_pct", 0.01)))
                        log.info(f"[{symbol}] Flattened orphaned position completely")
                    else:
                        # Частичное сокращение
                        if pos_data["long"] > 0:
                            bb.flatten_position(symbol, "Buy", flatten_pct)
                        if pos_data["short"] > 0:
                            bb.flatten_position(symbol, "Sell", flatten_pct)
                        log.info(f"[{symbol}] Reduced orphaned position by {flatten_pct:.0%}")

                    # Удаляем из списка осиротевших
                    if symbol in state.get("orphaned_positions", {}):
                        del state["orphaned_positions"][symbol]

                except Exception as e:
                    log.error(f"[{symbol}] Error flattening orphaned position: {e}")

    except Exception as e:
        log.error(f"Error handling orphaned positions: {e}")


def get_all_exchange_positions(bb: BybitClient, symbol_suffix: str) -> Dict[str, Dict[str, float]]:
    """Получает все позиции с биржи"""
    try:
        positions = {}
        all_symbols = bb.get_all_position_symbols()

        for symbol in all_symbols:
            if not symbol.endswith(symbol_suffix):
                continue

            try:
                long_q = bb.get_position_size(symbol, "Buy")
                short_q = bb.get_position_size(symbol, "Sell")

                if abs(long_q) > EPS or abs(short_q) > EPS:
                    positions[symbol] = {
                        "long": long_q,
                        "short": short_q
                    }
            except Exception as e:
                log.warning(f"Error getting position for {symbol}: {e}")

        return positions
    except Exception as e:
        log.error(f"Error getting all exchange positions: {e}")
        return {}


def check_tp_sl_conditions(
    long_q: float, short_q: float, be_long: float, be_short: float, price: float,
    tp_sl_cfg: Dict[str, Any], zone: str, inv_b: bool
) -> Dict[str, Any]:
    """Проверяет условия TP/SL"""
    result = {"flatten": False, "reason": None}

    try:
        # Текущий unrealized PnL
        pnl_long = (price - be_long) * long_q if be_long > 0 else 0.0
        pnl_short = (be_short - price) * short_q if be_short > 0 else 0.0
        total_pnl = pnl_long + pnl_short

        # Текущая зона
        if zone == "TP_LONG":
            result["flatten"] = True
            result["reason"] = "tp_long_reached"
        elif zone == "TP_SHORT":
            result["flatten"] = True
            result["reason"] = "tp_short_reached"
        elif zone == "SL_LONG":
            result["flatten"] = True
            result["reason"] = "sl_long_reached"
        elif zone == "SL_SHORT":
            result["flatten"] = True
            result["reason"] = "sl_short_reached"

  
    except Exception as e:
        log.warning(f"Error checking TP/SL conditions: {e}")

    return result


# Prometheus metrics
if METRICS_ENABLED:
    CIRCUIT_BREAKER_EVENTS = Counter('hedger_circuit_breaker_events_total', 'Circuit breaker triggers', ['symbol'])
    GRID_TRADES = Counter('hedger_grid_trades_total', 'Grid trades executed', ['symbol'])
    ORPHANED_FLATTENS = Counter('hedger_orphaned_flattens_total', 'Orphaned position flattens', ['symbol'])
else:
    # Mock metrics to avoid AttributeErrors
    class MockCounter:
        def labels(self, **kwargs): return self
        def inc(self, amount=1): pass

    CIRCUIT_BREAKER_EVENTS = MockCounter()
    GRID_TRADES = MockCounter()
    ORPHANED_FLATTENS = MockCounter()


# Global state for graceful shutdown
shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    """Handle SIGTERM for graceful shutdown."""
    global shutdown_requested
    log.warning("Received SIGTERM, initiating graceful shutdown...")
    shutdown_requested = True


def main() -> None:
    """Основная функция с graceful shutdown."""
    # Load configuration
    cfg_path = "config.yaml"
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        print(f"Failed to load config from {cfg_path}: {e}")
        sys.exit(1)

    # Initialize constants from config
    init_constants_from_config(cfg)

    # Setup logging
    setup_logging(cfg.get("logging", {}))
    log.info("=== HEDGER DUAL MODE v1.8.2 STARTED ===")
    log.info("Loaded config from %s", cfg_path)

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    log.info("Signal handlers installed (SIGTERM)")

    # Initialize components
    try:
        state = load_state("state.json")
        bb = BybitClient(cfg["trading"])
        cb = CircuitBreaker(cfg.get("circuit_breaker", {}))

        # Initialize LP reader if onchain reading is enabled
        lp_reader = None
        if cfg.get("blockchain", {}).get("enabled", False):
            lp_reader = OnchainLPReader(cfg)

        log.info("All components initialized successfully")

    except Exception as e:
        log.error("Failed to initialize components: %s", e)
        sys.exit(1)

    # Metrics server (optional)
    prometheus_cfg = cfg.get("prometheus", {})
    if prometheus_cfg.get("enabled", False) and "prometheus_client" in sys.modules:
        try:
            from prometheus_client import start_http_server
            port = int(prometheus_cfg.get("port", 8080))
            start_http_server(port)
            log.info("Prometheus metrics server started on port %d", port)
        except Exception as e:
            log.warning("Failed to start Prometheus server: %s", e)

    # Main trading loop
    loop_interval = int(cfg.get("loop_interval_sec", 5))
    persistence_cfg = cfg.get("persistence", {})
    state_path = persistence_cfg.get("state_file", "state.json")

    log.info("Starting main loop with %ds interval", loop_interval)

    while not shutdown_requested:
        try:
            start_time = time.time()

            # Execute one trading iteration
            run_once(cfg, state, bb, cb, lp_reader)

            # Save state if persistence enabled
            if persistence_cfg.get("enabled", True):
                save_state(state, state_path, cfg)

            # Calculate remaining time to sleep
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_interval - elapsed)

            if sleep_time > 0:
                log.debug("Iteration completed in %.3fs, sleeping %.3fs", elapsed, sleep_time)
                time.sleep(sleep_time)
            else:
                log.warning("Iteration took %.3fs longer than interval %.3fs", elapsed, loop_interval)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
            break
        except Exception as e:
            log.error("Unexpected error in main loop: %s", e, exc_info=True)
            # Sleep a bit to avoid rapid error loops
            time.sleep(loop_interval)

    # Graceful shutdown
    log.info("=== SHUTTING DOWN GRACEFULLY ===")
    try:
        if persistence_cfg.get("enabled", True):
            save_state(state, state_path, cfg)
            log.info("Final state saved")
    except Exception as e:
        log.error("Error saving final state: %s", e)

    log.info("HEDGER DUAL MODE stopped")