# -*- coding: utf-8 -*-
"""
Bybit Client v1.6.0 (Dual Mode)
================================

Упрощённый клиент для работы с фьючерсной биржей Bybit. Реализует
двухногий хедж (long и short) для одного символа. Для тестирования
поддерживается режим `dry_run`, в котором ордера не отправляются, и
отсутствие установленного модуля ccxt обрабатывается без ошибок.

Основные особенности:

* **Hedge‑mode**: включение режима разделения позиций на long/short.
* **Плечо**: установка плеча для торговой пары.
* **Получение данных**: загрузка стакана и funding rate. В `dry_run`
  возвращаются безопасные значения.
* **Рассчёт позиций**: получение размеров и средних цен для long и
  short позиции; учёт уже уплаченных комиссий в break-even.
* **Лимитные и маркет‑ордера**: размещение ордеров с postOnly/ReduceOnly
  флагами, с учётом размеров шага и минимального лота.
* **BLEED / Flatten**: симметричное закрытие обеих ног или полное
  сворачивание позиций.

Код рассчитан на работу в связке с hedger.py и не охватывает всю
функциональность биржи. При необходимости можно расширять.

v1.6.0 Исправления:
* Enhanced margin calculation protection - добавлены проверки от деления на ноль
  и некорректных значений при расчете использования маржи
* Improved error handling - усилены проверки типов и значений в финансовых расчетах
"""

from __future__ import annotations

import logging
import logging.handlers
import math
import os
import time
from typing import Dict, List, Optional, Tuple

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # модуль может отсутствовать в среде

# Konfiguracja loggera dla BybitClient
def setup_bybit_logger():
    """Konfiguruje logger dla BybitClient z zapisem do pliku."""
    logger = logging.getLogger("bybit")

    # Jeśli logger już ma handlers, nie dodawaj ponownie
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # File handler для логов hedger - ищем подходящую директорию
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Ищем директорию logs в нескольких местах
    possible_log_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'logs'),  # project root/logs
        os.path.join(current_dir, 'logs'),  # current dir/logs
        '/storage/emulated/0/Download/lp_hedge_bot/lp_hedge_bot/logs',  # fallback
    ]

    log_dir = None
    for path in possible_log_dirs:
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.exists(path):
                log_dir = path
                break
        except Exception:
            continue

    if not log_dir:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'logs')
        os.makedirs(log_dir, exist_ok=True)

    # Fixed: Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Use config values or fallbacks for logging
    max_bytes = 10*1024*1024  # 10MB fallback
    backup_count = 5  # fallback
    log_file_name = 'hedger.log'  # fallback

    # Try to get from global config if available
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging_cfg = config.get("logging", {})
        bybit_cfg = logging_cfg.get("bybit", {})
        max_bytes = bybit_cfg.get("max_bytes", max_bytes)
        backup_count = logging_cfg.get("backup_count", backup_count)
        log_file_name = bybit_cfg.get("log_file_name", log_file_name)
    except Exception:
        pass  # Use fallbacks

    log_file = os.path.join(log_dir, log_file_name)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Dodaj handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

log = setup_bybit_logger()

# -----------------------------------------------------------------------------
# Prometheus instrumentation for BybitClient
# -----------------------------------------------------------------------------
# Пытаемся импортировать Counter. При отсутствии — используем заглушку.
try:
    from prometheus_client import Counter  # type: ignore
except Exception:
    class _DummyCounter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass
        def labels(self, *args: object, **kwargs: object) -> "_DummyCounter":
            return self
        def inc(self, *args: object, **kwargs: object) -> None:
            return None
    Counter = _DummyCounter  # type: ignore

# Счётчик ошибок API: endpoint отражает имя вызова (fetch_order_book, fetch_ticker, fetch_funding_rate, fetch_positions)
API_ERROR_COUNTER = Counter(
    "bybit_client_api_errors_total",
    "Количество ошибок при вызовах методов BybitClient",
    ["endpoint"],
)


def _f(x: object, default: float = 0.0) -> float:
    """Преобразует значение в float, возвращая default при ошибке."""
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return default


class BybitClient:
    """Мини‑клиент для работы с Bybit в режиме dual hedge."""

    def __init__(self, api_key: str = "", secret: str = "", dry_run: bool = True, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.dry_run = dry_run
        self.cfg = cfg  # Сохраняем конфиг для использования в других методах
        self.ex = None  # type: ignore

        # Конфигурация ценовых спредов из cfg или значения по умолчанию
        if cfg:
            price_cfg = cfg.get("price_spreads", {})
            self.mock_spread_pct = float(price_cfg.get("mock_spread_pct", 0.05))
            self.market_spread_pct = float(price_cfg.get("market_spread_pct", 0.10))
            self.be_fee_adjustment_pct = float(price_cfg.get("be_fee_adjustment_pct", 0.10))
            self.default_funding_rate = float(price_cfg.get("default_funding_rate", 0.0001))
            # Параметр для получения реальных цен в dry run режиме
            bybit_cfg = cfg.get("bybit", {})
            self.dry_run_real_prices = bool(bybit_cfg.get("dry_run_real_prices", True))
        else:
            # Значения по умолчанию
            self.mock_spread_pct = 0.05
            self.market_spread_pct = 0.10
            self.be_fee_adjustment_pct = 0.10
            self.default_funding_rate = 0.0001
            self.dry_run_real_prices = True

        # Инициализируем клиент если нужно получить реальные цены или не в dry run
        if ccxt is not None and (not self.dry_run or self.dry_run_real_prices):
            try:
                self.ex = ccxt.bybit(
                    {
                        "apiKey": api_key or None,
                        "secret": secret or None,
                        "enableRateLimit": True,
                        "options": {"defaultType": "swap"},
                    }
                )
            except Exception as e:
                log.warning("Failed to initialize ccxt bybit client: %s", e)
                self.ex = None
        # markets cache
        self.markets: Optional[Dict[str, dict]] = None

    # ------------------------------------------------------------------
    # Market metadata
    # ------------------------------------------------------------------
    def load_markets(self) -> None:
        if self.ex is None:
            return
        if self.markets is None:
            try:
                self.markets = self.ex.load_markets()
            except Exception as e:
                log.warning("Failed to load markets: %s", e)
                self.markets = {}

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Нормализует символ под формат ccxt.bybit.

        Принимает "сырые" биржевые символы вида "SOONUSDT" и пытается
        сопоставить их с unified-символом ccxt (например, "SOON/USDT:USDT").

        Ничего не ломает: если соответствия не найдено, возвращает исходную строку.
        """
        if self.ex is None:
            return symbol

        self.load_markets()

        markets = self.markets or {}

        # Если символ уже есть в markets, ничего не меняем
        if symbol in markets:
            return symbol

        # Попробуем через markets_by_id (биржевой ID -> unified символ)
        markets_by_id = getattr(self.ex, "markets_by_id", {}) or {}
        m = markets_by_id.get(symbol)
        if isinstance(m, dict):
            unified = m.get("symbol")
            if isinstance(unified, str) and unified in markets:
                return unified

        # Эвристики для популярных случаев (USDT/USDC/USD‑перпы)
        base = ""
        quote = ""
        suffixes = ("USDT", "USDC", "USD")
        for sfx in suffixes:
            if symbol.endswith(sfx):
                base = symbol[:-len(sfx)]
                quote = sfx
                break

        if base and quote:
            candidates = [
                f"{base}/{quote}:{quote}",   # SOON/USDT:USDT
                f"{base}{quote}:{quote}",    # SOONUSDT:USDT
                f"{base}/{quote}",           # SOON/USDT
                f"{base}{quote}",            # SOONUSDT
            ]
            for cand in candidates:
                if cand in markets:
                    return cand

        return symbol

    def _market_meta(self, symbol: str) -> Tuple[float, float, float]:
        """Возвращает (qty_step, min_qty, tick_size) для символа."""
        # Используем значения из конфига или дефолты
        orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
        qty_step = orders_cfg.get("default_qty_step", 1.0)
        min_qty = orders_cfg.get("default_min_qty", 1.0)
        tick = orders_cfg.get("default_tick_size", 0.01)
        if self.ex is None or self.dry_run:
            return qty_step, min_qty, tick
        symbol = self._normalize_symbol(symbol)
        self.load_markets()
        m = (self.markets or {}).get(symbol) or {}
        info = m.get("info", {}) or {}
        lot = info.get("lotSizeFilter", {}) or {}
        pf = info.get("priceFilter", {}) or {}

        qty_step = _f(lot.get("qtyStep"), 0.0)
        min_qty = _f(lot.get("minOrderQty"), 0.0)
        tick = _f(pf.get("tickSize"), 0.0)

        # если нет инфы, используем precisions
        if qty_step <= 0:
            prec_amt = (m.get("precision") or {}).get("amount")
            if isinstance(prec_amt, (int, float)):
                qty_step = 10 ** (-int(prec_amt))
        if tick <= 0:
            prec_px = (m.get("precision") or {}).get("price")
            if isinstance(prec_px, (int, float)):
                tick = 10 ** (-int(prec_px))

        # запасные значения из конфига
        orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
        if qty_step <= 0:
            qty_step = orders_cfg.get("default_qty_step", 1.0)
        if min_qty <= 0:
            min_qty = qty_step
        if tick <= 0:
            tick = orders_cfg.get("fallback_tick_size", 0.000001)
        return qty_step, min_qty, tick

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_orderbook_and_ticker(self, symbol: str) -> Tuple[float, float, float]:
        """Возвращает (mid, bid, ask). В dry-run получает реальные цены, но не торгует."""
        if self.ex is None:
            # Если биржа недоступна совсем - используем заглушки из конфига
            price_cfg = self.cfg.get("price_mocking", {}) if hasattr(self, 'cfg') and self.cfg else {}
            fallback_prices = price_cfg.get("fallback_prices", {})
            default_price = price_cfg.get("default_price", 100.0)

            symbol_prices = fallback_prices
            # Цена по умолчанию если символа нет в маппинге
            mid = symbol_prices.get(symbol, default_price)
            spread_factor = self.mock_spread_pct / 100.0
            bid = mid * (1 - spread_factor)
            ask = mid * (1 + spread_factor)
            return mid, bid, ask
        symbol = self._normalize_symbol(symbol)
        bid = ask = last = 0.0
        # Попытки с экспоненциальным backoff для стакана
        api_cfg = self.cfg.get("api_retry", {}) if hasattr(self, 'cfg') and self.cfg else {}
        max_attempts = api_cfg.get("max_attempts", 3)
        base_delay = api_cfg.get("base_delay", 0.5)

        for attempt in range(max_attempts):
            try:
                ob = self.ex.fetch_order_book(symbol, limit=5)
                if ob:
                    if ob.get("bids"):
                        bid = _f(ob["bids"][0][0], 0.0)
                    if ob.get("asks"):
                        ask = _f(ob["asks"][0][0], 0.0)
                break
            except Exception as e:
                # регистрируем ошибку для order book
                try:
                    API_ERROR_COUNTER.labels(endpoint="fetch_order_book").inc()
                except Exception:
                    pass
                log.warning("fetch_order_book failed %s (attempt %d): %s", symbol, attempt + 1, e)
                if attempt < max_attempts - 1:
                    max_delay = api_cfg.get("max_delay", 8)
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
        # Попытки с backoff для тикера
        for attempt in range(max_attempts):
            try:
                t = self.ex.fetch_ticker(symbol)
                last = _f(t.get("last"), 0.0)
                break
            except Exception as e:
                try:
                    API_ERROR_COUNTER.labels(endpoint="fetch_ticker").inc()
                except Exception:
                    pass
                log.warning("fetch_ticker failed %s (attempt %d): %s", symbol, attempt + 1, e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
        # Запасные значения, если bid/ask не получены
        if bid <= 0 and last > 0:
            # Используем конфиг для запасного спреда
            price_cfg = self.cfg.get("price_mocking", {}) if hasattr(self, 'cfg') and self.cfg else {}
            default_spread_factor = price_cfg.get("default_spread_factor", 0.001)
            bid = last * (1 - default_spread_factor)
        if ask <= 0 and last > 0:
            spread_factor = self.market_spread_pct / 100.0
            ask = last * (1 + spread_factor)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
        return mid, bid, ask

    def fetch_funding_rate(self, symbol: str) -> float:
        """Возвращает ставку финансирования (доля, например 0.0001=0.01%)."""
        if self.ex is None:
            return self.default_funding_rate
        symbol = self._normalize_symbol(symbol)
        # Попытаемся 3 раза с экспоненциальным backoff
        for attempt in range(3):
            try:
                f = self.ex.fetch_funding_rate(symbol)
                return _f(f.get("fundingRate"), 0.0)
            except Exception as e:
                try:
                    API_ERROR_COUNTER.labels(endpoint="fetch_funding_rate").inc()
                except Exception:
                    pass
                log.warning("fetch_funding_rate failed %s (attempt %d): %s", symbol, attempt + 1, e)
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return 0.0

    # ------------------------------------------------------------------
    # Account / leverage / modes
    # ------------------------------------------------------------------
    def ensure_leverage(self, symbol: str, leverage: float) -> None:
        """Устанавливает плечо. В dry_run ничего не делает."""
        if self.dry_run or self.ex is None or leverage <= 1:
            return
        symbol = self._normalize_symbol(symbol)
        try:
            self.ex.set_leverage(leverage, symbol)
        except Exception as e:
            if "not modified" not in str(e):
                log.warning("Leverage set failed %s: %s", symbol, e)

    def ensure_hedge_mode(self, symbol: str, enabled: bool = True) -> None:
        """Включает hedge‑mode (разделение long/short)."""
        if self.dry_run or self.ex is None:
            return
        symbol = self._normalize_symbol(symbol)
        try:
            if hasattr(self.ex, "set_position_mode"):
                self.ex.set_position_mode(enabled, symbol)
            else:
                # старый интерфейс ccxt
                self.ex.setPositionMode({"hedged": enabled, "symbol": symbol})
        except Exception as e:
            if "not modified" not in str(e):
                log.warning("Hedge-mode failed %s: %s", symbol, e)

    def margin_usage_pct(self) -> Optional[float]:
        """Расчет использования маржи (%). Возвращает None при неудаче API."""
        if self.dry_run or self.ex is None:
            return 0.0

        # Пробуем получить данные от API
        api_result = self._get_margin_from_api()
        if api_result is not None:
            return api_result

        # Если API не работает, возвращаем None (будет использоваться логика по объемам)
        log.warning("Margin API failed, returning None - volume-based BLEED will be used instead")
        return None

    def _get_margin_from_api(self) -> Optional[float]:
        """Попытка получить маржу через API."""
        # Попытаемся несколько раз с экспоненциальным backoff
        for attempt in range(3):
            try:
                bal = self.ex.fetch_balance({"type": "swap"}) or {}
                info = bal.get("info") or {}

                # Пробуем разные поля Bybit API для получения данных о марже
                # Варианты полей для общего баланса маржи
                total_margin_balance = _f(info.get("totalMarginBalance"))
                total_wallet_balance = _f(info.get("totalWalletBalance"))
                account_im = _f(info.get("accountIM"))  # Initial Margin
                total = total_margin_balance or total_wallet_balance or _f(bal.get("total"))

                # Варианты полей для использованной маржи
                used_margin = _f(info.get("totalInitialMargin"))
                account_mm = _f(info.get("accountMM"))  # Maintenance Margin
                used = used_margin or account_im or _f(bal.get("used"))

                # Логируем полученные значения для отладки
                log.info("DEBUG margin_usage: total=%.2f, used=%.2f, totalMB=%.2f, walletB=%.2f, initialM=%.2f, accountIM=%.2f",
                        total, used, total_margin_balance, total_wallet_balance, used_margin, account_im)

                # Проверка корректности данных
                if total and total > 0 and isinstance(total, (int, float)):
                    if used and used >= 0:
                        margin_usage = (used / total) * 100.0
                        # Ограничиваем результат в диапазоне 0-100%
                        result = min(100.0, max(0.0, margin_usage))
                        log.info("API margin calculation successful: %.2f%%", result)
                        return result
                    else:
                        log.warning("Invalid used margin value: %s", used)
                else:
                    log.warning("Invalid total margin value: %s", total)

                # Если основные поля не сработали, пробуем альтернативные методы
                return self._calculate_margin_from_positions()

            except Exception as e:
                try:
                    API_ERROR_COUNTER.labels(endpoint="fetch_balance").inc()
                except Exception:
                    pass
                log.warning("fetch_balance failed (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))

        # Если все попытки неудачны
        log.error("All API attempts failed")
        return None

    
    def _calculate_margin_from_positions(self) -> Optional[float]:
        """Альтернативный расчет использования маржи через данные позиций."""
        if self.dry_run or self.ex is None:
            return 0.0

        try:
            # Получаем баланс счета
            bal = self.ex.fetch_balance({"type": "swap"}) or {}
            info = bal.get("info") or {}

            # Пробуем получить доступный баланс маржи
            available_balance = _f(info.get("availableBalance")) or _f(info.get("totalAvailableBalance"))

            if available_balance and available_balance > 0:
                # Расчет на основе available balance (если available = total - used)
                # Это приблизительный расчет, но лучше чем ничего
                orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
                leverage_multiplier = orders_cfg.get("leverage_multiplier", 4)
                estimated_total = available_balance * leverage_multiplier
                estimated_used = estimated_total - available_balance
                margin_usage = (estimated_used / estimated_total) * 100.0

                log.info("Alternative margin calculation: estimated_total=%.2f, estimated_used=%.2f, usage=%.2f%%",
                        estimated_total, estimated_used, margin_usage)

                return min(100.0, max(0.0, margin_usage))

            # Если не можем получить баланс, возвращаем None
            return None

        except Exception as e:
            log.warning("Alternative margin calculation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Position data
    # ------------------------------------------------------------------
    def _positions(self, symbol: str) -> List[dict]:
        if self.dry_run or self.ex is None:
            return []
        symbol = self._normalize_symbol(symbol)

        # Ensure markets are loaded
        try:
            if not hasattr(self.ex, 'markets') or not self.ex.markets:
                log.info("Loading markets for Bybit...")
                self.ex.load_markets()
        except Exception as e:
            log.warning("Failed to load markets: %s", e)

        # Попытаемся несколько раз получить позиции с экспоненциальным backoff
        for attempt in range(3):
            try:
                positions = self.ex.fetch_positions([symbol]) or []
                log.debug("fetch_positions %s returned %d positions", symbol, len(positions))
                return positions
            except Exception as e:
                # регистрируем ошибку
                try:
                    API_ERROR_COUNTER.labels(endpoint="fetch_positions").inc()
                except Exception:
                    pass
                log.warning("fetch_positions failed %s (attempt %d): %s", symbol, attempt + 1, e)
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        log.error("fetch_positions failed after 3 attempts for %s", symbol)
        return []

    def pos_sizes_dual(self, symbol: str) -> Tuple[float, float]:
        """Возвращает (long_qty, short_qty)."""
        long_q = short_q = 0.0
        if self.dry_run or self.ex is None:
            return 0.0, 0.0
        symbol = self._normalize_symbol(symbol)
        for p in self._positions(symbol):
            if p.get("symbol") != symbol:
                continue
            sz = _f(p.get("contracts") or (p.get("info", {}) or {}).get("size"), 0.0)
            if sz <= 0:
                continue
            side = str(p.get("side") or (p.get("info", {}) or {}).get("side") or "").lower()
            # ccxt в hedged режиме может возвращать side="long" или positionIdx=1
            if side == "long" or p.get("positionIdx") == 1:
                long_q += sz
            elif side == "short" or p.get("positionIdx") == 2:
                short_q += sz
        return long_q, short_q

    def get_realized_pnl(self, symbol: str) -> float:
        """
        Возвращает накопленный реализованный PnL по символу из текущих позиций.

        Returns:
            float: cumulative realized PnL в USD
        """
        if self.dry_run or self.ex is None:
            return 0.0

        symbol = self._normalize_symbol(symbol)
        total_realized_pnl = 0.0

        try:
            positions = self._positions(symbol)
            for p in positions:
                if p.get("symbol") != symbol:
                    continue

                sz = _f(p.get("contracts") or (p.get("info", {}) or {}).get("size"), 0.0)
                if sz == 0:
                    continue

                # Получаем cumulative realized PnL из info
                info = p.get("info", {})
                cum_realized_pnl = _f(info.get("cumRealisedPnl", "0"))
                total_realized_pnl += cum_realized_pnl

                log.debug("Realized PnL for %s (size %.1f): %.2f", symbol, sz, cum_realized_pnl)

        except Exception as e:
            log.warning("Failed to get realized PnL for %s: %s", symbol, e)
            return 0.0

        return total_realized_pnl

    def break_even_prices(self, symbol: str) -> Tuple[float, float]:
        """Возвращает (be_long, be_short). 0 если нет позиции. Учитывает комиссию."""
        be_long = be_short = 0.0
        if self.dry_run or self.ex is None:
            # Используем значения из конфига для break-even
            orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
            default_be_long = orders_cfg.get("default_be_long", 99.5)
            default_be_short = orders_cfg.get("default_be_short", 100.5)
            # фиктивные BE: long куплен по default_be_long, short продан по default_be_short
            return default_be_long, default_be_short
        symbol = self._normalize_symbol(symbol)
        for p in self._positions(symbol):
            if p.get("symbol") != symbol:
                continue
            side = str(p.get("side") or (p.get("info", {}) or {}).get("side") or "").lower()
            entry = _f(
                p.get("entryPrice")
                or (p.get("info", {}) or {}).get("entryPrice")
                or (p.get("info", {}) or {}).get("avgEntryPrice"),
                0.0,
            )
            if entry <= 0:
                continue
            # учтём приблизительно комиссию за обе стороны
            fee_factor = self.be_fee_adjustment_pct / 100.0
            if side == "long" or p.get("positionIdx") == 1:
                be_long = entry * (1 + fee_factor)
            elif side == "short" or p.get("positionIdx") == 2:
                be_short = entry * (1 - fee_factor)
        return be_long, be_short

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def cancel_all_orders(self, symbol: str) -> None:
        """Отменяет все открытые ордера для символа."""
        if self.dry_run or self.ex is None:
            return
        symbol = self._normalize_symbol(symbol)
        try:
            self.ex.cancel_all_orders(symbol)
        except Exception as e:
            if "Alias is not found" not in str(e):
                log.warning("Cancel all failed %s: %s", symbol, e)

    def _round_qty(self, qty: float, step: float, minimum: float) -> float:
        """Округляет количество вниз до шага и проверяет минимальный лот."""
        if step <= 0:
            return qty
        rounded = math.floor(qty / step) * step
        if rounded < minimum:
            return 0.0
        # округлим до нужного знака после запятой
        precision = int(round(-math.log10(step), 0)) if step < 1.0 else 0
        return round(rounded, precision)

    def _place_limit(self, symbol: str, side: str, amt: float, price: float, reduce_only: bool, pos_idx: int) -> bool:
        """Размещает PostOnly лимитный ордер. Возвращает True при успехе."""
        if amt <= 0:
            return True
        log.info(
            "PostOnly %s %s amt=%.6f px=%.6f reduce=%s idx=%d",
            symbol,
            side.upper(),
            amt,
            price,
            reduce_only,
            pos_idx,
        )
        if self.dry_run or self.ex is None:
            return True
        symbol = self._normalize_symbol(symbol)
        params = {"postOnly": True, "timeInForce": "PostOnly", "positionIdx": pos_idx}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            order = self.ex.create_order(symbol, "limit", side, amt, price, params=params)
            # Проверяем статус ордера
            if order and order.get('status') == 'closed':
                filled = _f(order.get('filled', 0))
                log.info("Order %s filled %.6f/%.6f", order.get('id', 'unknown'), filled, amt)
                return True
            elif order and order.get('status') in ['open', 'partially']:
                log.warning("Order %s not fully filled, status: %s, filled: %.6f",
                          order.get('id', 'unknown'), order.get('status'), _f(order.get('filled', 0)))
                # Даем ордеру время на исполнение из конфига
                orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
                status_check_delay = orders_cfg.get("order_status_check_delay", 2)
                time.sleep(status_check_delay)
                # Проверяем еще раз
                try:
                    updated_order = self.ex.fetch_order(order.get('id'), symbol)
                    if updated_order and updated_order.get('status') == 'closed':
                        filled = _f(updated_order.get('filled', 0))
                        log.info("Order %s filled after delay %.6f/%.6f", order.get('id'), filled, amt)
                        return True
                    else:
                        log.error("Order %s still not filled, cancelling", order.get('id'))
                        self.ex.cancel_order(order.get('id'), symbol)
                        return False
                except Exception as retry_e:
                    log.error("Failed to check order status: %s", retry_e)
                    return False
            else:
                log.error("Order creation failed or returned invalid response")
                return False
        except Exception as e:
            log.warning("Limit order failed %s: %s", symbol, e)
            return False

    def create_market_order(self, symbol: str, side: str, amount: float, reduce_only: bool, pos_idx: int) -> bool:
        """Размещает рыночный ордер. Возвращает True при успехе."""
        if amount <= 0:
            return True
        log.info(
            "Market %s %s amt=%.6f reduce=%s idx=%d",
            symbol,
            side.upper(),
            amount,
            reduce_only,
            pos_idx,
        )
        if self.dry_run or self.ex is None:
            return True
        symbol = self._normalize_symbol(symbol)
        params = {"positionIdx": pos_idx}
        if reduce_only:
            params["reduceOnly"] = True
        try:
            self.ex.create_order(symbol, "market", side, amount, params=params)
            return True
        except Exception as e:
            log.warning("Market order failed %s: %s", symbol, e)
            return False

    # ------------------------------------------------------------------
    # High-level adjusters
    # ------------------------------------------------------------------
    def ensure_dual_hedge_adjust_short(self, symbol: str, desired_short: float, maker_offset_pct: float) -> bool:
        """Увеличивает или уменьшает только SHORT‑ногу (positionIdx=2)."""
        qty_step, min_qty, tick = self._market_meta(symbol)
        _, short_q = self.pos_sizes_dual(symbol)
        delta = desired_short - short_q
        amt = self._round_qty(abs(delta), qty_step, min_qty)
        if amt <= 0:
            return True
        _, bid, ask = self.fetch_orderbook_and_ticker(symbol)
        if bid <= 0:
            return False

        log.info("SHORT ADJUSTMENT: target=%.6f, current=%.6f, delta=%.6f, amt=%.6f",
                desired_short, short_q, delta, amt)

        # Сохраняем исходные позиции для проверки
        orig_long, orig_short = self.pos_sizes_dual(symbol)

        self.cancel_all_orders(symbol)
        success = False

        if delta > 0:
            # увеличить SHORT → SELL по цене выше ask
            px = max(
                ask,
                math.ceil((ask * (1 + maker_offset_pct / 100.0)) / tick) * tick,
            )
            success = self._place_limit(symbol, "sell", amt, px, False, 2)
        else:
            # уменьшить SHORT → BUY по цене ниже bid
            px = min(
                bid,
                math.floor((bid * (1 - maker_offset_pct / 100.0)) / tick) * tick,
            )
            success = self._place_limit(symbol, "buy", amt, px, True, 2)

        if success:
            # Даем время на обработку ордера из конфига
            orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
            order_delay = orders_cfg.get("order_execution_delay", 1)
            time.sleep(order_delay)
            # Проверяем фактические позиции
            new_long, new_short = self.pos_sizes_dual(symbol)
            short_diff = new_short - orig_short

            log.info("POSITIONS AFTER SHORT ADJUST: long %.6f→%.6f, short %.6f→%.6f (Δ%.6f)",
                    orig_long, new_long, orig_short, new_short, short_diff)

            # Проверяем, что позиция изменилась в правильную сторону
            if delta > 0 and short_diff > 0:
                # Позиция увеличилась как надо
                if abs(new_short - desired_short) > qty_step * 2:
                    log.warning("SHORT position adjustment incomplete: target=%.6f, actual=%.6f",
                              desired_short, new_short)
                return True
            elif delta < 0 and short_diff < 0:
                # Позиция уменьшилась как надо
                if abs(new_short - desired_short) > qty_step * 2:
                    log.warning("SHORT position adjustment incomplete: target=%.6f, actual=%.6f",
                              desired_short, new_short)
                return True
            else:
                log.error("SHORT position moved in wrong direction! delta=%.6f, actual_change=%.6f",
                         delta, short_diff)
                return False
        else:
            log.error("SHORT order placement failed")
            return False

    def ensure_dual_hedge_adjust_long(self, symbol: str, desired_long: float, maker_offset_pct: float) -> bool:
        """Увеличивает или уменьшает только LONG‑ногу (positionIdx=1)."""
        qty_step, min_qty, tick = self._market_meta(symbol)
        long_q, _ = self.pos_sizes_dual(symbol)
        delta = desired_long - long_q
        amt = self._round_qty(abs(delta), qty_step, min_qty)
        if amt <= 0:
            return True
        _, bid, ask = self.fetch_orderbook_and_ticker(symbol)
        if bid <= 0:
            return False

        log.info("LONG ADJUSTMENT: target=%.6f, current=%.6f, delta=%.6f, amt=%.6f",
                desired_long, long_q, delta, amt)

        # Сохраняем исходные позиции для проверки
        orig_long, orig_short = self.pos_sizes_dual(symbol)

        self.cancel_all_orders(symbol)
        success = False

        if delta > 0:
            # увеличить LONG → BUY по цене ниже bid
            px = min(
                bid,
                math.floor((bid * (1 - maker_offset_pct / 100.0)) / tick) * tick,
            )
            success = self._place_limit(symbol, "buy", amt, px, False, 1)
        else:
            # уменьшить LONG → SELL по цене выше ask
            px = max(
                ask,
                math.ceil((ask * (1 + maker_offset_pct / 100.0)) / tick) * tick,
            )
            success = self._place_limit(symbol, "sell", amt, px, True, 1)

        if success:
            # Даем время на обработку ордера из конфига
            orders_cfg = self.cfg.get("orders", {}) if hasattr(self, 'cfg') and self.cfg else {}
            order_delay = orders_cfg.get("order_execution_delay", 1)
            time.sleep(order_delay)
            # Проверяем фактические позиции
            new_long, new_short = self.pos_sizes_dual(symbol)
            long_diff = new_long - orig_long

            log.info("POSITIONS AFTER LONG ADJUST: long %.6f→%.6f, short %.6f→%.6f (Δ%.6f)",
                    orig_long, new_long, orig_short, new_short, long_diff)

            # Проверяем, что позиция изменилась в правильную сторону
            if delta > 0 and long_diff > 0:
                # Позиция увеличилась как надо
                if abs(new_long - desired_long) > qty_step * 2:
                    log.warning("LONG position adjustment incomplete: target=%.6f, actual=%.6f",
                              desired_long, new_long)
                return True
            elif delta < 0 and long_diff < 0:
                # Позиция уменьшилась как надо
                if abs(new_long - desired_long) > qty_step * 2:
                    log.warning("LONG position adjustment incomplete: target=%.6f, actual=%.6f",
                              desired_long, new_long)
                return True
            else:
                log.error("LONG position moved in wrong direction! delta=%.6f, actual_change=%.6f",
                         delta, long_diff)
                return False
        else:
            log.error("LONG order placement failed")
            return False

    # ------------------------------------------------------------------
    # Flatten / Bleed
    # ------------------------------------------------------------------
    def flatten_both(self, symbol: str, maker_offset_pct: float = 0.02) -> None:
        """Закрывает обе ноги reduceOnly лимитами (полный выход)."""
        qty_step, min_qty, tick = self._market_meta(symbol)
        long_q, short_q = self.pos_sizes_dual(symbol)
        _, bid, ask = self.fetch_orderbook_and_ticker(symbol)
        # закрываем long
        if long_q > 0:
            amt = self._round_qty(long_q, qty_step, min_qty)
            if amt > 0:
                px = max(
                    ask,
                    math.ceil((ask * (1 + maker_offset_pct / 100.0)) / tick) * tick,
                )
                self._place_limit(symbol, "sell", amt, px, True, 1)
        # закрываем short
        if short_q > 0:
            amt = self._round_qty(short_q, qty_step, min_qty)
            if amt > 0:
                px = min(
                    bid,
                    math.floor((bid * (1 - maker_offset_pct / 100.0)) / tick) * tick,
                )
                self._place_limit(symbol, "buy", amt, px, True, 2)

    def bleed_reduce_both(self, symbol: str, fraction: float, base_offset_pct: float) -> None:
        """Пропорционально сокращает обе ноги reduceOnly (BLEED/bubble‑bleed)."""
        qty_step, min_qty, tick = self._market_meta(symbol)
        long_q, short_q = self.pos_sizes_dual(symbol)
        _, bid, ask = self.fetch_orderbook_and_ticker(symbol)
        # сокращаем long
        if long_q > 0:
            amt = self._round_qty(long_q * fraction, qty_step, min_qty)
            if amt > 0:
                px = max(
                    ask,
                    math.ceil((ask * (1 + base_offset_pct / 100.0)) / tick) * tick,
                )
                self._place_limit(symbol, "sell", amt, px, True, 1)
        # сокращаем short
        if short_q > 0:
            amt = self._round_qty(short_q * fraction, qty_step, min_qty)
            if amt > 0:
                px = min(
                    bid,
                    math.floor((bid * (1 - base_offset_pct / 100.0)) / tick) * tick,
                )
                self._place_limit(symbol, "buy", amt, px, True, 2)

    def reduce_equal_amount(self, symbol: str, reduction_usd: float, maker_offset_pct: float = 0.02) -> bool:
        """
        Сокращает обе позиции на равное количество монет, рассчитанное из USD.

        Args:
            symbol: Торговый символ
            reduction_usd: USD сумма для сокращения каждой позиции
            maker_offset_pct: Оффсет для лимитных ордеров

        Returns:
            bool: True если успешно
        """
        try:
            # Получаем текущую цену и позиции
            _, bid, ask = self.fetch_orderbook_and_ticker(symbol)
            if bid <= 0 or ask <= 0:
                log.warning("Cannot reduce positions: invalid bid/ask prices")
                return False

            long_q, short_q = self.pos_sizes_dual(symbol)
            log.info("REDUCE_EQUAL: current positions - long=%.0f, short=%.0f, reduction_usd=%.2f",
                    long_q, short_q, reduction_usd)

            # Рассчитываем количество монет для сокращения
            reduction_coins = reduction_usd / ((bid + ask) / 2)  # средняя цена

            # Проверяем что достаточно монет в обеих позициях
            if long_q < reduction_coins or short_q < reduction_coins:
                log.warning("REDUCE_EQUAL: insufficient positions - long=%.0f, short=%.0f, needed=%.0f",
                           long_q, short_q, reduction_coins)
                return False

            qty_step, min_qty, tick = self._market_meta(symbol)

            # Округляем до шага
            reduction_coins = self._round_qty(reduction_coins, qty_step, min_qty)

            if reduction_coins <= 0:
                log.warning("REDUCE_EQUAL: reduction_coins too small after rounding")
                return False

            # Отменяем все ордера перед размещением новых
            self.cancel_all_orders(symbol)

            success_long = False
            success_short = False

            # Сокращаем LONG (продаем)
            if long_q >= reduction_coins:
                px = max(
                    ask,
                    math.ceil((ask * (1 + maker_offset_pct / 100.0)) / tick) * tick,
                )
                success_long = self._place_limit(symbol, "sell", reduction_coins, px, True, 1)  # positionIdx=1 for long
                log.info("REDUCE_EQUAL: LONG reduction - amount=%.0f, price=%.6f", reduction_coins, px)

            # Сокращаем SHORT (покупаем для закрытия)
            if short_q >= reduction_coins:
                px = min(
                    bid,
                    math.floor((bid * (1 - maker_offset_pct / 100.0)) / tick) * tick,
                )
                success_short = self._place_limit(symbol, "buy", reduction_coins, px, True, 2)  # positionIdx=2 for short
                log.info("REDUCE_EQUAL: SHORT reduction - amount=%.0f, price=%.6f", reduction_coins, px)

            # Проверяем результаты
            if success_long and success_short:
                log.info("REDUCE_EQUAL: SUCCESS - both positions reduced by %.0f coins (%.2f USD each)",
                        reduction_coins, reduction_usd)
                return True
            else:
                log.error("REDUCE_EQUAL: FAILED - long=%s, short=%s", success_long, success_short)
                return False

        except Exception as e:
            log.error("REDUCE_EQUAL: error during reduction - %s", e)
            return False