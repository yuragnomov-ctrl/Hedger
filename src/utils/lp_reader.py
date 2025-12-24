# -*- coding: utf-8 -*-
"""
Onchain LP positions reader with redundant RPC system
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Deque, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Constants - will be loaded from config
Q128 = 340282366920938463463374607431768211456  # 2**128 fallback

# Dependencies for onchain functionality
try:
    from web3 import Web3
    from web3.contract import Contract
    _WEB3_ENABLED = True
except Exception as e:
    log.warning("Failed to import Web3: %s. Onchain reading disabled.", e)
    Web3 = None
    Contract = None
    _WEB3_ENABLED = False

try:
    from src.dex.uniswap_v3_math import get_amounts_for_liquidity
    from src.dex.uniswap_v3_math import tick_to_price
    _ONCHAIN_ENABLED = True
except Exception as e:
    log.warning("Failed to import onchain math: %s. Onchain reading disabled.", e)
    get_amounts_for_liquidity = None
    tick_to_price = None
    _ONCHAIN_ENABLED = False


class OnchainLPReader:
    """
    Читатель LP-позиций из onchain с системой резервирования RPC нод.
    Использует бесплатные ноды по умолчанию, переключается на платную при проблемах.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.bc_cfg = cfg.get("blockchain", {})

        # Список RPC нод
        self.free_rpcs = self.bc_cfg.get("free_rpcs", [
            "https://bsc-dataseed.binance.org",
            "https://bsc-dataseed1.defibit.io",
            "https://bsc-dataseed1.ninicoin.io"
        ])
        self.paid_rpc = self.bc_cfg.get("paid_rpc", "")

        # Timeouts
        self.free_timeout = int(self.bc_cfg.get("free_rpc_timeout_sec", 5))
        self.paid_timeout = int(self.bc_cfg.get("paid_rpc_timeout_sec", 10))

        # Пороги переключения
        self.max_free_failures = int(self.bc_cfg.get("max_free_failures", 2))
        self.health_check_interval = int(self.bc_cfg.get("health_check_interval_sec", 30))

        # Текущее состояние
        self.current_rpc_index = 0
        self.use_paid = False
        self.consecutive_failures = 0
        self.last_health_check = 0.0
        self.rpc_failure_counts: Dict[str, int] = {rpc: 0 for rpc in self.free_rpcs}

        # Web3 подключения
        self.w3_free: Optional[Web3] = None
        self.w3_paid: Optional[Web3] = None
        self.npm: Optional[Contract] = None
        self.factory: Optional[Contract] = None

        # Данные аккаунта
        self.wallet_address = ""
        self.private_key = ""

        # Кэшированные позиции
        self.positions_cache: Dict[int, Dict[str, Any]] = {}
        self.cache_timestamp = 0.0
        self.cache_ttl = int(self.bc_cfg.get("cache_ttl_sec", 30))

        # Инициализация
        self._init_web3_connections()
        self._init_account()

        log.info("[LP_READER] Initialized with %d free RPCs, paid_rpc=%s",
                 len(self.free_rpcs), "enabled" if self.paid_rpc else "disabled")

    def _init_web3_connections(self) -> None:
        """Инициализация Web3 подключений к RPC нодам"""
        try:
            # Подключение к текущей бесплатной ноде
            current_free_rpc = self.free_rpcs[self.current_rpc_index]
            self.w3_free = Web3(Web3.HTTPProvider(current_free_rpc, request_kwargs={'timeout': self.free_timeout}))

            # Подключение к платной ноде если доступна
            if self.paid_rpc:
                self.w3_paid = Web3(Web3.HTTPProvider(self.paid_rpc, request_kwargs={'timeout': self.paid_timeout}))
                log.info("[LP_READER] Connected to paid RPC: %s", self.paid_rpc)

            # Тест подключений
            if self.w3_free and self.w3_free.is_connected():
                log.info("[LP_READER] Connected to free RPC: %s", current_free_rpc)
            else:
                log.warning("[LP_READER] Failed to connect to free RPC: %s", current_free_rpc)

            if self.w3_paid and self.w3_paid.is_connected():
                log.info("[LP_READER] Paid RPC connection OK")
            elif self.paid_rpc:
                log.warning("[LP_READER] Failed to connect to paid RPC")

        except Exception as e:
            log.error("[LP_READER] Web3 connection error: %s", e)

    def _init_account(self) -> None:
        """Инициализация данных аккаунта из конфига"""
        try:
            self.wallet_address = self.bc_cfg.get("wallet_address", "")

            if self.wallet_address:
                # Нормализуем адрес
                if hasattr(Web3, 'to_checksum_address'):
                    self.wallet_address = Web3.to_checksum_address(self.wallet_address)
                log.info("[LP_READER] Wallet address initialized: %s", self.wallet_address)
            else:
                log.error("[LP_READER] Missing wallet_address in config")

        except Exception as e:
            log.error("[LP_READER] Account initialization error: %s", e)

    def _get_w3(self) -> Optional[Web3]:
        """Получить активное Web3 подключение с логикой переключения"""
        now = time.time()

        # Периодическая проверка состояния бесплатных нод
        if self.use_paid and (now - self.last_health_check) > self.health_check_interval:
            self._check_free_rpcs_health()
            self.last_health_check = now

        # Если используем платную, возвращаем её
        if self.use_paid and self.w3_paid and self.w3_paid.is_connected():
            return self.w3_paid

        # Пробуем бесплатные ноды
        for i in range(len(self.free_rpcs)):
            rpc_index = (self.current_rpc_index + i) % len(self.free_rpcs)
            rpc = self.free_rpcs[rpc_index]

            if self.rpc_failure_counts.get(rpc, 0) >= self.max_free_failures:
                log.debug("[LP_READER] Skipping failed RPC: %s", rpc)
                continue

            try:
                # Переподключаемся если нужно
                if self.w3_free is None or self.w3_free.provider.endpoint_uri != rpc:
                    self.w3_free = Web3(Web3.HTTPProvider(rpc, request_kwargs={'timeout': self.free_timeout}))
                    self.current_rpc_index = rpc_index

                if self.w3_free.is_connected():
                    return self.w3_free

            except Exception as e:
                self.rpc_failure_counts[rpc] = self.rpc_failure_counts.get(rpc, 0) + 1
                log.warning("[LP_READER] RPC %s failure #%d: %s", rpc, self.rpc_failure_counts[rpc], e)

        # Если все бесплатные не сработали, переключаемся на платную
        if not self.use_paid and self.w3_paid and self.w3_paid.is_connected():
            log.warning("[LP_READER] All free RPCs failed, switching to paid RPC")
            self.use_paid = True
            return self.w3_paid

        log.error("[LP_READER] No working RPC connections available")
        return None

    def _check_free_rpcs_health(self) -> None:
        """Проверяет состояние бесплатных нод и возвращает на них если работают"""
        if not self.use_paid:
            return

        for i, rpc in enumerate(self.free_rpcs):
            try:
                health_timeout = int(self.bc_cfg.get("health_check_timeout_sec", 3))
                w3_test = Web3(Web3.HTTPProvider(rpc, request_kwargs={'timeout': health_timeout}))
                if w3_test.is_connected():
                    log.info("[LP_READER] Free RPC %s is back online, switching back", rpc)
                    self.use_paid = False
                    self.current_rpc_index = i
                    self.w3_free = w3_test
                    self.consecutive_failures = 0
                    return
            except Exception:
                pass

        log.debug("[LP_READER] No free RPCs available yet, staying on paid")

    def _init_contracts(self) -> bool:
        """Инициализация контрактов NPM и Factory"""
        if not _ONCHAIN_ENABLED or not _WEB3_ENABLED:
            return False

        try:
            w3 = self._get_w3()
            if not w3:
                return False

            # Адреса контрактов (BSC Mainnet - Uniswap V3)
            npm_address = self.bc_cfg.get("npm_address", "0x7b8A01B39D58278b5DE7e48c8449c9f4F5170613")  # Uniswap V3 NonfungiblePositionManager
            factory_address = self.bc_cfg.get("factory_address", "0xdB1d10011AD0Ff90774D0C6Bb92e5C5c8b4461F7") # Uniswap V3 Factory

            # ABI контрактов (JSON формат для web3)
            NPM_ABI = [
                {
                    "type": "function",
                    "name": "balanceOf",
                    "stateMutability": "view",
                    "inputs": [{"name": "owner", "type": "address"}],
                    "outputs": [{"name": "", "type": "uint256"}]
                },
                {
                    "type": "function",
                    "name": "tokenOfOwnerByIndex",
                    "stateMutability": "view",
                    "inputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "index", "type": "uint256"}
                    ],
                    "outputs": [{"name": "", "type": "uint256"}]
                },
                {
                    "type": "function",
                    "name": "positions",
                    "stateMutability": "view",
                    "inputs": [{"name": "tokenId", "type": "uint256"}],
                    "outputs": [
                        {"name": "nonce", "type": "uint256"},
                        {"name": "operator", "type": "address"},
                        {"name": "token0", "type": "address"},
                        {"name": "token1", "type": "address"},
                        {"name": "fee", "type": "uint24"},
                        {"name": "tickLower", "type": "int24"},
                        {"name": "tickUpper", "type": "int24"},
                        {"name": "liquidity", "type": "uint128"},
                        {"name": "feeGrowthInside0LastX128", "type": "uint256"},
                        {"name": "feeGrowthInside1LastX128", "type": "uint256"},
                        {"name": "tokensOwed0", "type": "uint128"},
                        {"name": "tokensOwed1", "type": "uint128"}
                    ]
                }
            ]

            FACTORY_ABI = [
                {
                    "type": "function",
                    "name": "getPool",
                    "stateMutability": "view",
                    "inputs": [
                        {"name": "tokenA", "type": "address"},
                        {"name": "tokenB", "type": "address"},
                        {"name": "fee", "type": "uint24"}
                    ],
                    "outputs": [{"name": "pool", "type": "address"}]
                }
            ]

            POOL_ABI = [
                {
                    "type": "function",
                    "name": "slot0",
                    "stateMutability": "view",
                    "inputs": [],
                    "outputs": [
                        {"name": "sqrtPriceX96", "type": "uint160"},
                        {"name": "tick", "type": "int24"},
                        {"name": "observationIndex", "type": "uint16"},
                        {"name": "observationCardinality", "type": "uint16"},
                        {"name": "observationCardinalityNext", "type": "uint16"},
                        {"name": "feeProtocol", "type": "uint8"},
                        {"name": "unlocked", "type": "bool"}
                    ]
                }
            ]

            self.npm = w3.eth.contract(address=w3.to_checksum_address(npm_address), abi=NPM_ABI)
            self.factory = w3.eth.contract(address=w3.to_checksum_address(factory_address), abi=FACTORY_ABI)
            self.pool_abi = POOL_ABI  # Сохраняем как атрибут класса

            log.info("[LP_READER] Contracts initialized: NPM=%s, Factory=%s", npm_address, factory_address)
            return True

        except Exception as e:
            log.error("[LP_READER] Contract initialization error: %s", e)
            return False

    def read_positions(self, state: Optional["State"] = None) -> Dict[str, Dict[str, float]]:
        """
        Основная функция чтения LP-позиций.
        Возвращает словарь: {symbol: {vol, dex_px, value, price_min, price_max, token_id}}
        """
        if not _ONCHAIN_ENABLED or not self.wallet_address:
            log.error("[LP_READER] Onchain reading disabled or missing wallet config")
            return {}

        # Проверяем кэш
        now = time.time()
        if (now - self.cache_timestamp) < self.cache_ttl and self.positions_cache:
            log.info("[LP_READER] Using cached positions (%d symbols)", len(self.positions_cache))
            return {k: v for k, v in self.positions_cache.items()}

        try:
            # Инициализируем контракты если нужно
            if not self.npm or not self.factory:
                if not self._init_contracts():
                    log.error("[LP_READER] Failed to initialize contracts")
                    return {}

            w3 = self._get_w3()
            if not w3:
                log.error("[LP_READER] No working Web3 connection")
                return {}

            # Получаем количество позиций
            balance = int(self.npm.functions.balanceOf(self.wallet_address).call())
            log.info("[LP_READER] Found %d LP positions", balance)

            positions = {}
            stables = set(self.cfg.get("stablecoins", ["USDT", "USDC", "DAI"]))

            # Читаем каждую позицию
            for i in range(balance):
                try:
                    token_id = int(self.npm.functions.tokenOfOwnerByIndex(self.wallet_address, i).call())
                    pos_data = self._read_single_position(token_id, w3, stables)

                    if pos_data and not pos_data.get("zero_liquidity", False):
                        symbol = pos_data['symbol']
                        positions[symbol] = pos_data

                        # Трекинг LP позиции с начальной стоимостью
                        if state:
                            self._track_lp_position(token_id, pos_data, state)

                        log.info("[LP_READER] Position %s: vol=%.6f, value=%.2f, token_id=%d",
                                symbol, pos_data['vol'], pos_data['value'], token_id)
                    elif pos_data.get("zero_liquidity", False):
                        # Пропускаем zero_liquidity позиции - они обрабатываются отдельно
                        log.debug("[LP_READER] Skipping zero liquidity position %d", token_id)
                        continue

                except Exception as e:
                    log.warning("[LP_READER] Error reading position %d: %s", i, e)
                    continue

            # Проверяем позиции, которые исчезли (архивируем их)
            if state:
                self._archive_disappeared_positions(positions, state)

            # Кэшируем результаты
            self.positions_cache = positions
            self.cache_timestamp = now

            log.info("[LP_READER] Successfully read %d active positions", len(positions))
            return positions

        except Exception as e:
            log.error("[LP_READER] Critical error reading positions: %s", e)
            # При ошибке возвращаем кэшированные данные если есть
            if self.positions_cache:
                log.warning("[LP_READER] Using stale cached data due to error")
                return {k: v for k, v in self.positions_cache.items()}
            return {}

    def _track_lp_position(self, token_id: int, pos_data: Dict[str, float], state: "State") -> None:
        """Отслеживает LP позицию и фиксирует начальные данные"""
        try:
            current_time = time.time()
            symbol = pos_data['symbol']
            current_value = pos_data['value']
            current_vol = pos_data['vol']
            current_price = pos_data['dex_px']

            # Если позиции нет в state, создаем запись
            if token_id not in state.lp_positions:
                state.lp_positions[token_id] = {
                    'symbol': symbol,
                    'initial_timestamp': current_time,
                    'initial_value_usdt': current_value,
                    'initial_vol_amount': current_vol,
                    'initial_price': current_price,
                    'last_seen_timestamp': current_time,
                    'status': 'active',
                    # CEX PnL baselines (заполним, когда узнаем bybit_symbol в run_once)
                    'cex_symbol': None,
                    'cex_realized_baseline': None,
                    'cex_realized_last': None,
                    'entries': [{
                        'timestamp': current_time,
                        'value_usdt': current_value,
                        'vol_amount': current_vol,
                        'price': current_price,
                        'event_type': 'entry'
                    }],
                    'exits': []
                }
                state.active_positions_count += 1
                log.info("[LP_TRACKER] New position tracked: %s (token_id=%d) initial_value=%.2f",
                        symbol, token_id, current_value)
            else:
                # Обновляем данные существующей позиции
                lp_info = state.lp_positions[token_id]
                lp_info['last_seen_timestamp'] = current_time

                # Если статус archived, возвращаем в active
                if lp_info['status'] == 'archived':
                    lp_info['status'] = 'active'
                    state.active_positions_count += 1
                    state.archived_positions_count -= 1

                    # Добавляем entry
                    lp_info['entries'].append({
                        'timestamp': current_time,
                        'value_usdt': current_value,
                        'vol_amount': current_vol,
                        'price': current_price,
                        'event_type': 're_entry'
                    })

                    log.info("[LP_TRACKER] Position re-activated: %s (token_id=%d) value=%.2f",
                            symbol, token_id, current_value)

        except Exception as e:
            log.error("[LP_TRACKER] Error tracking position %d: %s", token_id, e)

    def _archive_disappeared_positions(self, current_positions: Dict[str, Dict[str, float]], state: "State") -> None:
        """Архивирует позиции, которые исчезли с blockchain"""
        try:
            current_token_ids = set()
            for pos_data in current_positions.values():
                current_token_ids.add(pos_data['token_id'])

            # Проверяем все отслеживаемые позиции
            for token_id, lp_info in list(state.lp_positions.items()):
                if lp_info['status'] == 'active' and token_id not in current_token_ids:
                    # Позиция исчезла - архивируем
                    lp_info['status'] = 'archived'
                    lp_info['exit_timestamp'] = time.time()

                    # Добавляем запись о выходе (используем последние известные данные)
                    lp_info['exits'].append({
                        'timestamp': lp_info['exit_timestamp'],
                        'value_usdt': lp_info.get('last_value_usdt', 0),
                        'vol_amount': lp_info.get('last_vol_amount', 0),
                        'price': lp_info.get('last_price', 0),
                        'event_type': 'exit'
                    })

                    state.active_positions_count -= 1
                    state.archived_positions_count += 1

                    symbol = lp_info.get('symbol', 'UNKNOWN')
                    log.info("[LP_TRACKER] Position archived: %s (token_id=%d) - disappeared from blockchain",
                            symbol, token_id)

        except Exception as e:
            log.error("[LP_TRACKER] Error archiving disappeared positions: %s", e)

    def _read_single_position(self, token_id: int, w3: Web3, stables: set) -> Optional[Dict[str, float]]:
        """Читает отдельную LP-позицию"""
        try:
            # Получаем данные позиции
            pos_data = self.npm.functions.positions(token_id).call()
            nonce, operator, token0, token1, fee, tick_lower, tick_upper, liquidity, fee_growth_inside0_last_x128, fee_growth_inside1_last_x128, tokens_owed0, tokens_owed1 = pos_data

            if int(liquidity) == 0:
                log.warning("[LP_READER] Position %d has zero liquidity - returning special object", token_id)
                # Возвращаем специальный объект с признаком zero liquidity
                return {
                    "zero_liquidity": True,
                    "token_id": token_id,
                    "status": "zero_liquidity"
                }

            # Получаем адрес пула
            pool_address = self.factory.functions.getPool(token0, token1, fee).call()
            if pool_address == "0x0000000000000000000000000000000000000000":
                log.warning("[LP_READER] Position %d has no valid pool", token_id)
                return None

            # Получаем данные пула
            pool = w3.eth.contract(address=w3.to_checksum_address(pool_address), abi=self.pool_abi)

            sqrt_price_x96, current_tick, *_ = pool.functions.slot0().call()

            # Получаем decimals сначала
            ERC20_ABI = [
                {"type": "function", "name": "symbol", "stateMutability": "view", "inputs": [], "outputs": [{"name": "", "type": "string"}]},
                {"type": "function", "name": "decimals", "stateMutability": "view", "inputs": [], "outputs": [{"name": "", "type": "uint8"}]}
            ]
            token0_contract = w3.eth.contract(address=w3.to_checksum_address(token0), abi=ERC20_ABI)
            token1_contract = w3.eth.contract(address=w3.to_checksum_address(token1), abi=ERC20_ABI)

            symbol0 = token0_contract.functions.symbol().call()
            symbol1 = token1_contract.functions.symbol().call()

            try:
                decimals0 = token0_contract.functions.decimals().call()
            except:
                decimals0 = 18
            try:
                decimals1 = token1_contract.functions.decimals().call()
            except:
                decimals1 = 18

            # Вычисляем текущие количества токенов (нужно импортировать get_amounts_for_liquidity)
            if get_amounts_for_liquidity:
                # Правильный расчет sqrt цен (как в lp_monitor_bot.py)
                from math import sqrt
                price_lower = Decimal(1.0001) ** Decimal(tick_lower)
                price_upper = Decimal(1.0001) ** Decimal(tick_upper)
                sqrt_price_lower = int((price_lower.sqrt()) * (2 ** 96))  # Используем Q96
                sqrt_price_upper = int((price_upper.sqrt()) * (2 ** 96))  # Используем Q96

                try:
                    amount0_raw, amount1_raw = get_amounts_for_liquidity(
                        int(liquidity),
                        int(sqrt_price_x96),
                        sqrt_price_lower,  # Уже умножено на Q96
                        sqrt_price_upper   # Уже умножено на Q96
                    )

                    # Конвертируем с учетом decimals (как в lp_monitor_bot.py)
                    amount0 = float(amount0_raw / (Decimal(10) ** decimals0))
                    amount1 = float(amount1_raw / (Decimal(10) ** decimals1))
                except Exception as e:
                    log.warning("[LP_READER] Error calculating amounts for position %d: %s", token_id, e)
                    return None
            else:
                # Fallback если не удалось импортировать функцию
                amount0 = amount1 = 0.0

            # Символы и decimals уже получены выше

            # Определяем волатильный токен (нормализуем регистр)
            is0_stable = symbol0.upper() in stables
            is1_stable = symbol1.upper() in stables

            if is0_stable and not is1_stable:
                vol_symbol = symbol1.upper()  # Нормализуем в UPPER
                vol_amount = amount1
            elif is1_stable and not is0_stable:
                vol_symbol = symbol0.upper()  # Нормализуем в UPPER
                vol_amount = amount0
            else:
                # Если оба или ни один не стейбл, берем токен0
                vol_symbol = symbol0.upper()  # Нормализуем в UPPER
                vol_amount = amount0

            if vol_amount <= 0:
                log.debug("[LP_READER] Position %d has zero volatile amount", token_id)
                return None

            # Вычисляем цену из тиков (как в lp_monitor_bot.py)
            if tick_to_price:
                p_now = float(tick_to_price(Decimal(current_tick)))
                p_min = float(tick_to_price(Decimal(tick_lower)))
                p_max = float(tick_to_price(Decimal(tick_upper)))
            else:
                # Fallback если tick_to_price не доступен
                p_now = p_min = p_max = 0.0

            # Определяем цену volatile токена относительно стейбла (как в lp_monitor_bot.py)
            if is0_stable and not is1_stable:
                # token0 - стейбл, token1 - volatile. ИНВЕРТИРУЕМ все цены
                current_price = 1.0 / p_now if p_now > 0 else 0.0
                price_lower = 1.0 / p_max if p_max > 0 else 0.0  # инвертируем!
                price_upper = 1.0 / p_min if p_min > 0 else 0.0  # инвертируем!
            else:
                # token1 - стейбл, token0 - volatile. Цена как есть
                current_price = p_now
                price_lower = p_min
                price_upper = p_max

            # Вычисляем стоимость БЕЗ учета накопленных комиссий (как в lp_monitor_bot.py)
            # Используем amount0 и amount1 которые уже нормализованы по decimals
            if current_price > 0:
                if is0_stable and not is1_stable:
                    # token0 - стейбл, token1 - volatile
                    value_without_fees = amount0 + amount1 * current_price
                elif is1_stable and not is0_stable:
                    # token1 - стейбл, token0 - volatile
                    value_without_fees = amount1 + amount0 * current_price
                else:
                    # Оба или ни один не стейбл - грубая оценка
                    value_without_fees = amount0 + amount1 * current_price
            else:
                value_without_fees = 0.0

            # Логируем детальную информацию о позиции с реальными балансами
            stable_amount = amount0 if is0_stable else amount1
            stable_symbol = symbol0 if is0_stable else symbol1
            volatile_amount = amount1 if is0_stable else amount0
            volatile_symbol = symbol1 if is0_stable else symbol0

            log.info("[LP_READER] Position %s (ID:%d): range [%.6f-%.6f], price=%.6f, %.6f %s + %.6f %s, value=%.2f$",
                    vol_symbol, token_id,
                    price_lower, price_upper,
                    current_price,
                    stable_amount, stable_symbol,
                    volatile_amount, volatile_symbol,
                    value_without_fees)

            # Дополнительная информация для отладки
            log.debug("[LP_READER] Position details: symbol0=%s, symbol1=%s, decimals=%d/%d, is0_stable=%s",
                     symbol0, symbol1, decimals0, decimals1, is0_stable)

            # Логируем конвертацию символов для отладки
            log.debug("[LP_READER] Symbol conversion: %s/%s → volatile=%s (normalized to UPPER)",
                     symbol0, symbol1, vol_symbol)

            return {
                'symbol': vol_symbol,
                'vol': vol_amount,
                'dex_px': current_price,
                'value': value_without_fees,  # Правильная стоимость в USD без комиссий
                'price_min': price_lower,
                'price_max': price_upper,
                'token_id': token_id,
                # Расширенные метаданные для точного расчета LP стоимости
                'liquidity': int(liquidity),
                'tick_lower': int(tick_lower),
                'tick_upper': int(tick_upper),
                'sqrt_price_lower_x96': int(sqrt_price_lower) if 'sqrt_price_lower' in locals() else 0,
                'sqrt_price_upper_x96': int(sqrt_price_upper) if 'sqrt_price_upper' in locals() else 0,
                'sqrt_price_x96': int(sqrt_price_x96),
                'is0_stable': bool(is0_stable),
                'decimals0': int(decimals0),
                'decimals1': int(decimals1),
                'symbol0': symbol0,
                'symbol1': symbol1
            }

        except Exception as e:
            log.error("[LP_READER] Error reading position %d: %s", token_id, e)
            return None


def read_targets_from_lp(cfg: Dict[str, Any], lp_reader: Optional[OnchainLPReader] = None, state: Optional["State"] = None) -> Dict[str, Dict[str, float]]:
    """
    Читает LP-позиции из onchain через OnchainLPReader.
    Заменяет read_targets_from_csv().

    Возвращает словарь вида:
        { symbol: { 'vol': vol_amount, 'dex_px': dex_price_hint, 'value': position_value, 'price_min': lower_bound, 'price_max': upper_bound, 'token_id': position_id } }
    """
    if not _ONCHAIN_ENABLED:
        log.error("[LP_TARGETS] Onchain dependencies not available")
        return {}

    # Создаем reader если не передан
    if lp_reader is None:
        lp_reader = OnchainLPReader(cfg)

    try:
        # Читаем позиции из onchain с трекингом
        positions = lp_reader.read_positions(state)

        # Преобразуем в формат совместимый с хеджером
        out: Dict[str, Dict[str, float]] = {}
        for symbol, data in positions.items():
            if data.get('vol', 0) > 0:  # Только позиции с волатильным объемом
                out[symbol.upper()] = {
                    'vol': float(data['vol']),
                    'dex_px': float(data['dex_px']) if data.get('dex_px', 0) > 0 else 0.0,
                    'value': float(data.get('value', 0.0)),
                    'price_min': float(data.get('price_min', 0.0)),
                    'price_max': float(data.get('price_max', 0.0)),
                    'token_id': int(data.get('token_id', 0)),
                    'liquidity': int(data.get('liquidity', 0)),
                    'sqrt_price_lower_x96': int(data.get('sqrt_price_lower_x96', 0)),
                    'sqrt_price_upper_x96': int(data.get('sqrt_price_upper_x96', 0)),
                    'decimals0': int(data.get('decimals0', 18)),
                    'decimals1': int(data.get('decimals1', 18)),
                    'is0_stable': bool(data.get('is0_stable', False))
                }

        log.info("[LP_TARGETS] Successfully processed %d symbols from onchain: %s",
                 len(out), list(out.keys()))
        return out

    except Exception as e:
        log.error("[LP_TARGETS] Error reading onchain positions: %s", e)
        return {}