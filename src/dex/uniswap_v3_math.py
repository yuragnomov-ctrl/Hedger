# -*- coding: utf-8 -*-
"""
Uniswap V3 Mathematics - Minimal functions for hedger
Только необходимые математические функции Uniswap V3
"""

from decimal import Decimal
import math

# Константа из Uniswap V3 - will be loaded from config
Q96 = Decimal(79228162514264337593543950336)  # 2**96 fallback


def tick_to_price(tick: int, tick_base: Decimal = None) -> Decimal:
    """
    Конвертирует tick в цену для Uniswap V3

    :param tick: значение tick
    :param tick_base: основание для расчета (по умолчанию 1.0001)
    :return: цена как Decimal
    """
    if tick_base is None:
        tick_base = Decimal(1.0001)  # fallback
    return tick_base ** Decimal(tick)


def get_amounts_for_liquidity(
    liquidity: int,
    sqrt_price_x96: int,
    sqrt_price_lower_x96: int,
    sqrt_price_upper_x96: int,
) -> tuple[Decimal, Decimal]:
    """
    Вычисляет amounts0 и amounts1 для заданной ликвидности

    Формула из whitepaper Uniswap v3:
      если P <= Pa:   весь объём в token0
      если Pa < P < Pb: часть в token0 + часть в token1
      если P >= Pb:   весь объём в token1

    :param liquidity: количество ликвидности
    :param sqrt_price_x96: текущая sqrt price * 2^96
    :param sqrt_price_lower_x96: нижняя граница sqrt price * 2^96
    :param sqrt_price_upper_x96: верхняя граница sqrt price * 2^96
    :return: (amount0, amount1) как Decimal
    """
    L = Decimal(liquidity)
    sp = Decimal(sqrt_price_x96)
    spa = Decimal(sqrt_price_lower_x96)
    spb = Decimal(sqrt_price_upper_x96)

    if sp <= spa:
        # Все в token0
        amount0 = L * (spb - spa) * Q96 / (spa * spb)
        amount1 = Decimal(0)
    elif spa < sp < spb:
        # Распределено между token0 и token1
        amount0 = L * (spb - sp) * Q96 / (sp * spb)
        amount1 = L * (sp - spa) / Q96
    else:
        # Все в token1
        amount0 = Decimal(0)
        amount1 = L * (sp - spa) / Q96

    return amount0, amount1