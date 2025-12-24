# -*- coding: utf-8 -*-
"""
PnL projection and drawdown guard functions
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Import helper functions from utils
from src.utils.helpers import (
    _safe_float,
    _get_lp_info_by_token_id,
    _get_lp_initial_value_usdt,
    _get_cex_realized_delta_since_lp_start,
    lp_value_at_vol_price,
    project_strategy_pnl_at_price,
    dd_guard_triggered
)

# All functions are imported from utils, this module serves as a namespace
# for guard-related functionality

__all__ = [
    '_safe_float',
    '_get_lp_info_by_token_id',
    '_get_lp_initial_value_usdt',
    '_get_cex_realized_delta_since_lp_start',
    'lp_value_at_vol_price',
    'project_strategy_pnl_at_price',
    'dd_guard_triggered'
]