#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEDGER DUAL MODE v1.8.2 - Main Entry Point
=========================================

This is the main entry point that imports from the modular structure.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

# Import core functionality
from src.core.hedger import (
    State, load_state, save_state, run_once, setup_logging, load_config,
    CircuitBreaker, BybitClient, OnchainLPReader
)

log = logging.getLogger(__name__)

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


if __name__ == "__main__":
    main()