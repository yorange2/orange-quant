"""Trading server: schedules the RL rotation runner on a daily cron.

Usage:
    python -m orange_quant.server --config binance-rl-rotation --once --dry-run
    python -m orange_quant.server --config binance-rl-rotation --hour 0 --minute 15

The loop sleeps until the configured wall-clock time each day, executes the
strategy once (idempotent via the runner's state file), writes a heartbeat file
for the Docker HEALTHCHECK, and watches for hangs (a watchdog thread force-exits
on deadlock so Docker restarts it).
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("orange-quant")

HEARTBEAT_PATH = "/tmp/oq_heartbeat"  # int epoch, consumed by healthcheck.py
_WATCHDOG_SECONDS = 3600


def _heartbeat() -> None:
    Path(HEARTBEAT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(HEARTBEAT_PATH).write_text(str(int(time.time())))


def _watchdog(done: threading.Event) -> None:
    """Force-exit if the current run hangs (Docker restart policy recovers)."""
    if not done.wait(_WATCHDOG_SECONDS):
        log.error(f"watchdog: no completion within {_WATCHDOG_SECONDS}s, exiting")
        os._exit(1)  # noqa: SLF001 - deliberate hard exit


def run(config_name: str, once: bool, dry_run: bool, force: bool,
        hour: int, minute: int) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info(f"server: config={config_name} dry_run={dry_run} once={once}")

    from orange_quant.rl.dataset import load_config
    from orange_quant.trading import make_broker

    cfg = load_config(config_name)
    broker = make_broker(cfg, "paper" if dry_run else "live")

    if cfg.get("strategy", {}).get("type", "rl") == "lgb":
        from orange_quant.lgb.runner import LGBRotationRunner
        runner = LGBRotationRunner(config_name, broker, force=force)
    else:
        from orange_quant.live import RLRotationRunner
        runner = RLRotationRunner(config_name, broker, force=force)

    if once:
        _heartbeat()
        result = runner.run_once()
        log.info(f"server: run_once result skipped={result.get('skipped', False)} "
                 f"orders={len(result.get('orders', []))}")
        _heartbeat()
        return

    while True:
        now = datetime.now()
        if now.hour == hour and now.minute == minute:
            _heartbeat()
            # arm the watchdog for this run only: a hung run_once force-exits
            # and Docker's restart policy recovers; the wait loop between runs
            # is not watched (so no spurious hourly restarts)
            done = threading.Event()
            threading.Thread(target=_watchdog, args=(done,), daemon=True).start()
            try:
                result = runner.run_once()
                log.info(f"server: run result orders={len(result.get('orders', []))}")
            except Exception:  # noqa: BLE001 - keep the loop alive
                log.exception("server: run_once failed")
            finally:
                done.set()
            _heartbeat()
            # sleep past the minute to avoid double-fire
            time.sleep(61)
        else:
            _heartbeat()  # fresh on every wait tick: healthcheck liveness
            time.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser(description="Orange Quant RL trading server")
    ap.add_argument("--config", required=True, help="config name, e.g. binance-rl-rotation")
    ap.add_argument("--once", action="store_true", help="run once and exit")
    ap.add_argument("--dry-run", action="store_true", help="use PaperBroker")
    ap.add_argument("--force", action="store_true", help="override the state file")
    ap.add_argument("--hour", type=int, default=0, help="daily run hour (UTC)")
    ap.add_argument("--minute", type=int, default=15, help="daily run minute")
    args = ap.parse_args()
    run(args.config, once=args.once, dry_run=args.dry_run, force=args.force,
        hour=args.hour, minute=args.minute)


if __name__ == "__main__":
    main()
