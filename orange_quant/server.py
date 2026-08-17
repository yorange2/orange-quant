"""Trading server: schedules the RL rotation runner on a daily cron.

Usage:
    python -m orange_quant.server --config binance-rl-rotation --once --dry-run
    python -m orange_quant.server --config binance-rl-rotation --hour 0 --minute 15

The loop wakes every 20s and fires as soon as the configured UTC time has
*passed* and the strategy has not acted yet on that date — so a run missed
while the host slept (Docker suspends the VM; a wall-clock minute window would
be skipped entirely) is caught up on the next tick instead of lost for the day.
It executes the strategy once (idempotent via the runner's state file), writes a
heartbeat file for the Docker HEALTHCHECK, and watches for hangs (a watchdog
thread force-exits on deadlock so Docker restarts it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
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


def _last_acted_date(runner):
    """UTC date the runner last acted on, from its state file (None if unknown).

    Both runners write ``{"date": "YYYY-MM-DD"}`` (UTC) after every run and use
    it for their own idempotency guard; reading it here just avoids re-entering
    a full run_once that would immediately skip.
    """
    path = getattr(runner, "state_file", None)
    if path is None or not Path(path).exists():
        return None
    try:
        raw = json.loads(Path(path).read_text()).get("date")
        return datetime.strptime(raw, "%Y-%m-%d").date() if raw else None
    except (ValueError, TypeError, json.JSONDecodeError, OSError):
        return None  # unreadable/legacy state — treat as never run


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

    # Seed from the runner's state file so a restart later the same day does
    # not redo the (expensive) dataset + policy load just to have run_once
    # skip on its own idempotency check.
    last_fired = None if force else _last_acted_date(runner)
    log.info(f"server: schedule {hour:02d}:{minute:02d} UTC, last acted {last_fired}")

    while True:
        now = datetime.now(timezone.utc)
        due_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # fire on "the time has passed and today has not run yet" rather than
        # on an exact minute match: the window survives a suspended host, and
        # a multi-day gap still fires once (missed days are stale, not queued)
        if now >= due_at and last_fired != now.date():
            _heartbeat()
            # arm the watchdog for this run only: a hung run_once force-exits
            # and Docker's restart policy recovers; the wait loop between runs
            # is not watched (so no spurious hourly restarts)
            done = threading.Event()
            threading.Thread(target=_watchdog, args=(done,), daemon=True).start()
            late = (now - due_at).total_seconds()
            log.info(f"server: firing for {now.date()} ({late / 60:.0f} min after {due_at:%H:%M} UTC)")
            try:
                result = runner.run_once()
                log.info(f"server: run result orders={len(result.get('orders', []))}")
            except Exception:  # noqa: BLE001 - keep the loop alive
                log.exception("server: run_once failed")
            finally:
                done.set()
            # mark the date done either way: a failed run must not re-fire in a
            # tight loop, and the next day's tick recovers on schedule
            last_fired = now.date()
            _heartbeat()
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
