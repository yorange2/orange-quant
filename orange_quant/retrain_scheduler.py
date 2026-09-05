"""Persistent quarterly scheduler for production model retraining.

The scheduler lives in its own container so a long training run cannot block
the daily trading loop.  Successful-period state is stored beside the model,
which makes restarts idempotent when ``models/`` is a Docker volume.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orange_quant.rl.dataset import load_config


UTC = timezone.utc


def period_start(now: datetime, interval_months: int) -> datetime:
    """Return the UTC start of the schedule period containing ``now``."""
    month = ((now.month - 1) // interval_months) * interval_months + 1
    return datetime(now.year, month, 1, tzinfo=UTC)


def due_at(now: datetime, interval_months: int, day: int,
           hour: int, minute: int) -> datetime:
    """Return this period's scheduled UTC instant."""
    start = period_start(now, interval_months)
    return start.replace(day=day, hour=hour, minute=minute)


def next_period(start: datetime, interval_months: int) -> datetime:
    """Advance a period start without requiring dateutil."""
    month_index = start.year * 12 + start.month - 1 + interval_months
    return datetime(month_index // 12, month_index % 12 + 1, 1, tzinfo=UTC)


def period_key(start: datetime) -> str:
    return start.strftime("%Y-%m")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return default


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def _history_covers_period(history_path: Path, start: datetime) -> bool:
    """Use retrain history to bootstrap state when upgrading an existing setup."""
    for item in reversed(_read_json(history_path, [])):
        try:
            stamp = datetime.fromisoformat(item["time"])
        except (KeyError, TypeError, ValueError):
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        if stamp.astimezone(UTC) >= start:
            return True
    return False


def _run_retrain(config: str, train_years: int, valid_months: int,
                 max_epoch: int, stop: threading.Event) -> int | None:
    cmd = [
        sys.executable, "-m", "scripts.retrain_live",
        "--config", config,
        "--train-years", str(train_years),
        "--valid-months", str(valid_months),
        "--max-epoch", str(max_epoch),
    ]
    print(f"[retrain-scheduler] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        if stop.wait(5):
            print("[retrain-scheduler] stopping active retrain", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return None
    return proc.returncode


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_dir = Path(cfg["paths"]["model_dir"])
    state_path = model_dir / "retrain_schedule.json"
    history_path = model_dir / "retrain_history.json"
    state = _read_json(state_path, {})

    stop = threading.Event()

    def request_stop(signum, _frame) -> None:
        print(f"[retrain-scheduler] received signal {signum}", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(
        f"[retrain-scheduler] config={args.config}; every {args.interval_months} "
        f"months on day {args.day} at {args.hour:02d}:{args.minute:02d} UTC",
        flush=True,
    )

    last_announced = None
    while not stop.is_set():
        now = datetime.now(UTC)
        start = period_start(now, args.interval_months)
        due = due_at(now, args.interval_months, args.day, args.hour, args.minute)
        key = period_key(start)

        # Existing installations already have retrain_history.json.  Seed the
        # new scheduler state from it to avoid an unnecessary upgrade-time run.
        if state.get("last_success_period") != key and _history_covers_period(
                history_path, start):
            state["last_success_period"] = key
            state["bootstrapped_from_history"] = True
            _write_json_atomic(state_path, state)

        retry_raw = state.get("next_retry_at")
        try:
            retry_at = datetime.fromisoformat(retry_raw) if retry_raw else None
        except ValueError:
            retry_at = None
        if retry_at is not None and retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)

        completed = state.get("last_success_period") == key
        ready_to_retry = retry_at is None or now >= retry_at
        if now >= due and not completed and ready_to_retry:
            state.update({
                "last_attempt_at": now.isoformat(),
                "last_attempt_period": key,
            })
            _write_json_atomic(state_path, state)
            returncode = _run_retrain(
                args.config, args.train_years, args.valid_months,
                args.max_epoch, stop,
            )
            if returncode is None:
                break
            finished = datetime.now(UTC)
            if returncode == 0:
                state.update({
                    "last_success_at": finished.isoformat(),
                    "last_success_period": key,
                    "last_returncode": 0,
                })
                state.pop("next_retry_at", None)
                print(f"[retrain-scheduler] period {key} completed", flush=True)
            else:
                retry_at = finished + timedelta(hours=args.retry_hours)
                state.update({
                    "last_returncode": returncode,
                    "next_retry_at": retry_at.isoformat(),
                })
                print(
                    f"[retrain-scheduler] failed with {returncode}; retry at "
                    f"{retry_at:%Y-%m-%d %H:%M} UTC",
                    flush=True,
                )
            _write_json_atomic(state_path, state)
            continue

        if not completed and now >= due and retry_at is not None:
            wake_at, label = retry_at, "next retry"
        else:
            wake_at = due if not completed and now < due else next_period(
                start, args.interval_months).replace(
                    day=args.day, hour=args.hour, minute=args.minute)
            label = "next run"
        announced = (label, wake_at)
        if announced != last_announced:
            print(f"[retrain-scheduler] {label}: {wake_at:%Y-%m-%d %H:%M} UTC",
                  flush=True)
            last_announced = announced
        stop.wait(min(args.poll_seconds, max(1, (wake_at - now).total_seconds())))


def main() -> None:
    ap = argparse.ArgumentParser(description="Schedule periodic model retraining")
    ap.add_argument("--config", required=True)
    ap.add_argument("--interval-months", type=int, default=3)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--hour", type=int, default=3)
    ap.add_argument("--minute", type=int, default=0)
    ap.add_argument("--retry-hours", type=int, default=24)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--valid-months", type=int, default=6)
    ap.add_argument("--max-epoch", type=int, default=50)
    args = ap.parse_args()
    if args.interval_months not in (1, 2, 3, 4, 6, 12):
        ap.error("--interval-months must divide 12")
    if not 1 <= args.day <= 28:
        ap.error("--day must be between 1 and 28")
    if not 0 <= args.hour <= 23 or not 0 <= args.minute <= 59:
        ap.error("invalid UTC time")
    run(args)


if __name__ == "__main__":
    main()
