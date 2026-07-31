#!/usr/bin/env python3
"""Docker healthcheck: liveness of the trading loop, not just connectivity.

Exits non-zero (unhealthy) if the server's heartbeat file is missing or stale.
The heartbeat is written by orange_quant.server's main loop on every wait tick
and around retrain/rebalance, so a hung loop (e.g. a wedged multiprocessing
pool) stops updating it and is caught here — unlike the old ccxt-ping check,
which passed while the loop was deadlocked.

Run as: python -m orange_quant.healthcheck
"""

import os
import sys
import time

HEARTBEAT_FILE = os.environ.get("OQ_HEARTBEAT_FILE", "/tmp/oq_heartbeat")
TIMEOUT = int(os.environ.get("OQ_WATCHDOG_TIMEOUT", "1800"))


def main() -> int:
    try:
        last = int(open(HEARTBEAT_FILE).read().strip())
    except Exception as e:
        print(f"unhealthy: no readable heartbeat at {HEARTBEAT_FILE} ({e})")
        return 1
    age = int(time.time()) - last
    if age > TIMEOUT:
        print(f"unhealthy: heartbeat stale {age}s (> {TIMEOUT}s)")
        return 1
    print(f"ok: heartbeat {age}s old")
    return 0


if __name__ == "__main__":
    sys.exit(main())
