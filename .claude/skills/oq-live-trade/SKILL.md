---
name: oq-live-trade
description: Execute the trained RL rotation policy through the crypto broker — paper or live — idempotent per day
argument-hint: "[--config binance-rl-rotation|hyperliquid-rl-rotation]"
---

# Live Trade

Run the daily RL rotation for a crypto market. The runner rebuilds today's
observation (same features + z-score params as training), maps the policy's
tiers to target weights (× risk_degree), diffs against holdings, and places
market orders. Idempotent: a state file prevents double execution on the same
date unless `--force`.

## Paper (safe, recommended first)

```bash
python -m orange_quant.server --config binance-rl-rotation --once --dry-run
# prints tiers/weights/orders; writes data/live_state/binance.json
```

## Live

```bash
# .env must contain the venue keys (BINANCE_API_KEY/BIANCE_SECRET_KEY or
# HYPERLIQUID_ADDRESS/HYPERLIQUID_PRIVATE_KEY)
python -m orange_quant.server --config binance-rl-rotation --once
```

## Scheduled (Docker)

```bash
docker compose --profile binance-live up -d       # daily at 00:15 UTC
docker compose --profile hl-live up -d
docker compose --profile binance-once up          # one-shot dry run
```

## Checks

- Heartbeat: `cat data/live_state/heartbeat.json` freshness (healthcheck).
- State: `cat data/live_state/binance.json` → date, tiers, weights, orders.
- A failed order (e.g. Binance reduce-only) is blacklisted and skipped next run.
