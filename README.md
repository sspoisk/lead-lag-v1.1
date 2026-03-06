# Lead-Lag Trader

Cryptocurrency futures trading bot based on the **lead-lag effect**: BTC moves first — altcoins follow within seconds.

When BTC makes a sharp 1-minute candle move (impulse), the bot immediately opens positions in correlated altcoins in the same direction.

---

## Two versions

| | V1 | V1.1 |
|---|---|---|
| Path | `/root/lead_lag_trader/` | `/root/lead_lag_v1.1/` |
| Port | 8086 | 8090 |
| GitHub | [lead-lag-trader-v1-new](https://github.com/sspoisk/lead-lag-trader-v1-new) | [lead-lag-v1.1](https://github.com/sspoisk/lead-lag-v1.1) |
| Orders | Software SL/TP | Exchange SL/TP (attachAlgoOrds) |
| Position open | Sequential fetch_ticker per symbol | Batch fetch_tickers (1 API call) |
| Position size | `position_size` in config | `margin × leverage` in config |
| LIVE trading | Not ready | Ready (API keys UI, balance sync) |
| Status | Running, 6000+ trades, WR~66% | Running, fresh start |

Both bots share the same strategy logic and scanner. V1.1 is the upgraded version ready for live trading on OKX.

---

## How it works

### 1. Scanner (every 20 min)
- Fetches top 240 altcoin pairs by volume on OKX
- Downloads 12h of 1m candles for each pair
- Finds historical BTC impulses (>0.18% per 1m candle)
- Simulates trades: entry at open of next candle after impulse, TP/SL/TIME exit
- Ranks pairs by Profit Factor, then Win Rate
- Activates pairs that meet minimum criteria (WR≥46%, PF≥0.8, trades≥3)

### 2. BTC Impulse Detection (every 1s)
- Fetches last 2 BTC/USDT 1m candles
- Uses the last **closed** candle (index -2) — confirmed, not forming
- If `|close/open - 1| >= impulse_threshold` → impulse detected
- Quality filter: `quality = magnitude × time_gap_factor`
  - Recent impulses (gap < 120s) → factor 0.5 (weak)
  - Old impulses (gap > 900s) → factor 2.0 (strong)
  - If quality < min_quality → skip

### 3. Position Opening
- On accepted impulse: fetch prices for all active pairs in **one batch call**
- Open positions in the same direction as BTC impulse
- V1: software SL/TP monitoring every 1s
- V1.1 PAPER: same as V1
- V1.1 LIVE: `create_orders()` with `attachAlgoOrds` (SL+TP live on exchange)

### 4. Exit conditions
- **TP**: price reaches `entry × (1 + tp_pct)` for LONG
- **SL**: price reaches `entry × (1 - sl_pct)` for LONG
- **TIME**: position held longer than `hold_max_sec` seconds

### 5. Pair lifecycle
- Pairs added by scanner when metrics meet thresholds
- After 5+ live trades: evaluated for degradation (WR < 45% or PF < 1.1 → removed)
- Manual add/remove via UI
- **Export/Import**: save active pairs to JSON, load into another bot instance

---

## Architecture

```
app.py          — Flask web app, background threads, REST API
trader.py       — LeadLagTrader: impulse detection, position management
scanner.py      — PairScanner: finds correlated altcoin pairs
pair_manager.py — PairManager: pair lifecycle (add/remove/degrade)
database.py     — SQLite: trades, pairs, impulses, logs, state
config.json     — All settings (single source of truth)
templates/
  index.html    — Web UI (dark theme, real-time updates)
```

### Background threads
- **ScannerThread** — runs pair scan every `scanner.interval_min` minutes
- **TraderThread** — polls BTC 1m candles every 1s, opens positions on impulse
- **PriceThread** — updates open position prices every 1s, triggers exits

---

## Configuration (`config.json`)

```json
{
  "exchange": "okx",
  "trade_mode": "PAPER",
  "port": 8090,

  "leader": {
    "impulse_threshold": 0.0018,
    "quality": {
      "enabled": true,
      "min_quality": 0.001,
      "gap_weak_sec": 120,
      "gap_strong_sec": 900,
      "gap_min_factor": 0.5,
      "gap_max_factor": 2
    }
  },

  "follower": {
    "max_active": 30,
    "min_wr": 46,
    "min_pf": 0.8,
    "min_trades": 3,
    "rolling_window": 50,
    "degrade_wr": 45,
    "degrade_pf": 1.1
  },

  "trading": {
    "tp_pct": 0.3,
    "sl_pct": 0.3,
    "hold_max_sec": 3,
    "fee_pct": 0.05,
    "margin": 5,
    "leverage": 2,
    "max_positions": 15,
    "initial_balance": 1000.0
  },

  "scanner": {
    "interval_min": 20,
    "lookback_hours": 12,
    "top_pairs": 240,
    "min_volume_usd": 200000
  }
}
```

**position size** = `margin × leverage` = 5 × 2 = **$10 notional** per trade

---

## V1.1: Live trading setup

### 1. Enter API keys in the web UI
Go to Settings → OKX API KEYS → enter Key, Secret, Passphrase → SAVE KEYS

Keys are saved to `.env` file with `chmod 600` permissions. Never stored in config.json.

### 2. Switch to LIVE mode
Header → click **PAPER MODE** button → confirm → restarts in LIVE mode

On LIVE startup:
- Reads real USDT balance from exchange
- Syncs any already-open positions from exchange into bot memory

### 3. Live order flow
For each impulse, V1.1 sends **one batch API call** with all orders:
```
create_orders([
  { symbol, type: market, side: buy/sell, amount,
    attachAlgoOrds: [{ tpTriggerPx, slTriggerPx }] }
  ...up to 20 per batch
])
```
SL and TP are placed on the exchange — survive bot restart.

TIME exit: bot sends a `reduceOnly` market close order.

---

## V1.1: Batch orders vs V1 sequential

**V1** opens positions one by one:
```
for each symbol:
    fetch_ticker(symbol)   ← N API calls
    create_order(symbol)   ← N API calls
```

**V1.1** opens all at once:
```
fetch_tickers([all symbols])  ← 1 API call
create_orders([all orders])   ← 1-2 API calls (up to 20 per batch)
```

For 15 pairs this is ~15× faster entry, which matters at 3-second hold times.

---

## Export / Import pairs

**Export** (from any bot UI): Active Pairs → EXPORT → downloads `pairs_YYYY-MM-DD.json`

**Import** (into any bot UI): Active Pairs → IMPORT → select JSON file → pairs added instantly

File format:
```json
[
  {"symbol": "ETH", "full_symbol": "ETH/USDT:USDT"},
  {"symbol": "SOL", "full_symbol": "SOL/USDT:USDT"}
]
```

Use case: V1 has 29 battle-tested pairs after months of trading → export → import into fresh V1.1 instance to skip the warm-up period.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Bot status, balance, WR |
| GET | `/api/pairs` | Active pairs with live stats |
| GET | `/api/pairs/export` | Export pairs as JSON |
| POST | `/api/pairs/import` | Import pairs from JSON |
| POST | `/api/pair/add` | Add pair manually |
| POST | `/api/pair/remove` | Remove pair |
| GET | `/api/positions` | Open positions |
| GET | `/api/trades` | Trade history (current session) |
| GET | `/api/trades/stats` | Aggregated stats |
| GET | `/api/impulses` | BTC impulse log |
| GET | `/api/settings` | Get config |
| POST | `/api/settings` | Update config (hot reload) |
| POST | `/api/scan/run` | Trigger manual scan |
| POST | `/api/pause` | Pause/resume trading |
| POST | `/api/reset` | Reset stats + new session |
| POST | `/api/close_all` | Close all positions |
| GET | `/api/env` | Check which API keys are set |
| POST | `/api/env` | Save API keys to .env |
| POST | `/api/restart` | Restart via systemd |

---

## Database (SQLite)

Tables: `trades`, `pairs`, `pair_stats`, `impulses`, `bot_logs`, `state`

Key state keys:
- `session_start` — UI tabs show only data from current session
- `balance`, `total_pnl`, `trade_counter` — persisted across restarts

---

## Systemd service

```bash
# V1.1
systemctl status lead-lag-v1.1.service
systemctl restart lead-lag-v1.1.service
journalctl -u lead-lag-v1.1.service -f

# V1
systemctl status lead-lag-trader.service
systemctl restart lead-lag-trader.service
```

---

## Exchange

**OKX** USDT-margined perpetual swaps (`swap` defaultType).

OKX specifics handled in code:
- `quoteVolume` may be `None` → fallback: `baseVolume × price`
- 1m candles max 300 per request
- Instrument ID format: `SYM-USDT-SWAP`
- `attachAlgoOrds` for OCO (TP+SL) on entry order
