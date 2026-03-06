#!/usr/bin/env python3
"""
Lead-Lag — Бэктест по текущему конфигу.
Тестирует РОВНО текущие настройки на топ-парах за N часов.
"""
import json
import os
import sys
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'backtest_cache')


def load_config():
    with open(os.path.join(BASE_DIR, 'config.json')) as f:
        return json.load(f)


# ─── Cache ───────────────────────────────────────────────────────────────────

def cache_path(symbol: str, hours: int) -> str:
    safe = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(CACHE_DIR, f"{safe}_{hours}h.npy")


def load_cached(symbol: str, hours: int, max_age_min=60) -> Optional[np.ndarray]:
    p = cache_path(symbol, hours)
    if not os.path.exists(p):
        return None
    if time.time() - os.path.getmtime(p) > max_age_min * 60:
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def save_cache(symbol: str, hours: int, data: np.ndarray):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path(symbol, hours), data)


# ─── Data ────────────────────────────────────────────────────────────────────

def create_exchange():
    import ccxt
    return ccxt.okx({'options': {'defaultType': 'swap'}})


def fetch_candles(exchange, symbol: str, hours: int) -> Optional[np.ndarray]:
    cached = load_cached(symbol, hours)
    if cached is not None:
        return cached
    since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
    all_candles = []
    try:
        while True:
            batch = exchange.fetch_ohlcv(symbol, '1m', since=since, limit=300)
            if not batch:
                break
            all_candles.extend(batch)
            if len(batch) < 300:
                break
            since = batch[-1][0] + 1
            time.sleep(0.05)
    except Exception as e:
        logger.error(f"Fetch {symbol}: {e}")
        return None
    if len(all_candles) < 30:
        return None
    arr = np.array(all_candles, dtype=float)
    save_cache(symbol, hours, arr)
    return arr


def fetch_top_pairs(exchange, limit: int, min_vol: float) -> List[str]:
    tickers = exchange.fetch_tickers()
    pairs = []
    for symbol, t in tickers.items():
        if not symbol.endswith(':USDT'):
            continue
        if symbol.split('/')[0] == 'BTC':
            continue
        vol = t.get('quoteVolume') or 0
        if not vol:
            vol = (t.get('baseVolume') or 0) * (t.get('last') or 0)
        if vol >= min_vol:
            pairs.append((symbol, vol))
    pairs.sort(key=lambda x: -x[1])
    return [p[0] for p in pairs[:limit]]


# ─── Simulation ──────────────────────────────────────────────────────────────

def find_impulses(btc: np.ndarray, threshold: float,
                  force_direction: Optional[str] = None) -> List[Dict]:
    moves = (btc[:, 4] - btc[:, 1]) / btc[:, 1]
    result = []
    for i, move in enumerate(moves):
        if abs(move) >= threshold:
            raw_dir = 'LONG' if move > 0 else 'SHORT'
            direction = force_direction if force_direction in ('LONG', 'SHORT') else raw_dir
            result.append({'index': i, 'direction': direction, 'magnitude': float(move)})
    return result


def simulate(follower: np.ndarray, impulses: List[Dict],
             tp_pct: float, sl_pct: float, hold_max_candles: int,
             fee: float) -> Optional[Dict]:
    n = len(follower)
    pnls, reasons = [], []

    for imp in impulses:
        ei = imp['index'] + 1
        if ei >= n:
            continue
        entry = follower[ei, 1]
        if entry <= 0:
            continue

        direction = imp['direction']
        exit_price, reason = None, 'TIME'

        for j in range(ei, min(ei + hold_max_candles, n)):
            h, l = follower[j, 2], follower[j, 3]
            if direction == 'LONG':
                if (l - entry) / entry <= -sl_pct:
                    exit_price = entry * (1 - sl_pct); reason = 'SL'; break
                if (h - entry) / entry >= tp_pct:
                    exit_price = entry * (1 + tp_pct); reason = 'TP'; break
            else:
                if (h - entry) / entry >= sl_pct:
                    exit_price = entry * (1 + sl_pct); reason = 'SL'; break
                if (entry - l) / entry >= tp_pct:
                    exit_price = entry * (1 - tp_pct); reason = 'TP'; break

        if exit_price is None:
            exit_price = follower[min(ei + hold_max_candles - 1, n - 1), 4]

        if direction == 'LONG':
            pnl = (exit_price - entry) / entry - 2 * fee
        else:
            pnl = (entry - exit_price) / entry - 2 * fee

        pnls.append(pnl)
        reasons.append(reason)

    if not pnls:
        return None

    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    tp_profit = float(abs(wins.sum())) if len(wins) else 0
    sl_loss = float(abs(losses.sum())) if len(losses) else 0
    return {
        'n': len(pnls),
        'wins': int(len(wins)),
        'losses': int(len(losses)),
        'wr': round(len(wins) / len(pnls) * 100, 1),
        'pf': round(tp_profit / sl_loss, 2) if sl_loss > 0 else 999,
        'avg_pct': round(float(arr.mean()) * 100, 3),
        'total_pct': round(float(arr.sum()) * 100, 2),
        'tp': reasons.count('TP'),
        'sl': reasons.count('SL'),
        'time': reasons.count('TIME'),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=48)
    ap.add_argument('--pairs', type=int, default=50)
    args = ap.parse_args()

    cfg = load_config()
    leader_cfg = cfg.get('leader', {})
    trading_cfg = cfg.get('trading', {})
    scanner_cfg = cfg.get('scanner', {})

    threshold  = leader_cfg.get('impulse_threshold', 0.003)
    tp_pct     = trading_cfg.get('tp_pct', 0.5) / 100
    sl_pct     = trading_cfg.get('sl_pct', 0.3) / 100
    hold_sec   = trading_cfg.get('hold_max_sec', 300)
    fee        = trading_cfg.get('fee_pct', 0.05) / 100
    force_dir  = trading_cfg.get('force_direction')
    if force_dir not in ('LONG', 'SHORT'):
        force_dir = None

    # convert hold seconds to 1m candles (min 1)
    hold_candles = max(1, round(hold_sec / 60))

    min_vol = scanner_cfg.get('min_volume_usd', 500000)

    print(f"\n{'='*65}")
    print(f"  Lead-Lag Бэктест — ТЕКУЩИЙ КОНФИГ")
    print(f"{'='*65}")
    print(f"  threshold={threshold*100:.3f}%  TP={tp_pct*100:.2f}%  SL={sl_pct*100:.2f}%")
    print(f"  hold={hold_sec}s (~{hold_candles}m candles)  fee={fee*100:.3f}%")
    print(f"  direction={force_dir or 'BTC-signal'}  lookback={args.hours}h  pairs={args.pairs}")
    print(f"{'='*65}\n")

    exchange = create_exchange()

    logger.info("Загружаю BTC свечи...")
    btc = fetch_candles(exchange, 'BTC/USDT:USDT', args.hours)
    if btc is None:
        print("Ошибка: не удалось загрузить BTC свечи")
        sys.exit(1)
    logger.info(f"BTC: {len(btc)} свечей ({len(btc)/60:.1f}ч)")

    impulses = find_impulses(btc, threshold, force_dir)
    logger.info(f"Найдено импульсов BTC: {len(impulses)} "
                f"(LONG={sum(1 for i in impulses if i['direction']=='LONG')}, "
                f"SHORT={sum(1 for i in impulses if i['direction']=='SHORT')})")

    if len(impulses) < 3:
        print("Слишком мало импульсов. Попробуй снизить threshold или увеличить --hours")
        sys.exit(1)

    logger.info(f"Загружаю топ-{args.pairs} пар...")
    pairs = fetch_top_pairs(exchange, args.pairs, min_vol)
    logger.info(f"Найдено {len(pairs)} пар")

    results_per_pair = {}
    for i, symbol in enumerate(pairs):
        candles = fetch_candles(exchange, symbol, args.hours)
        if candles is None or len(candles) < 30:
            continue
        short = symbol.split('/')[0]
        stats = simulate(candles, impulses, tp_pct, sl_pct, hold_candles, fee)
        if stats:
            results_per_pair[short] = stats
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(pairs)} пар обработано...")
        time.sleep(0.02)

    if not results_per_pair:
        print("Нет результатов")
        sys.exit(1)

    # Aggregate
    all_pnls_pct = []
    total_n = total_wins = total_losses = total_tp = total_sl = total_time = 0
    for s in results_per_pair.values():
        total_n += s['n']
        total_wins += s['wins']
        total_losses += s['losses']
        total_tp += s['tp']
        total_sl += s['sl']
        total_time += s['time']
        # weighted pnls
        for _ in range(s['n']):
            all_pnls_pct.append(s['avg_pct'])

    arr = np.array(all_pnls_pct)
    total_pnl = sum(s['total_pct'] for s in results_per_pair.values())
    wins_sum = sum(s['avg_pct'] * s['wins'] for s in results_per_pair.values() if s['avg_pct'] > 0)
    losses_sum = abs(sum(s['avg_pct'] * s['losses'] for s in results_per_pair.values() if s['avg_pct'] <= 0))
    pf = round(wins_sum / losses_sum, 2) if losses_sum > 0 else 999
    wr = round(total_wins / total_n * 100, 1) if total_n else 0

    print(f"\n{'='*65}")
    print(f"  ИТОГОВЫЕ РЕЗУЛЬТАТЫ ({len(results_per_pair)} пар, {total_n} сделок)")
    print(f"{'='*65}")
    print(f"  WR:     {wr}%")
    print(f"  PF:     {pf}")
    print(f"  Сделок: {total_n}  (TP:{total_tp}  SL:{total_sl}  TIME:{total_time})")
    print(f"  Суммарный PnL: {total_pnl:+.2f}% (по всем парам суммарно)")
    print(f"  Ср. PnL/сделку: {total_pnl/total_n:+.3f}%")

    # Per-pair table (top 20)
    pairs_sorted = sorted(results_per_pair.items(),
                          key=lambda x: -x[1]['total_pct'])

    print(f"\n  {'Пара':<10} {'Сделки':>7} {'WR%':>6} {'PF':>6} {'Avg%':>7} {'Total%':>8} {'TP/SL/T':>10}")
    print(f"  {'-'*58}")
    for sym, s in pairs_sorted[:25]:
        exits = f"{s['tp']}/{s['sl']}/{s['time']}"
        print(f"  {sym:<10} {s['n']:>7} {s['wr']:>5.1f}% {s['pf']:>6.2f} "
              f"{s['avg_pct']:>6.3f}% {s['total_pct']:>7.2f}% {exits:>10}")

    if len(pairs_sorted) > 25:
        rest_pnl = sum(s['total_pct'] for _, s in pairs_sorted[25:])
        print(f"  ... ещё {len(pairs_sorted)-25} пар  (total: {rest_pnl:+.2f}%)")

    # Worst pairs
    print(f"\n  Худшие 5 пар:")
    for sym, s in pairs_sorted[-5:]:
        print(f"  {sym:<10} {s['n']:>4} сделок  WR={s['wr']}%  PF={s['pf']}  total={s['total_pct']:+.2f}%")

    print(f"\n{'='*65}\n")


if __name__ == '__main__':
    main()
