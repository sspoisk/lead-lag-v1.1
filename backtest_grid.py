#!/usr/bin/env python3
"""
Lead-Lag Trader — Grid Search Backtest

Downloads BTC + top alt 1m candles from OKX,
runs grid search over all key parameters,
reports best combinations.

Usage:
    python backtest_grid.py                    # defaults: 48h, top 50 pairs
    python backtest_grid.py --hours 72 --pairs 100
    python backtest_grid.py --quick            # fast: 12h, top 20 pairs
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import ccxt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Cache ───────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'backtest_cache')


def _cache_path(symbol: str, hours: int) -> str:
    safe = symbol.replace('/', '_').replace(':', '_')
    return os.path.join(CACHE_DIR, f"{safe}_{hours}h.npy")


def load_cached(symbol: str, hours: int, max_age_min: int = 60) -> Optional[np.ndarray]:
    path = _cache_path(symbol, hours)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > max_age_min * 60:
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def save_cache(symbol: str, hours: int, data: np.ndarray):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(_cache_path(symbol, hours), data)


# ─── Data fetching ───────────────────────────────────────────────────────────

def create_exchange() -> ccxt.Exchange:
    return ccxt.okx({'options': {'defaultType': 'swap'}})


def fetch_candles(exchange: ccxt.Exchange, symbol: str,
                  hours: int = 48) -> Optional[np.ndarray]:
    """Fetch 1m candles with pagination. Returns [ts, o, h, l, c, vol]."""
    cached = load_cached(symbol, hours)
    if cached is not None:
        return cached

    since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
    all_candles = []
    batch = 300

    try:
        while True:
            candles = exchange.fetch_ohlcv(symbol, '1m', since=since, limit=batch)
            if not candles:
                break
            all_candles.extend(candles)
            if len(candles) < batch:
                break
            since = candles[-1][0] + 1
            time.sleep(0.05)
    except Exception as e:
        logger.error(f"Fetch {symbol}: {e}")
        return None

    if len(all_candles) < 30:
        return None

    arr = np.array(all_candles, dtype=float)
    save_cache(symbol, hours, arr)
    return arr


def fetch_top_pairs(exchange: ccxt.Exchange, limit: int = 50,
                    min_vol: float = 500000) -> List[str]:
    """Top USDT swap pairs by volume, excluding BTC."""
    tickers = exchange.fetch_tickers()
    pairs = []
    for symbol, t in tickers.items():
        if not symbol.endswith(':USDT'):
            continue
        base = symbol.split('/')[0]
        if base == 'BTC':
            continue
        vol = t.get('quoteVolume') or 0
        if not vol:
            vol = (t.get('baseVolume') or 0) * (t.get('last') or 0)
        if vol >= min_vol:
            pairs.append((symbol, vol))

    pairs.sort(key=lambda x: -x[1])
    return [p[0] for p in pairs[:limit]]


# ─── Simulation engine ──────────────────────────────────────────────────────

def find_impulses(btc_candles: np.ndarray, threshold: float) -> List[Dict]:
    """Find BTC 1m candles with |move| >= threshold."""
    opens = btc_candles[:, 1]
    closes = btc_candles[:, 4]
    moves = (closes - opens) / opens
    impulses = []
    for i, move in enumerate(moves):
        if abs(move) >= threshold:
            impulses.append({
                'index': i,
                'direction': 'LONG' if move > 0 else 'SHORT',
                'magnitude': float(move)
            })
    return impulses


def simulate_pair(follower: np.ndarray, impulses: List[Dict],
                  tp_pct: float, sl_pct: float, max_hold: int,
                  fee: float = 0.0005) -> Dict:
    """
    Simulate lead-lag trades on one follower pair.
    tp_pct, sl_pct as decimals (e.g. 0.005 = 0.5%).
    max_hold in candles (minutes).
    Returns stats dict.
    """
    n = len(follower)
    pnls = []

    tp_count = 0
    sl_count = 0
    time_count = 0

    for imp in impulses:
        entry_idx = imp['index'] + 1
        if entry_idx >= n:
            continue

        entry_price = follower[entry_idx, 1]  # open of next candle
        if entry_price <= 0:
            continue

        direction = imp['direction']
        exit_price = None
        reason = None

        for j in range(entry_idx, min(entry_idx + max_hold, n)):
            high = follower[j, 2]
            low = follower[j, 3]

            if direction == 'LONG':
                if (low - entry_price) / entry_price <= -sl_pct:
                    exit_price = entry_price * (1 - sl_pct)
                    reason = 'SL'
                    break
                if (high - entry_price) / entry_price >= tp_pct:
                    exit_price = entry_price * (1 + tp_pct)
                    reason = 'TP'
                    break
            else:
                if (entry_price - high) / entry_price <= -sl_pct:
                    exit_price = entry_price * (1 + sl_pct)
                    reason = 'SL'
                    break
                if (entry_price - low) / entry_price >= tp_pct:
                    exit_price = entry_price * (1 - tp_pct)
                    reason = 'TP'
                    break

        if exit_price is None:
            last_idx = min(entry_idx + max_hold - 1, n - 1)
            exit_price = follower[last_idx, 4]
            reason = 'TIME'

        if direction == 'LONG':
            pnl = (exit_price - entry_price) / entry_price - 2 * fee
        else:
            pnl = (entry_price - exit_price) / entry_price - 2 * fee

        pnls.append(pnl)
        if reason == 'TP':
            tp_count += 1
        elif reason == 'SL':
            sl_count += 1
        else:
            time_count += 1

    if not pnls:
        return None

    returns = np.array(pnls)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    total_profit = float(wins.sum()) if len(wins) else 0
    total_loss = float(abs(losses.sum())) if len(losses) else 0
    pf = total_profit / total_loss if total_loss > 0 else 999
    avg = float(returns.mean())
    std = float(returns.std()) if len(returns) > 1 else 1
    sharpe = avg / std if std > 0 else 0

    return {
        'n_trades': len(pnls),
        'wins': int(len(wins)),
        'losses': int(len(losses)),
        'wr': round(len(wins) / len(pnls) * 100, 1),
        'pf': round(pf, 2),
        'sharpe': round(sharpe, 3),
        'avg_return': round(avg * 100, 3),
        'total_return': round(float(returns.sum()) * 100, 2),
        'tp_count': tp_count,
        'sl_count': sl_count,
        'time_count': time_count
    }


# ─── Grid search ─────────────────────────────────────────────────────────────

DEFAULT_GRID = {
    'impulse_threshold': [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005],
    'tp_pct':            [0.003, 0.004, 0.005, 0.006, 0.008, 0.01],
    'sl_pct':            [0.002, 0.003, 0.004, 0.005, 0.006],
    'hold_max_min':      [3, 5, 7, 10, 15],
}


def run_grid_search(btc_candles: np.ndarray,
                    follower_data: Dict[str, np.ndarray],
                    grid: Dict = None,
                    fee: float = 0.0005) -> List[Dict]:
    """
    Run grid search across all parameter combinations.
    For each combo, aggregates results across ALL follower pairs.
    Returns sorted list of results (best first).
    """
    if grid is None:
        grid = DEFAULT_GRID

    combos = list(product(
        grid['impulse_threshold'],
        grid['tp_pct'],
        grid['sl_pct'],
        grid['hold_max_min']
    ))

    total = len(combos)
    logger.info(f"Grid search: {total} combinations × {len(follower_data)} pairs")

    results = []

    for i, (imp_thresh, tp, sl, hold) in enumerate(combos):
        # Skip absurd combos
        if sl >= tp:
            continue

        impulses = find_impulses(btc_candles, imp_thresh)
        if len(impulses) < 3:
            continue

        all_trades = 0
        all_wins = 0
        all_losses = 0
        all_pnl = 0.0
        all_pnls = []
        all_tp = 0
        all_sl = 0
        all_time = 0
        pairs_with_trades = 0

        for sym, candles in follower_data.items():
            stats = simulate_pair(candles, impulses, tp, sl, hold, fee)
            if stats is None:
                continue
            pairs_with_trades += 1
            all_trades += stats['n_trades']
            all_wins += stats['wins']
            all_losses += stats['losses']
            all_pnl += stats['total_return']
            all_tp += stats['tp_count']
            all_sl += stats['sl_count']
            all_time += stats['time_count']
            # Per-trade pnls for sharpe
            all_pnls.extend([stats['avg_return'] / 100] * stats['n_trades'])

        if all_trades < 10:
            continue

        wr = round(all_wins / all_trades * 100, 1)
        total_profit = sum(p for p in all_pnls if p > 0)
        total_loss_abs = abs(sum(p for p in all_pnls if p <= 0))
        pf = round(total_profit / total_loss_abs, 2) if total_loss_abs > 0 else 999
        avg_ret = round(all_pnl / all_trades, 3)

        arr = np.array(all_pnls)
        sharpe = round(float(arr.mean() / arr.std()), 3) if arr.std() > 0 else 0

        results.append({
            'impulse_threshold': imp_thresh,
            'tp_pct': round(tp * 100, 2),
            'sl_pct': round(sl * 100, 2),
            'hold_max_min': hold,
            'n_impulses': len(impulses),
            'n_trades': all_trades,
            'pairs_traded': pairs_with_trades,
            'wins': all_wins,
            'losses': all_losses,
            'wr': wr,
            'pf': pf,
            'sharpe': sharpe,
            'avg_return_pct': avg_ret,
            'total_return_pct': round(all_pnl, 2),
            'tp_exits': all_tp,
            'sl_exits': all_sl,
            'time_exits': all_time,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{total} combos tested, "
                        f"{len(results)} valid results")

    # Sort by total_return descending
    results.sort(key=lambda x: -x['total_return_pct'])
    return results


# ─── Output ──────────────────────────────────────────────────────────────────

def print_results(results: List[Dict], top_n: int = 30):
    if not results:
        print("\nNo valid results found.")
        return

    print(f"\n{'='*110}")
    print(f"{'TOP RESULTS':^110}")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Impulse':>8} {'TP%':>6} {'SL%':>6} {'Hold':>5} "
          f"{'Trades':>7} {'Pairs':>6} {'WR%':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'AvgRet%':>8} {'TotRet%':>9} {'TP/SL/T':>10}")
    print(f"{'-'*110}")

    for i, r in enumerate(results[:top_n]):
        imp = f"{r['impulse_threshold']*100:.2f}%"
        exits = f"{r['tp_exits']}/{r['sl_exits']}/{r['time_exits']}"
        print(f"{i+1:>3} {imp:>8} {r['tp_pct']:>5.2f}% {r['sl_pct']:>5.2f}% "
              f"{r['hold_max_min']:>4}m {r['n_trades']:>7} {r['pairs_traded']:>6} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} {r['sharpe']:>7.3f} "
              f"{r['avg_return_pct']:>7.3f}% {r['total_return_pct']:>8.2f}% "
              f"{exits:>10}")

    # Summary of best by different metrics
    print(f"\n{'='*110}")
    print("BEST BY METRIC:")

    best_ret = max(results, key=lambda x: x['total_return_pct'])
    best_wr = max(results, key=lambda x: x['wr'])
    best_pf = max(results, key=lambda x: x['pf'] if x['pf'] < 900 else 0)
    best_sharpe = max(results, key=lambda x: x['sharpe'])

    for label, r in [("Total Return", best_ret), ("Win Rate", best_wr),
                      ("Profit Factor", best_pf), ("Sharpe", best_sharpe)]:
        print(f"  {label:>15}: impulse={r['impulse_threshold']*100:.2f}% "
              f"TP={r['tp_pct']:.2f}% SL={r['sl_pct']:.2f}% "
              f"hold={r['hold_max_min']}m → "
              f"WR={r['wr']}% PF={r['pf']} Sharpe={r['sharpe']} "
              f"Return={r['total_return_pct']:.2f}% ({r['n_trades']} trades)")

    # Current config comparison
    try:
        with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
            cfg = json.load(f)
        cur_imp = cfg['leader']['impulse_threshold']
        cur_tp = cfg['trading']['tp_pct']
        cur_sl = cfg['trading']['sl_pct']
        cur_hold = cfg['trading']['hold_max_min']
        print(f"\n  {'CURRENT CONFIG':>15}: impulse={cur_imp*100:.2f}% "
              f"TP={cur_tp:.2f}% SL={cur_sl:.2f}% hold={cur_hold}m")

        # Find current config result
        for r in results:
            if (abs(r['impulse_threshold'] - cur_imp) < 1e-6 and
                abs(r['tp_pct'] - cur_tp) < 0.01 and
                abs(r['sl_pct'] - cur_sl) < 0.01 and
                r['hold_max_min'] == cur_hold):
                rank = results.index(r) + 1
                print(f"  {'CURRENT RANK':>15}: #{rank}/{len(results)} — "
                      f"WR={r['wr']}% PF={r['pf']} Return={r['total_return_pct']:.2f}%")
                break
        else:
            print(f"  {'CURRENT RANK':>15}: not in grid (add to grid to compare)")
    except Exception:
        pass


def save_results(results: List[Dict], hours: int):
    """Save results to JSON."""
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M')
    path = os.path.join(out_dir, f'backtest_grid_{hours}h_{ts}.json')

    output = {
        'timestamp': datetime.utcnow().isoformat(),
        'hours': hours,
        'n_combos_tested': len(results),
        'top_30': results[:30],
        'all_results': results
    }

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved: {path}")
    return path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Lead-Lag Grid Search Backtest')
    parser.add_argument('--hours', type=int, default=48,
                        help='Lookback hours (default: 48)')
    parser.add_argument('--pairs', type=int, default=50,
                        help='Number of top pairs (default: 50)')
    parser.add_argument('--min-vol', type=float, default=500000,
                        help='Min 24h volume USD (default: 500000)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 12h, 20 pairs, smaller grid')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore cached candle data')
    args = parser.parse_args()

    if args.quick:
        args.hours = 12
        args.pairs = 20

    print(f"\n{'='*60}")
    print(f"  Lead-Lag Trader — Grid Search Backtest")
    print(f"  Lookback: {args.hours}h | Pairs: {args.pairs} | MinVol: ${args.min_vol:,.0f}")
    print(f"{'='*60}\n")

    exchange = create_exchange()

    # 1. Fetch BTC candles
    logger.info(f"Fetching BTC/USDT:USDT 1m candles ({args.hours}h)...")
    btc_candles = fetch_candles(exchange, 'BTC/USDT:USDT', args.hours)
    if btc_candles is None:
        logger.error("Failed to fetch BTC candles")
        sys.exit(1)
    logger.info(f"BTC: {len(btc_candles)} candles "
                f"({len(btc_candles)/60:.1f}h)")

    # 2. Fetch top pairs
    logger.info(f"Fetching top {args.pairs} pairs...")
    pairs = fetch_top_pairs(exchange, args.pairs, args.min_vol)
    logger.info(f"Found {len(pairs)} pairs")

    # 3. Fetch follower candles
    follower_data = {}
    for i, symbol in enumerate(pairs):
        candles = fetch_candles(exchange, symbol, args.hours)
        if candles is not None and len(candles) >= 30:
            short = symbol.split('/')[0]
            follower_data[short] = candles
        if (i + 1) % 10 == 0:
            logger.info(f"  Downloaded {i+1}/{len(pairs)} pairs...")
        time.sleep(0.05)

    logger.info(f"Loaded {len(follower_data)} follower pairs")

    # 4. Grid search
    if args.quick:
        grid = {
            'impulse_threshold': [0.0015, 0.002, 0.003, 0.004],
            'tp_pct':            [0.003, 0.005, 0.008],
            'sl_pct':            [0.002, 0.003, 0.005],
            'hold_max_min':      [3, 5, 10],
        }
    else:
        grid = DEFAULT_GRID

    start_time = time.time()
    results = run_grid_search(btc_candles, follower_data, grid)
    elapsed = time.time() - start_time

    logger.info(f"Grid search done in {elapsed:.1f}s: "
                f"{len(results)} valid combinations")

    # 5. Output
    print_results(results)

    # 6. Save
    path = save_results(results, args.hours)
    print(f"\nResults saved to: {path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
