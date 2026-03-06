"""
PairScanner — discovers follower pairs correlated with BTC impulses.

Every N minutes:
1. Fetch top pairs by volume from OKX
2. Download 1m candles for lookback period
3. Find BTC impulses (>threshold)
4. For each impulse, measure follower cumulative return at 1-5 min lags
5. Compute WR, PF, Sharpe, best_lag
6. Return ranked list
"""

import json
import os
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt

from database import db, get_gmt2_str

logger = logging.getLogger(__name__)


def load_config() -> Dict:
    with open('config.json', 'r') as f:
        return json.load(f)


class PairScanner:
    def __init__(self, exchange: ccxt.Exchange = None):
        self.config = load_config()
        self.exchange = exchange
        self.last_scan_time = None
        self.last_results: List[Dict] = []

    def _get_exchange(self) -> ccxt.Exchange:
        if self.exchange:
            return self.exchange
        raise RuntimeError("Exchange not set")

    def fetch_top_pairs(self, limit: int = 50, min_volume_usd: float = 500000) -> List[str]:
        """Fetch top USDT-margined swap pairs by volume from OKX."""
        ex = self._get_exchange()
        try:
            tickers = ex.fetch_tickers()
        except Exception as e:
            logger.error(f"[SCANNER] fetch_tickers error: {e}")
            return []

        pairs = []
        for symbol, t in tickers.items():
            if not symbol.endswith(':USDT') and not symbol.endswith('/USDT'):
                continue
            if 'BTC' in symbol.split('/')[0]:
                continue
            vol = t.get('quoteVolume') or 0
            if vol is None or vol == 0:
                base_vol = t.get('baseVolume') or 0
                last = t.get('last') or 0
                vol = base_vol * last
            if vol >= min_volume_usd:
                pairs.append((symbol, vol))

        pairs.sort(key=lambda x: -x[1])
        result = [p[0] for p in pairs[:limit]]
        logger.info(f"[SCANNER] Found {len(result)} pairs with vol>={min_volume_usd}")
        return result

    def fetch_candles(self, symbol: str, timeframe: str = '1m',
                      hours: int = 6) -> Optional[np.ndarray]:
        """Fetch 1m candles. Returns array of [timestamp, open, high, low, close, volume]."""
        ex = self._get_exchange()
        since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
        all_candles = []
        batch_size = 300  # OKX max for 1m candles
        try:
            while True:
                candles = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
                if not candles:
                    break
                all_candles.extend(candles)
                if len(candles) < batch_size:
                    break
                since = candles[-1][0] + 1
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"[SCANNER] fetch_candles {symbol} error: {e}")
            return None

        if len(all_candles) < 10:
            return None
        return np.array(all_candles, dtype=float)

    def find_impulses(self, btc_candles: np.ndarray,
                      threshold: float = 0.003) -> List[Dict]:
        """Find BTC candles with |close/open - 1| > threshold."""
        impulses = []
        opens = btc_candles[:, 1]
        closes = btc_candles[:, 4]
        moves = (closes - opens) / opens

        for i, move in enumerate(moves):
            if abs(move) >= threshold:
                impulses.append({
                    'index': i,
                    'timestamp': int(btc_candles[i, 0]),
                    'direction': 'LONG' if move > 0 else 'SHORT',
                    'magnitude': round(float(move), 6),
                    'open': float(opens[i]),
                    'close': float(closes[i])
                })
        return impulses

    def analyze_pair(self, follower_candles: np.ndarray,
                     impulses: List[Dict],
                     tp_pct: float = 0.005, sl_pct: float = 0.003,
                     max_hold: int = 5) -> Dict:
        """
        For each BTC impulse, simulate entry in follower at next candle.
        Check TP/SL/timeout using candle-level data.
        Returns stats: WR, PF, Sharpe, avg_return, best_lag, trades list.
        """
        n_candles = len(follower_candles)
        trades = []
        fee = self.config.get('trading', {}).get('fee_pct', 0.05) / 100

        for imp in impulses:
            entry_idx = imp['index'] + 1  # enter on next candle after impulse
            if entry_idx >= n_candles:
                continue

            entry_price = follower_candles[entry_idx, 1]  # open of next candle
            if entry_price <= 0:
                continue

            direction = imp['direction']
            exit_price = None
            exit_reason = None
            exit_idx = entry_idx

            # Check candles from entry+0 to entry+max_hold
            for j in range(entry_idx, min(entry_idx + max_hold, n_candles)):
                high = follower_candles[j, 2]
                low = follower_candles[j, 3]
                close = follower_candles[j, 4]

                if direction == 'LONG':
                    # Check SL first (conservative)
                    pnl_low = (low - entry_price) / entry_price
                    if pnl_low <= -sl_pct:
                        exit_price = entry_price * (1 - sl_pct)
                        exit_reason = 'SL'
                        exit_idx = j
                        break
                    # Check TP
                    pnl_high = (high - entry_price) / entry_price
                    if pnl_high >= tp_pct:
                        exit_price = entry_price * (1 + tp_pct)
                        exit_reason = 'TP'
                        exit_idx = j
                        break
                else:  # SHORT
                    pnl_high = (entry_price - high) / entry_price
                    if pnl_high <= -sl_pct:
                        exit_price = entry_price * (1 + sl_pct)
                        exit_reason = 'SL'
                        exit_idx = j
                        break
                    pnl_low = (entry_price - low) / entry_price
                    if pnl_low >= tp_pct:
                        exit_price = entry_price * (1 - tp_pct)
                        exit_reason = 'TP'
                        exit_idx = j
                        break

            # Time exit
            if exit_price is None:
                exit_idx = min(entry_idx + max_hold - 1, n_candles - 1)
                exit_price = follower_candles[exit_idx, 4]  # close of last candle
                exit_reason = 'TIME'

            if direction == 'LONG':
                pnl = (exit_price - entry_price) / entry_price - 2 * fee
            else:
                pnl = (entry_price - exit_price) / entry_price - 2 * fee

            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': round(pnl, 6),
                'reason': exit_reason,
                'hold_candles': exit_idx - entry_idx + 1,
                'impulse_magnitude': imp['magnitude']
            })

        if not trades:
            return {'n_trades': 0, 'wr': 0, 'pf': 0, 'sharpe': 0,
                    'avg_return': 0, 'total_return': 0}

        returns = [t['pnl'] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0
        pf = total_profit / total_loss if total_loss > 0 else 999
        avg = np.mean(returns)
        std = np.std(returns) if len(returns) > 1 else 1
        sharpe = float(avg / std) if std > 0 else 0

        avg_hold = np.mean([t['hold_candles'] for t in trades])

        return {
            'n_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'wr': round(len(wins) / len(trades) * 100, 1),
            'pf': round(pf, 2),
            'sharpe': round(sharpe, 3),
            'avg_return': round(float(avg) * 100, 3),
            'total_return': round(sum(returns) * 100, 2),
            'avg_hold_candles': round(float(avg_hold), 1),
            'trades': trades
        }

    def run_scan(self) -> List[Dict]:
        """Full scan: find best follower pairs for BTC lead-lag."""
        cfg = load_config()
        scanner_cfg = cfg.get('scanner', {})
        leader_cfg = cfg.get('leader', {})
        trading_cfg = cfg.get('trading', {})

        threshold = leader_cfg.get('impulse_threshold', 0.003)
        lookback = scanner_cfg.get('lookback_hours', 6)
        top_n = scanner_cfg.get('top_pairs', 50)
        min_vol = scanner_cfg.get('min_volume_usd', 500000)
        tp = trading_cfg.get('tp_pct', 0.5) / 100
        sl = trading_cfg.get('sl_pct', 0.3) / 100
        max_hold = trading_cfg.get('hold_max_min', 5)

        logger.info(f"[SCANNER] Starting scan: threshold={threshold}, "
                    f"lookback={lookback}h, top_n={top_n}")

        # 1. Fetch BTC candles
        btc_symbol = 'BTC/USDT:USDT'
        btc_candles = self.fetch_candles(btc_symbol, '1m', lookback)
        if btc_candles is None or len(btc_candles) < 30:
            logger.error("[SCANNER] Not enough BTC candles")
            return []

        # 2. Find impulses
        impulses = self.find_impulses(btc_candles, threshold)
        logger.info(f"[SCANNER] Found {len(impulses)} BTC impulses "
                    f"(>{threshold*100:.1f}%) in {lookback}h")
        if len(impulses) < 3:
            logger.warning("[SCANNER] Too few impulses, skipping scan")
            return []

        # 3. Fetch top pairs
        pairs = self.fetch_top_pairs(top_n, min_vol)
        if not pairs:
            return []

        # 4. Analyze each pair
        results = []
        for i, symbol in enumerate(pairs):
            candles = self.fetch_candles(symbol, '1m', lookback)
            if candles is None or len(candles) < 30:
                continue

            stats = self.analyze_pair(candles, impulses, tp, sl, max_hold)
            if stats['n_trades'] >= 3:
                short_sym = symbol.split('/')[0] if '/' in symbol else symbol
                result = {
                    'symbol': symbol,
                    'short_symbol': short_sym,
                    **stats
                }
                results.append(result)

                # Save snapshot to DB
                db.save_pair_stat({
                    'symbol': short_sym,
                    'wr': stats['wr'],
                    'pf': stats['pf'],
                    'sharpe': stats['sharpe'],
                    'lag_seconds': 60,  # 1m candle = 60s
                    'n_trades': stats['n_trades'],
                    'avg_return': stats['avg_return'],
                    'source': 'scanner'
                })

            if (i + 1) % 10 == 0:
                logger.info(f"[SCANNER] Analyzed {i+1}/{len(pairs)} pairs")
            time.sleep(0.1)

        # Sort by PF then WR
        results.sort(key=lambda x: (-x['pf'], -x['wr']))
        self.last_results = results
        self.last_scan_time = get_gmt2_str()

        logger.info(f"[SCANNER] Scan complete: {len(results)} pairs analyzed, "
                    f"top: {results[0]['short_symbol']} PF={results[0]['pf']} "
                    f"WR={results[0]['wr']}%" if results else
                    "[SCANNER] No viable pairs found")

        # Log to DB
        db.log('scan', f'Scan complete: {len(results)} pairs', {
            'n_impulses': len(impulses),
            'n_pairs_analyzed': len(pairs),
            'n_viable': len(results),
            'top_3': [{'s': r['short_symbol'], 'wr': r['wr'], 'pf': r['pf']}
                      for r in results[:3]]
        })

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    import ccxt
    ex = ccxt.okx({'options': {'defaultType': 'swap'}})
    scanner = PairScanner(exchange=ex)
    results = scanner.run_scan()
    print(f"\n{'='*60}")
    print(f"Found {len(results)} viable pairs:")
    print(f"{'Symbol':<12} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'Return%':>8}")
    print(f"{'-'*60}")
    for r in results[:20]:
        print(f"{r['short_symbol']:<12} {r['n_trades']:>6} {r['wr']:>5.1f}% "
              f"{r['pf']:>6.2f} {r['sharpe']:>7.3f} {r['total_return']:>7.2f}%")
