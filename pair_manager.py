"""
PairManager — manages active pair lifecycle.

- Evaluates scanner results, adds pairs meeting criteria
- Monitors live performance, removes degraded pairs
- Logs all additions/removals with reason
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from database import db, get_gmt2_str

logger = logging.getLogger(__name__)


def load_config() -> Dict:
    with open('config.json', 'r') as f:
        return json.load(f)


class PairStats:
    def __init__(self, symbol: str, full_symbol: str, wr: float, pf: float,
                 sharpe: float, n_trades: int, avg_return: float):
        self.symbol = symbol
        self.full_symbol = full_symbol
        self.wr = wr
        self.pf = pf
        self.sharpe = sharpe
        self.n_trades = n_trades
        self.avg_return = avg_return
        self.live_trades = 0
        self.live_wins = 0
        self.live_pnl = 0.0
        self.activated_at = get_gmt2_str()

    @property
    def live_wr(self) -> float:
        return (self.live_wins / self.live_trades * 100) if self.live_trades > 0 else 0

    @property
    def live_pf(self) -> float:
        if self.live_trades == 0:
            return 0
        trades = db.get_trades(limit=self.live_trades, symbol=self.symbol)
        wins_sum = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
        loss_sum = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0))
        return round(wins_sum / loss_sum, 2) if loss_sum > 0 else 999

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'full_symbol': self.full_symbol,
            'scanner_wr': self.wr,
            'scanner_pf': self.pf,
            'sharpe': self.sharpe,
            'scanner_trades': self.n_trades,
            'avg_return': self.avg_return,
            'live_trades': self.live_trades,
            'live_wins': self.live_wins,
            'live_wr': round(self.live_wr, 1),
            'live_pnl': round(self.live_pnl, 3),
            'activated_at': self.activated_at
        }


class PairManager:
    def __init__(self):
        self.active_pairs: Dict[str, PairStats] = {}
        self._load_from_db()

    def _load_from_db(self):
        """Restore active pairs from DB on startup."""
        db_pairs = db.get_active_pairs()
        for p in db_pairs:
            sym = p['symbol']
            stats = db.get_trade_stats(symbol=sym)
            self.active_pairs[sym] = PairStats(
                symbol=sym,
                full_symbol=f"{sym}/USDT:USDT",
                wr=p.get('win_rate', 0),
                pf=p.get('profit_factor', 0),
                sharpe=0,
                n_trades=p.get('total_trades', 0),
                avg_return=0
            )
            self.active_pairs[sym].live_trades = stats.get('total', 0)
            self.active_pairs[sym].live_wins = stats.get('wins', 0)
            self.active_pairs[sym].live_pnl = stats.get('total_pnl', 0)
            self.active_pairs[sym].activated_at = p.get('activated_at', '')

        if self.active_pairs:
            logger.info(f"[PAIRS] Restored {len(self.active_pairs)} active pairs from DB")

    def evaluate_new(self, scanner_results: List[Dict]) -> List[str]:
        """Add new pairs that meet criteria. Returns list of newly added symbols."""
        cfg = load_config()
        follower_cfg = cfg.get('follower', {})
        min_wr = follower_cfg.get('min_wr', 55)
        min_pf = follower_cfg.get('min_pf', 1.3)
        min_trades = follower_cfg.get('min_trades', 10)
        max_active = follower_cfg.get('max_active', 10)

        added = []
        for r in scanner_results:
            sym = r.get('short_symbol', r.get('symbol', ''))
            full_sym = r.get('symbol', f"{sym}/USDT:USDT")

            if sym in self.active_pairs:
                # Update scanner stats for existing pair
                p = self.active_pairs[sym]
                p.wr = r['wr']
                p.pf = r['pf']
                p.sharpe = r.get('sharpe', 0)
                p.n_trades = r['n_trades']
                continue

            if len(self.active_pairs) >= max_active:
                break

            if r['wr'] >= min_wr and r['pf'] >= min_pf and r['n_trades'] >= min_trades:
                self.active_pairs[sym] = PairStats(
                    symbol=sym,
                    full_symbol=full_sym,
                    wr=r['wr'],
                    pf=r['pf'],
                    sharpe=r.get('sharpe', 0),
                    n_trades=r['n_trades'],
                    avg_return=r.get('avg_return', 0)
                )
                reason = (f"Scanner: WR={r['wr']}%, PF={r['pf']}, "
                          f"trades={r['n_trades']}, sharpe={r.get('sharpe', 0)}")
                db.save_pair(sym, 'BTC', reason)
                db.log('pair_add', f'Added {sym}', {'reason': reason, **r})
                logger.info(f"[PAIRS] +++ Added {sym}: {reason}")
                added.append(sym)

        return added

    def evaluate_existing(self) -> List[str]:
        """Check active pairs for degradation. Returns list of removed symbols."""
        cfg = load_config()
        follower_cfg = cfg.get('follower', {})
        rolling = follower_cfg.get('rolling_window', 50)
        degrade_wr = follower_cfg.get('degrade_wr', 45)
        degrade_pf = follower_cfg.get('degrade_pf', 0.9)
        min_live = 5  # need at least 5 live trades to evaluate

        removed = []
        for sym in list(self.active_pairs.keys()):
            p = self.active_pairs[sym]
            if p.live_trades < min_live:
                continue

            stats = db.get_trade_stats(symbol=sym, last_n=rolling)
            wr = stats.get('wr', 0)
            pf = stats.get('pf', 0)

            # Update live stats
            p.live_trades = stats.get('total', 0)
            p.live_wins = stats.get('wins', 0)
            p.live_pnl = stats.get('total_pnl', 0)
            db.update_pair_stats_live(sym, p.live_trades, wr, pf)

            if wr < degrade_wr or pf < degrade_pf:
                reason = f"Degraded: WR={wr}% (<{degrade_wr}), PF={pf} (<{degrade_pf})"
                db.deactivate_pair(sym, reason)
                db.log('pair_remove', f'Removed {sym}', {'reason': reason, 'wr': wr, 'pf': pf})
                logger.warning(f"[PAIRS] --- Removed {sym}: {reason}")
                del self.active_pairs[sym]
                removed.append(sym)

        return removed

    def record_trade(self, symbol: str, pnl_pct: float):
        """Update live stats after a trade closes."""
        if symbol in self.active_pairs:
            p = self.active_pairs[symbol]
            p.live_trades += 1
            if pnl_pct > 0:
                p.live_wins += 1
            p.live_pnl += pnl_pct

    def get_active_pairs(self) -> List[str]:
        """Return list of active full symbols (for exchange queries)."""
        return [p.full_symbol for p in self.active_pairs.values()]

    def get_active_short_symbols(self) -> List[str]:
        return list(self.active_pairs.keys())

    def get_pairs_info(self) -> List[Dict]:
        return [p.to_dict() for p in self.active_pairs.values()]

    def get_pair(self, symbol: str) -> Optional[PairStats]:
        return self.active_pairs.get(symbol)

    def remove_pair(self, symbol: str, reason: str = 'manual'):
        """Manually remove a pair."""
        if symbol in self.active_pairs:
            db.deactivate_pair(symbol, reason)
            db.log('pair_remove', f'Removed {symbol}', {'reason': reason})
            logger.info(f"[PAIRS] Removed {symbol}: {reason}")
            del self.active_pairs[symbol]

    def add_pair_manual(self, symbol: str, full_symbol: str = None):
        """Manually add a pair."""
        if symbol in self.active_pairs:
            return
        if full_symbol is None:
            full_symbol = f"{symbol}/USDT:USDT"
        self.active_pairs[symbol] = PairStats(
            symbol=symbol, full_symbol=full_symbol,
            wr=0, pf=0, sharpe=0, n_trades=0, avg_return=0
        )
        db.save_pair(symbol, 'BTC', 'manual')
        db.log('pair_add', f'Added {symbol} (manual)', {})
        logger.info(f"[PAIRS] Added {symbol} (manual)")
