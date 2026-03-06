"""
LeadLagTrader — trading engine.

Monitors BTC 1m candles for impulses.
On impulse: opens positions in all active follower pairs in BTC direction.
Manages TP/SL/time-based exits.
"""

import json
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import ccxt

from database import db, get_gmt2_str, get_gmt2_time

logger = logging.getLogger(__name__)

TZ_OFFSET = timedelta(hours=2)


def load_config() -> Dict:
    with open('config.json', 'r') as f:
        return json.load(f)


@dataclass
class Position:
    id: str
    symbol: str
    short_symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    current_price: float
    size_usdt: float
    leverage: int
    stop_loss: float
    take_profit: float
    status: str = "OPEN"
    pnl_usdt: float = 0.0
    pnl_percent: float = 0.0
    opened_at: str = ""
    closed_at: str = ""
    close_reason: str = ""
    impulse_id: int = 0
    impulse_magnitude: float = 0.0
    hold_max_seconds: int = 300
    trade_mode: str = "PAPER"
    max_pnl_percent: float = 0.0

    def calculate_pnl(self, fee_rate: float = 0.0005) -> Tuple[float, float]:
        if self.side == "LONG":
            pct = (self.current_price - self.entry_price) / self.entry_price
        else:
            pct = (self.entry_price - self.current_price) / self.entry_price
        pct -= 2 * fee_rate
        pnl_pct = pct * self.leverage * 100
        pnl_usdt = self.size_usdt * (pnl_pct / 100)
        return pnl_usdt, pnl_pct

    def to_dict(self) -> Dict:
        return asdict(self)


class LeadLagTrader:
    def __init__(self, exchange: ccxt.Exchange = None):
        self.config = load_config()
        self.exchange = exchange
        self.positions: Dict[str, Position] = {}
        self.trade_counter = 0
        self.balance = self.config.get('trading', {}).get('initial_balance', 1000.0)
        self.total_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.lock = threading.Lock()
        self.running = False
        self.paused = False

        # BTC monitoring state
        self.last_btc_candle_ts = 0
        self.last_impulse_ts = 0
        self.btc_price = 0.0

        # Prices cache
        self.prices: Dict[str, float] = {}

        # Restore state from DB
        self._restore_state()

    def _restore_state(self):
        """Restore trade counter and balance from DB."""
        counter = db.get_state('trade_counter')
        if counter:
            self.trade_counter = int(counter)
        balance = db.get_state('balance')
        if balance:
            self.balance = float(balance)
        total_pnl = db.get_state('total_pnl')
        if total_pnl:
            self.total_pnl = float(total_pnl)

        stats = db.get_trade_stats()
        self.total_trades = stats.get('total', 0)
        self.wins = stats.get('wins', 0)
        self.losses = stats.get('losses', 0)

        logger.info(f"[TRADER] Restored: balance={self.balance:.2f}, "
                    f"trades={self.total_trades}, counter={self.trade_counter}")

    def _save_state(self):
        db.set_state('trade_counter', str(self.trade_counter))
        db.set_state('balance', str(round(self.balance, 2)))
        db.set_state('total_pnl', str(round(self.total_pnl, 4)))

    def reload_config(self):
        self.config = load_config()

    def _next_trade_id(self) -> str:
        self.trade_counter += 1
        return f"LL-{self.trade_counter:04d}"

    def _calc_time_gap_factor(self, gap_seconds: float) -> float:
        """
        Calculate time gap quality factor.
        Gap < gap_weak_sec → factor = gap_min_factor (weak signal, noise)
        Gap > gap_strong_sec → factor = gap_max_factor (fresh signal)
        Between → linear interpolation
        """
        cfg = self.config.get('leader', {}).get('quality', {})
        gap_weak = cfg.get('gap_weak_sec', 120)      # < 2 min = weak
        gap_strong = cfg.get('gap_strong_sec', 900)   # > 15 min = strong
        f_min = cfg.get('gap_min_factor', 0.5)
        f_max = cfg.get('gap_max_factor', 2.0)

        if gap_seconds <= gap_weak:
            return f_min
        if gap_seconds >= gap_strong:
            return f_max
        # Linear interpolation
        ratio = (gap_seconds - gap_weak) / (gap_strong - gap_weak)
        return f_min + ratio * (f_max - f_min)

    def check_btc_impulse(self) -> Optional[Dict]:
        """
        Check latest BTC 1m candle for impulse.
        Uses quality scoring: quality = magnitude × time_gap_factor.
        Returns impulse dict with quality info, or None.
        """
        cfg = self.config.get('leader', {})
        threshold = cfg.get('impulse_threshold', 0.003)
        quality_cfg = cfg.get('quality', {})
        quality_enabled = quality_cfg.get('enabled', True)
        min_quality = quality_cfg.get('min_quality', 0.0003)

        now = time.time()

        try:
            candles = self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1m', limit=2)
            if not candles or len(candles) < 2:
                return None

            # Use the last completed candle (index -2)
            candle = candles[-2]
            ts = candle[0]

            # Already processed this candle
            if ts <= self.last_btc_candle_ts:
                return None

            self.last_btc_candle_ts = ts
            self.btc_price = candle[4]  # close

            o, c = candle[1], candle[4]
            move = (c - o) / o

            if abs(move) >= threshold:
                direction = 'LONG' if move > 0 else 'SHORT'

                # Direction filter
                trading_cfg = self.config.get('trading', {})
                force = trading_cfg.get('force_direction')
                skip_status = None
                if force in ('LONG', 'SHORT'):
                    direction = force
                else:
                    if direction == 'LONG' and not trading_cfg.get('enable_long', True):
                        logger.info(f"[IMPULSE] BTC {direction} {abs(move)*100:.2f}% SKIP: LONG disabled")
                        skip_status = 'skip_long_disabled'
                    elif direction == 'SHORT' and not trading_cfg.get('enable_short', True):
                        logger.info(f"[IMPULSE] BTC {direction} {abs(move)*100:.2f}% SKIP: SHORT disabled")
                        skip_status = 'skip_short_disabled'

                if skip_status:
                    return {
                        'timestamp': get_gmt2_str(),
                        'leader': 'BTC',
                        'direction': direction,
                        'magnitude': round(move, 6),
                        'candle_open': o,
                        'candle_close': c,
                        'n_followers_entered': 0,
                        'gap_seconds': 0,
                        'gap_factor': 1.0,
                        'quality': 0,
                        'status': skip_status,
                    }

                # Quality scoring
                gap_sec = now - self.last_impulse_ts if self.last_impulse_ts > 0 else 9999
                gap_factor = self._calc_time_gap_factor(gap_sec)
                quality = abs(move) * gap_factor

                # Quality gate
                if quality_enabled and quality < min_quality:
                    logger.info(f"[IMPULSE] BTC {direction} {abs(move)*100:.2f}% "
                                f"SKIP: quality={quality*100:.4f}% < min {min_quality*100:.4f}% "
                                f"(gap={gap_sec:.0f}s, factor={gap_factor:.2f})")
                    # Still update last_impulse_ts so gap accumulates from last DETECTED impulse
                    self.last_impulse_ts = now
                    return {
                        'timestamp': get_gmt2_str(),
                        'leader': 'BTC',
                        'direction': direction,
                        'magnitude': round(move, 6),
                        'candle_open': o,
                        'candle_close': c,
                        'n_followers_entered': 0,
                        'gap_seconds': round(gap_sec, 0),
                        'gap_factor': round(gap_factor, 2),
                        'quality': round(quality, 6),
                        'status': 'skip_quality',
                    }

                impulse = {
                    'timestamp': get_gmt2_str(),
                    'leader': 'BTC',
                    'direction': direction,
                    'magnitude': round(move, 6),
                    'candle_open': o,
                    'candle_close': c,
                    'n_followers_entered': 0,
                    'gap_seconds': round(gap_sec, 0),
                    'gap_factor': round(gap_factor, 2),
                    'quality': round(quality, 6),
                    'status': 'accepted',
                }
                self.last_impulse_ts = now
                logger.info(f"[IMPULSE] BTC {direction} {abs(move)*100:.2f}% "
                            f"quality={quality*100:.4f}% (gap={gap_sec:.0f}s, "
                            f"factor={gap_factor:.2f}) → ENTER")
                return impulse

        except Exception as e:
            logger.error(f"[TRADER] BTC candle fetch error: {e}")

        return None

    def open_positions(self, active_symbols: List[Tuple[str, str]],
                       direction: str, impulse_id: int,
                       impulse_magnitude: float) -> int:
        """
        Open positions in all active follower pairs.
        active_symbols: list of (short_symbol, full_symbol)
        Returns count of opened positions.
        """
        cfg = self.config.get('trading', {})
        tp_pct = cfg.get('tp_pct', 0.5) / 100
        sl_pct = cfg.get('sl_pct', 0.3) / 100
        position_size = cfg.get('position_size', 10)
        leverage = cfg.get('leverage', 2)
        max_positions = cfg.get('max_positions', 10)
        hold_max = cfg.get('hold_max_sec', 30)

        opened = 0

        for short_sym, full_sym in active_symbols:
            # Skip if already have position in this symbol
            if short_sym in self.positions:
                continue

            if len(self.positions) >= max_positions:
                break

            # Get current price
            price = self.prices.get(short_sym)
            if not price:
                try:
                    ticker = self.exchange.fetch_ticker(full_sym)
                    price = ticker.get('last', 0)
                    self.prices[short_sym] = price
                except Exception as e:
                    logger.error(f"[TRADER] Price fetch {short_sym}: {e}")
                    continue

            if price <= 0:
                continue

            # Calculate SL/TP
            if direction == 'LONG':
                sl = price * (1 - sl_pct)
                tp = price * (1 + tp_pct)
            else:
                sl = price * (1 + sl_pct)
                tp = price * (1 - tp_pct)

            trade_id = self._next_trade_id()
            pos = Position(
                id=trade_id,
                symbol=full_sym,
                short_symbol=short_sym,
                side=direction,
                entry_price=price,
                current_price=price,
                size_usdt=position_size,
                leverage=leverage,
                stop_loss=sl,
                take_profit=tp,
                status="OPEN",
                opened_at=get_gmt2_str(),
                impulse_id=impulse_id,
                impulse_magnitude=impulse_magnitude,
                hold_max_seconds=hold_max,
                trade_mode=self.config.get('trade_mode', 'PAPER')
            )

            with self.lock:
                self.positions[short_sym] = pos

            logger.info(f"[OPEN] {trade_id} {short_sym} {direction} @ {price:.4f} "
                        f"SL={sl:.4f} TP={tp:.4f}")
            opened += 1

        self._save_state()
        return opened

    def update_prices(self, prices: Dict[str, float]):
        """Update prices and check exits for all open positions."""
        self.prices.update(prices)
        cfg = self.config.get('trading', {})
        fee = cfg.get('fee_pct', 0.05) / 100

        to_close = []
        now = get_gmt2_time()

        with self.lock:
            for sym, pos in self.positions.items():
                if pos.status != "OPEN":
                    continue

                price = prices.get(sym)
                if not price:
                    continue

                pos.current_price = price
                pnl_usdt, pnl_pct = pos.calculate_pnl(fee)
                pos.pnl_usdt = pnl_usdt
                pos.pnl_percent = pnl_pct
                if pnl_pct > pos.max_pnl_percent:
                    pos.max_pnl_percent = pnl_pct

                # Check SL
                if pos.side == 'LONG' and price <= pos.stop_loss:
                    to_close.append((sym, 'SL'))
                elif pos.side == 'SHORT' and price >= pos.stop_loss:
                    to_close.append((sym, 'SL'))
                # Check TP
                elif pos.side == 'LONG' and price >= pos.take_profit:
                    to_close.append((sym, 'TP'))
                elif pos.side == 'SHORT' and price <= pos.take_profit:
                    to_close.append((sym, 'TP'))
                else:
                    # Check time exit
                    try:
                        opened = datetime.strptime(pos.opened_at, '%Y-%m-%d %H:%M:%S')
                        elapsed = (now - opened).total_seconds()
                        if elapsed >= pos.hold_max_seconds:
                            to_close.append((sym, 'TIME'))
                    except (ValueError, TypeError):
                        pass

        for sym, reason in to_close:
            self.close_position(sym, reason)

    def close_position(self, symbol: str, reason: str) -> Optional[Dict]:
        """Close a position and record the trade."""
        with self.lock:
            pos = self.positions.pop(symbol, None)

        if not pos:
            return None

        cfg = self.config.get('trading', {})
        fee = cfg.get('fee_pct', 0.05) / 100

        pos.status = "CLOSED"
        pos.closed_at = get_gmt2_str()
        pos.close_reason = reason

        # Final PnL
        pnl_usdt, pnl_pct = pos.calculate_pnl(fee)
        pos.pnl_usdt = pnl_usdt
        pos.pnl_percent = pnl_pct

        # Update balance
        self.balance += pnl_usdt
        self.total_pnl += pnl_pct
        self.total_trades += 1
        if pnl_pct > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Hold time
        try:
            opened = datetime.strptime(pos.opened_at, '%Y-%m-%d %H:%M:%S')
            closed = datetime.strptime(pos.closed_at, '%Y-%m-%d %H:%M:%S')
            hold_sec = int((closed - opened).total_seconds())
        except (ValueError, TypeError):
            hold_sec = 0

        # Save to DB
        trade_data = {
            'trade_id': pos.id,
            'symbol': pos.short_symbol,
            'leader': 'BTC',
            'side': pos.side,
            'entry_price': pos.entry_price,
            'exit_price': pos.current_price,
            'pnl_pct': round(pnl_pct, 4),
            'pnl_usdt': round(pnl_usdt, 4),
            'close_reason': reason,
            'hold_seconds': hold_sec,
            'impulse_magnitude': pos.impulse_magnitude,
            'impulse_id': pos.impulse_id,
            'opened_at': pos.opened_at,
            'closed_at': pos.closed_at,
            'trade_mode': pos.trade_mode
        }
        db.save_trade(trade_data)
        self._save_state()

        icon = '+' if pnl_pct > 0 else ''
        logger.info(f"[CLOSE] {pos.id} {pos.short_symbol} {pos.side} "
                    f"{icon}{pnl_pct:.2f}% ({reason}) hold={hold_sec}s")

        return trade_data

    def close_all(self, reason: str = 'manual') -> int:
        """Close all open positions."""
        symbols = list(self.positions.keys())
        count = 0
        for sym in symbols:
            if self.close_position(sym, reason):
                count += 1
        return count

    def get_open_positions(self) -> List[Dict]:
        with self.lock:
            return [pos.to_dict() for pos in self.positions.values()
                    if pos.status == "OPEN"]

    def get_status(self) -> Dict:
        wr = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        initial = self.config.get('trading', {}).get('initial_balance', 1000.0)
        pnl_usd = self.balance - initial
        pnl_pct = (pnl_usd / initial * 100) if initial > 0 else 0
        return {
            'balance': round(self.balance, 2),
            'initial_balance': initial,
            'pnl_usd': round(pnl_usd, 2),
            'pnl_pct': round(pnl_pct, 2),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(wr, 1),
            'open_positions': len(self.positions),
            'trade_mode': self.config.get('trade_mode', 'PAPER'),
            'paused': self.paused,
            'btc_price': self.btc_price
        }

    def reset_stats(self):
        """Reset balance and trade stats."""
        cfg = self.config.get('trading', {})
        self.balance = cfg.get('initial_balance', 1000.0)
        self.total_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self._save_state()
        db.log('reset', 'Stats reset', {})
        logger.info("[TRADER] Stats reset")
