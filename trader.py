"""
LeadLagTrader v1.1 — trading engine.

BTC detection: 1m candle close (V1 logic — reliable, confirmed impulse).
Orders: batch-orders via OKX API with attachAlgoOrds (SL+TP on exchange).

PAPER: software SL/TP/TIME monitoring.
LIVE:  batch-orders + attachAlgoOrds. TIME exit → market close order.
       Exchange-closed positions (SL/TP) detected via sync_exchange_positions().

position_size = margin * leverage  (configured in config.json)
"""

import json
import os
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
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
    size_usdt: float   # notional = margin * leverage
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

        # Symbols where leverage was already set on the exchange
        self._leverage_set: set = set()

        # Restore state from DB
        self._restore_state()

    def _restore_state(self):
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

    def sync_live_on_start(self):
        """
        LIVE mode startup: read real USDT balance and sync any open exchange
        positions into self.positions so the bot can manage them (TIME exit etc).
        Called once from main() after exchange is ready.
        """
        if self.config.get('trade_mode') != 'LIVE':
            return

        # 1. Read real balance
        try:
            bal = self.exchange.fetch_balance()
            usdt = bal.get('USDT', {}).get('free', 0) or \
                   bal.get('USDT', {}).get('total', 0)
            if usdt and float(usdt) > 0:
                self.balance = float(usdt)
                self._save_state()
                logger.info(f"[LIVE-START] Real USDT balance: {self.balance:.2f}")
        except Exception as e:
            logger.error(f"[LIVE-START] fetch_balance error: {e}")

        # 2. Sync open positions from exchange
        try:
            cfg = self.config.get('trading', {})
            hold_max = cfg.get('hold_max_sec', 30)
            leverage = cfg.get('leverage', 2)
            position_size = self._position_size()

            ex_positions = self.exchange.fetch_positions()
            synced = 0
            for ep in ex_positions:
                contracts = float(ep.get('contracts') or
                                  ep.get('info', {}).get('pos', 0))
                if contracts <= 0:
                    continue
                sym = ep.get('symbol', '')
                short = sym.split('/')[0].replace(':USDT', '')
                if short in self.positions:
                    continue  # already tracked

                side = 'LONG' if ep.get('side') == 'long' else 'SHORT'
                entry = float(ep.get('entryPrice') or ep.get('info', {}).get('avgPx', 0))
                if entry <= 0:
                    continue

                sl_pct = cfg.get('sl_pct', 0.3) / 100
                tp_pct = cfg.get('tp_pct', 0.3) / 100
                sl = entry * (1 - sl_pct) if side == 'LONG' else entry * (1 + sl_pct)
                tp = entry * (1 + tp_pct) if side == 'LONG' else entry * (1 - tp_pct)

                trade_id = self._next_trade_id()
                pos = Position(
                    id=trade_id, symbol=sym, short_symbol=short,
                    side=side, entry_price=entry, current_price=entry,
                    size_usdt=position_size, leverage=leverage,
                    stop_loss=sl, take_profit=tp, status="OPEN",
                    opened_at=get_gmt2_str(), impulse_id=0,
                    impulse_magnitude=0, hold_max_seconds=hold_max,
                    trade_mode='LIVE'
                )
                with self.lock:
                    self.positions[short] = pos
                synced += 1
                logger.info(f"[LIVE-START] Synced open position: {short} {side} @ {entry}")

            if synced:
                logger.info(f"[LIVE-START] Synced {synced} open positions from exchange")
        except Exception as e:
            logger.error(f"[LIVE-START] sync positions error: {e}")

    def _save_state(self):
        db.set_state('trade_counter', str(self.trade_counter))
        db.set_state('balance', str(round(self.balance, 2)))
        db.set_state('total_pnl', str(round(self.total_pnl, 4)))

    def reload_config(self):
        old_leverage = self.config.get('trading', {}).get('leverage', 2)
        self.config = load_config()
        new_leverage = self.config.get('trading', {}).get('leverage', 2)
        if new_leverage != old_leverage:
            self._leverage_set.clear()
            logger.info(f"[TRADER] Leverage changed {old_leverage}→{new_leverage}, "
                        f"will re-apply to all symbols")

    def _next_trade_id(self) -> str:
        self.trade_counter += 1
        return f"LL-{self.trade_counter:04d}"

    def _position_size(self) -> float:
        """notional = margin * leverage"""
        cfg = self.config.get('trading', {})
        margin = cfg.get('margin', cfg.get('position_size', 10) / cfg.get('leverage', 2))
        return margin * cfg.get('leverage', 2)

    def _calc_time_gap_factor(self, gap_seconds: float) -> float:
        cfg = self.config.get('leader', {}).get('quality', {})
        gap_weak   = cfg.get('gap_weak_sec', 120)
        gap_strong = cfg.get('gap_strong_sec', 900)
        f_min = cfg.get('gap_min_factor', 0.5)
        f_max = cfg.get('gap_max_factor', 2.0)

        if gap_seconds <= gap_weak:
            return f_min
        if gap_seconds >= gap_strong:
            return f_max
        ratio = (gap_seconds - gap_weak) / (gap_strong - gap_weak)
        return f_min + ratio * (f_max - f_min)

    # ─── BTC impulse detection (1m candle close) ──────────────────────────────

    def check_btc_impulse(self) -> Optional[Dict]:
        """
        Check latest BTC 1m candle for impulse.
        quality = magnitude × time_gap_factor.
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

            candle = candles[-2]
            ts = candle[0]

            if ts <= self.last_btc_candle_ts:
                return None

            self.last_btc_candle_ts = ts
            self.btc_price = candle[4]

            o, c = candle[1], candle[4]
            move = (c - o) / o

            if abs(move) >= threshold:
                direction = 'LONG' if move > 0 else 'SHORT'

                trading_cfg = self.config.get('trading', {})
                force = trading_cfg.get('force_direction')
                skip_status = None
                if force in ('LONG', 'SHORT'):
                    direction = force
                else:
                    if direction == 'LONG' and not trading_cfg.get('enable_long', True):
                        skip_status = 'skip_long_disabled'
                    elif direction == 'SHORT' and not trading_cfg.get('enable_short', True):
                        skip_status = 'skip_short_disabled'

                if skip_status:
                    return {
                        'timestamp': get_gmt2_str(), 'leader': 'BTC',
                        'direction': direction, 'magnitude': round(move, 6),
                        'candle_open': o, 'candle_close': c,
                        'n_followers_entered': 0, 'gap_seconds': 0,
                        'gap_factor': 1.0, 'quality': 0, 'status': skip_status,
                    }

                gap_sec = now - self.last_impulse_ts if self.last_impulse_ts > 0 else 9999
                gap_factor = self._calc_time_gap_factor(gap_sec)
                quality = abs(move) * gap_factor

                if quality_enabled and quality < min_quality:
                    logger.info(f"[IMPULSE] BTC {direction} {abs(move)*100:.2f}% "
                                f"SKIP quality={quality*100:.4f}% (gap={gap_sec:.0f}s)")
                    self.last_impulse_ts = now
                    return {
                        'timestamp': get_gmt2_str(), 'leader': 'BTC',
                        'direction': direction, 'magnitude': round(move, 6),
                        'candle_open': o, 'candle_close': c,
                        'n_followers_entered': 0, 'gap_seconds': round(gap_sec, 0),
                        'gap_factor': round(gap_factor, 2),
                        'quality': round(quality, 6), 'status': 'skip_quality',
                    }

                impulse = {
                    'timestamp': get_gmt2_str(), 'leader': 'BTC',
                    'direction': direction, 'magnitude': round(move, 6),
                    'candle_open': o, 'candle_close': c,
                    'n_followers_entered': 0, 'gap_seconds': round(gap_sec, 0),
                    'gap_factor': round(gap_factor, 2),
                    'quality': round(quality, 6), 'status': 'accepted',
                }
                self.last_impulse_ts = now
                logger.info(f"[IMPULSE] BTC {direction} {abs(move)*100:.2f}% "
                            f"quality={quality*100:.4f}% (gap={gap_sec:.0f}s) → ENTER")
                return impulse

        except Exception as e:
            logger.error(f"[TRADER] BTC candle fetch error: {e}")

        return None

    # ─── Open positions ───────────────────────────────────────────────────────

    def open_positions(self, active_symbols: List[Tuple[str, str]],
                       direction: str, impulse_id: int,
                       impulse_magnitude: float) -> int:
        """
        Open positions in all active follower pairs.
        1. Batch-fetch ALL prices in one API call.
        2. PAPER → _open_paper_batch()
           LIVE  → _open_live_batch() (OKX batch-orders + attachAlgoOrds)
        """
        cfg = self.config.get('trading', {})
        tp_pct        = cfg.get('tp_pct', 0.5) / 100
        sl_pct        = cfg.get('sl_pct', 0.3) / 100
        leverage      = cfg.get('leverage', 2)
        position_size = self._position_size()
        max_positions = cfg.get('max_positions', 10)
        hold_max      = cfg.get('hold_max_sec', 30)
        trade_mode    = self.config.get('trade_mode', 'PAPER')

        # Build candidate list
        candidates = []
        for short_sym, full_sym in active_symbols:
            if short_sym in self.positions:
                continue
            if len(self.positions) + len(candidates) >= max_positions:
                break
            candidates.append((short_sym, full_sym))

        if not candidates:
            self._save_state()
            return 0

        # Batch-fetch prices (1 API call)
        full_syms = [fs for _, fs in candidates]
        prices: Dict[str, float] = {}
        try:
            tickers = self.exchange.fetch_tickers(full_syms)
            for sym, t in tickers.items():
                price = t.get('last', 0)
                if price:
                    short = sym.split('/')[0].replace(':USDT', '')
                    prices[short] = price
        except Exception as e:
            logger.error(f"[TRADER] Batch price fetch failed: {e}")
            for short, _ in candidates:
                if short in self.prices:
                    prices[short] = self.prices[short]

        self.prices.update(prices)

        # Build to_open list
        to_open = []
        for short_sym, full_sym in candidates:
            price = prices.get(short_sym, 0)
            if price <= 0:
                continue
            sl = price * (1 - sl_pct) if direction == 'LONG' else price * (1 + sl_pct)
            tp = price * (1 + tp_pct) if direction == 'LONG' else price * (1 - tp_pct)
            to_open.append((short_sym, full_sym, price, sl, tp))

        if not to_open:
            return 0

        # LIVE: validate margin against real balance
        if trade_mode == 'LIVE':
            margin = cfg.get('margin', 5.0)
            available = self.balance
            max_affordable = int(available / margin) if margin > 0 else 0
            if max_affordable <= 0:
                logger.warning(f"[LIVE] Insufficient balance ({available:.2f}) "
                               f"for margin={margin} — skip")
                return 0
            if len(to_open) > max_affordable:
                logger.warning(f"[LIVE] Limiting to {max_affordable} positions "
                               f"(balance={available:.2f}, margin={margin})")
                to_open = to_open[:max_affordable]

        if trade_mode == 'LIVE':
            return self._open_live_batch(to_open, direction, impulse_id,
                                         impulse_magnitude, position_size,
                                         leverage, hold_max)
        else:
            return self._open_paper_batch(to_open, direction, impulse_id,
                                          impulse_magnitude, position_size,
                                          leverage, hold_max)

    def _open_paper_batch(self, to_open, direction, impulse_id, magnitude,
                          position_size, leverage, hold_max) -> int:
        opened = 0
        for short_sym, full_sym, price, sl, tp in to_open:
            trade_id = self._next_trade_id()
            pos = Position(
                id=trade_id, symbol=full_sym, short_symbol=short_sym,
                side=direction, entry_price=price, current_price=price,
                size_usdt=position_size, leverage=leverage,
                stop_loss=sl, take_profit=tp, status="OPEN",
                opened_at=get_gmt2_str(), impulse_id=impulse_id,
                impulse_magnitude=magnitude, hold_max_seconds=hold_max,
                trade_mode='PAPER'
            )
            with self.lock:
                self.positions[short_sym] = pos
            logger.info(f"[OPEN] {trade_id} {short_sym} {direction} @ {price:.6g} "
                        f"SL={sl:.6g} TP={tp:.6g}")
            opened += 1
        self._save_state()
        return opened

    def _open_live_batch(self, to_open, direction, impulse_id, magnitude,
                         position_size, leverage, hold_max) -> int:
        """
        One OKX batch-orders call (up to 20 per batch).
        Each order has attachAlgoOrds → SL+TP live on the exchange.
        """
        side = 'buy' if direction == 'LONG' else 'sell'
        BATCH_SIZE = 20

        # Set leverage for new symbols (sequential to avoid rate limits)
        for short_sym, full_sym, *_ in to_open:
            if short_sym in self._leverage_set:
                continue
            try:
                self.exchange.set_leverage(leverage, full_sym,
                                           params={'mgnMode': 'isolated'})
                self._leverage_set.add(short_sym)
                time.sleep(0.1)  # 100ms between calls — OKX rate limit safe
            except Exception as e:
                logger.warning(f"[LIVE] set_leverage {short_sym}: {e}")

        # Build order specs
        order_specs = []
        for short_sym, full_sym, price, sl, tp in to_open:
            try:
                market = self.exchange.market(full_sym)
                contract_size = float(market.get('contractSize') or 1)
                raw_amount = position_size / price / contract_size
                amount = float(self.exchange.amount_to_precision(full_sym, raw_amount))
                if amount <= 0:
                    logger.warning(f"[LIVE] {short_sym}: amount too small, skip")
                    continue
            except Exception as e:
                logger.error(f"[LIVE] Contract calc {short_sym}: {e}")
                continue

            order_specs.append({
                'short_sym': short_sym,
                'full_sym': full_sym,
                'price': price,
                'sl': sl,
                'tp': tp,
                'order': {
                    'symbol': full_sym,
                    'type': 'market',
                    'side': side,
                    'amount': amount,
                    'params': {
                        'tdMode': 'isolated',
                        'attachAlgoOrds': [{
                            'attachType': 'oco',
                            'tpTriggerPx': str(round(tp, 8)),
                            'tpOrdPx': '-1',
                            'slTriggerPx': str(round(sl, 8)),
                            'slOrdPx': '-1',
                        }],
                    },
                },
            })

        if not order_specs:
            return 0

        batches = [order_specs[i:i + BATCH_SIZE]
                   for i in range(0, len(order_specs), BATCH_SIZE)]

        t_start = time.time()
        all_results = []

        def send_batch(batch):
            return self.exchange.create_orders([spec['order'] for spec in batch])

        if len(batches) == 1:
            try:
                all_results.append((batches[0], send_batch(batches[0])))
            except Exception as e:
                logger.error(f"[LIVE] batch-orders error: {e}")
                return 0
        else:
            with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                futs = {pool.submit(send_batch, b): b for b in batches}
                for fut in as_completed(futs):
                    try:
                        all_results.append((futs[fut], fut.result()))
                    except Exception as e:
                        logger.error(f"[LIVE] batch error: {e}")

        logger.info(f"[LIVE] {len(order_specs)} orders in {(time.time()-t_start)*1000:.0f}ms")

        spec_by_sym = {spec['full_sym']: spec for spec in order_specs}
        opened = 0
        for batch, raw_orders in all_results:
            if not raw_orders:
                continue
            for raw in raw_orders:
                full_sym = raw.get('symbol', '')
                spec = spec_by_sym.get(full_sym)
                if not spec:
                    continue
                if raw.get('status') in ('canceled', 'rejected'):
                    logger.warning(f"[LIVE] {spec['short_sym']} rejected")
                    continue

                fill_price = float(raw.get('average') or raw.get('price') or spec['price'])
                trade_id = self._next_trade_id()
                pos = Position(
                    id=trade_id, symbol=full_sym,
                    short_symbol=spec['short_sym'],
                    side=direction, entry_price=fill_price,
                    current_price=fill_price,
                    size_usdt=position_size, leverage=leverage,
                    stop_loss=spec['sl'], take_profit=spec['tp'],
                    status="OPEN", opened_at=get_gmt2_str(),
                    impulse_id=impulse_id, impulse_magnitude=magnitude,
                    hold_max_seconds=hold_max, trade_mode='LIVE'
                )
                with self.lock:
                    self.positions[spec['short_sym']] = pos
                logger.info(f"[LIVE-OPEN] {trade_id} {spec['short_sym']} {direction} "
                            f"@ {fill_price:.6g}")
                opened += 1

        self._save_state()
        return opened

    # ─── Price updates & exits ────────────────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]):
        """
        PAPER: check SL, TP, TIME.
        LIVE:  check TIME only (SL/TP handled by exchange via attachAlgoOrds).
        """
        self.prices.update(prices)
        cfg = self.config.get('trading', {})
        fee = cfg.get('fee_pct', 0.05) / 100
        trade_mode = self.config.get('trade_mode', 'PAPER')

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

                if trade_mode == 'LIVE':
                    try:
                        opened_dt = datetime.strptime(pos.opened_at, '%Y-%m-%d %H:%M:%S')
                        if (now - opened_dt).total_seconds() >= pos.hold_max_seconds:
                            to_close.append((sym, 'TIME'))
                    except (ValueError, TypeError):
                        pass
                else:
                    if pos.side == 'LONG' and price <= pos.stop_loss:
                        to_close.append((sym, 'SL'))
                    elif pos.side == 'SHORT' and price >= pos.stop_loss:
                        to_close.append((sym, 'SL'))
                    elif pos.side == 'LONG' and price >= pos.take_profit:
                        to_close.append((sym, 'TP'))
                    elif pos.side == 'SHORT' and price <= pos.take_profit:
                        to_close.append((sym, 'TP'))
                    else:
                        try:
                            opened_dt = datetime.strptime(pos.opened_at, '%Y-%m-%d %H:%M:%S')
                            if (now - opened_dt).total_seconds() >= pos.hold_max_seconds:
                                to_close.append((sym, 'TIME'))
                        except (ValueError, TypeError):
                            pass

        for sym, reason in to_close:
            self.close_position(sym, reason)

    def sync_exchange_positions(self):
        """
        LIVE: fetch open positions from exchange, detect ones closed by SL/TP.
        """
        if self.config.get('trade_mode') != 'LIVE':
            return
        if not self.positions:
            return
        try:
            ex_positions = self.exchange.fetch_positions()
            ex_open = set()
            for ep in ex_positions:
                contracts = float(ep.get('contracts') or ep.get('info', {}).get('pos', 0))
                if contracts > 0:
                    short = ep.get('symbol', '').split('/')[0].replace(':USDT', '')
                    ex_open.add(short)

            to_close = []
            with self.lock:
                for sym in list(self.positions.keys()):
                    if sym not in ex_open:
                        to_close.append(sym)

            for sym in to_close:
                logger.info(f"[SYNC] {sym} closed by exchange")
                self.close_position(sym, 'EXCHANGE')

        except Exception as e:
            logger.error(f"[SYNC] fetch_positions error: {e}")

    def close_position(self, symbol: str, reason: str) -> Optional[Dict]:
        """
        Close position. LIVE + TIME → send market close order to exchange.
        """
        with self.lock:
            pos = self.positions.pop(symbol, None)

        if not pos:
            return None

        cfg = self.config.get('trading', {})
        fee = cfg.get('fee_pct', 0.05) / 100

        pos.status = "CLOSED"
        pos.closed_at = get_gmt2_str()
        pos.close_reason = reason

        pnl_usdt, pnl_pct = pos.calculate_pnl(fee)
        pos.pnl_usdt = pnl_usdt
        pos.pnl_percent = pnl_pct

        # LIVE: refresh real balance from exchange after close
        if pos.trade_mode == 'LIVE' and self.exchange:
            try:
                bal = self.exchange.fetch_balance()
                usdt = bal.get('USDT', {}).get('free', 0) or \
                       bal.get('USDT', {}).get('total', 0)
                if usdt and float(usdt) > 0:
                    self.balance = float(usdt)
            except Exception:
                pass  # fallback: software balance already updated below

        # LIVE + TIME → send market close (cancel attached algo orders)
        if pos.trade_mode == 'LIVE' and reason == 'TIME' and self.exchange:
            try:
                close_side = 'sell' if pos.side == 'LONG' else 'buy'
                market = self.exchange.market(pos.symbol)
                contract_size = float(market.get('contractSize') or 1)
                amount = pos.size_usdt / pos.entry_price / contract_size
                amount = float(self.exchange.amount_to_precision(pos.symbol, amount))
                self.exchange.create_order(
                    pos.symbol, 'market', close_side, amount,
                    params={'tdMode': 'isolated', 'reduceOnly': True}
                )
                logger.info(f"[LIVE-CLOSE] {pos.id} {pos.short_symbol} TIME close sent")
            except Exception as e:
                logger.error(f"[LIVE-CLOSE] {pos.short_symbol}: {e}")

        self.balance += pnl_usdt
        self.total_pnl += pnl_pct
        self.total_trades += 1
        if pnl_pct > 0:
            self.wins += 1
        else:
            self.losses += 1

        try:
            opened_dt = datetime.strptime(pos.opened_at, '%Y-%m-%d %H:%M:%S')
            closed_dt = datetime.strptime(pos.closed_at, '%Y-%m-%d %H:%M:%S')
            hold_sec = int((closed_dt - opened_dt).total_seconds())
        except (ValueError, TypeError):
            hold_sec = 0

        trade_data = {
            'trade_id': pos.id, 'symbol': pos.short_symbol, 'leader': 'BTC',
            'side': pos.side, 'entry_price': pos.entry_price,
            'exit_price': pos.current_price,
            'pnl_pct': round(pnl_pct, 4), 'pnl_usdt': round(pnl_usdt, 4),
            'close_reason': reason, 'hold_seconds': hold_sec,
            'impulse_magnitude': pos.impulse_magnitude,
            'impulse_id': pos.impulse_id,
            'opened_at': pos.opened_at, 'closed_at': pos.closed_at,
            'trade_mode': pos.trade_mode
        }
        db.save_trade(trade_data)
        self._save_state()

        icon = '+' if pnl_pct > 0 else ''
        logger.info(f"[CLOSE] {pos.id} {pos.short_symbol} {pos.side} "
                    f"{icon}{pnl_pct:.2f}% ({reason}) hold={hold_sec}s")

        return trade_data

    def close_all(self, reason: str = 'manual') -> int:
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
        cfg = self.config.get('trading', {})
        initial  = cfg.get('initial_balance', 1000.0)
        margin   = cfg.get('margin', 5.0)
        leverage = cfg.get('leverage', 2)
        pnl_usd  = self.balance - initial
        pnl_pct  = (pnl_usd / initial * 100) if initial > 0 else 0
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
            'btc_price': self.btc_price,
            'margin': margin,
            'leverage': leverage,
            'position_size': margin * leverage,
        }

    def reset_stats(self):
        cfg = self.config.get('trading', {})
        self.balance = cfg.get('initial_balance', 1000.0)
        self.total_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self._save_state()
        # New session — UI tabs show only trades/impulses from this point
        db.set_state('session_start', get_gmt2_str())
        db.log('reset', 'Stats reset', {})
        logger.info("[TRADER] Stats reset")
