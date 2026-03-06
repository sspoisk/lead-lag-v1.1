import sqlite3
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

TZ_OFFSET = timedelta(hours=2)

def get_gmt2_time():
    return datetime.utcnow() + TZ_OFFSET

def get_gmt2_str():
    return get_gmt2_time().strftime('%Y-%m-%d %H:%M:%S')


class Database:
    def __init__(self, db_path: str = "data/lead_lag.db"):
        self.db_path = db_path
        self.local = threading.local()
        self._ensure_dir()
        self._init_db()
        logger.info(f"[DB] Initialized: {db_path}")

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
            self.local.conn.execute("PRAGMA journal_mode=WAL")
        return self.local.conn

    @contextmanager
    def get_cursor(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[DB] Error: {e}")
            raise
        finally:
            cursor.close()

    def _init_db(self):
        with self.get_cursor() as cur:
            cur.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                leader TEXT DEFAULT 'BTC',
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                pnl_pct REAL DEFAULT 0,
                pnl_usdt REAL DEFAULT 0,
                close_reason TEXT,
                hold_seconds INTEGER DEFAULT 0,
                impulse_magnitude REAL DEFAULT 0,
                impulse_id INTEGER,
                opened_at TEXT,
                closed_at TEXT,
                trade_mode TEXT DEFAULT 'PAPER'
            )''')

            cur.execute('''
            CREATE TABLE IF NOT EXISTS pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                leader TEXT DEFAULT 'BTC',
                activated_at TEXT,
                deactivated_at TEXT,
                reason TEXT,
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                status TEXT DEFAULT 'active'
            )''')

            cur.execute('''
            CREATE TABLE IF NOT EXISTS pair_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT,
                wr REAL DEFAULT 0,
                pf REAL DEFAULT 0,
                sharpe REAL DEFAULT 0,
                lag_seconds INTEGER DEFAULT 0,
                n_trades INTEGER DEFAULT 0,
                avg_return REAL DEFAULT 0,
                source TEXT DEFAULT 'scanner'
            )''')

            cur.execute('''
            CREATE TABLE IF NOT EXISTS impulses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                leader TEXT DEFAULT 'BTC',
                direction TEXT,
                magnitude REAL,
                n_followers_entered INTEGER DEFAULT 0,
                candle_open REAL,
                candle_close REAL
            )''')

            cur.execute('''
            CREATE TABLE IF NOT EXISTS bot_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                type TEXT,
                message TEXT,
                data TEXT
            )''')

            cur.execute('''
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )''')

            # Indexes
            cur.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_pairs_status ON pairs(status)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_impulses_ts ON impulses(timestamp)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_pair_stats_symbol ON pair_stats(symbol)')

            # Migration: add quality columns to impulses
            try:
                cur.execute("ALTER TABLE impulses ADD COLUMN gap_seconds REAL DEFAULT 0")
                cur.execute("ALTER TABLE impulses ADD COLUMN gap_factor REAL DEFAULT 1.0")
                cur.execute("ALTER TABLE impulses ADD COLUMN quality REAL DEFAULT 0")
            except Exception:
                pass  # columns already exist
            try:
                cur.execute("ALTER TABLE impulses ADD COLUMN status TEXT DEFAULT 'accepted'")
            except Exception:
                pass  # column already exists

    # === TRADES ===

    def save_trade(self, trade: Dict) -> int:
        with self.get_cursor() as cur:
            cur.execute('''
            INSERT INTO trades (trade_id, symbol, leader, side, entry_price, exit_price,
                pnl_pct, pnl_usdt, close_reason, hold_seconds, impulse_magnitude,
                impulse_id, opened_at, closed_at, trade_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('trade_id'), trade.get('symbol'), trade.get('leader', 'BTC'),
                trade.get('side'), trade.get('entry_price'), trade.get('exit_price'),
                trade.get('pnl_pct', 0), trade.get('pnl_usdt', 0),
                trade.get('close_reason'), trade.get('hold_seconds', 0),
                trade.get('impulse_magnitude', 0), trade.get('impulse_id'),
                trade.get('opened_at'), trade.get('closed_at'),
                trade.get('trade_mode', 'PAPER')
            ))
            return cur.lastrowid

    def get_trades(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        with self.get_cursor() as cur:
            if symbol:
                cur.execute(
                    'SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?',
                    (symbol, limit))
            else:
                cur.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_trade_stats(self, symbol: str = None, last_n: int = None) -> Dict:
        with self.get_cursor() as cur:
            where = []
            params = []
            if symbol:
                where.append('symbol = ?')
                params.append(symbol)
            where_str = f"WHERE {' AND '.join(where)}" if where else ""

            if last_n:
                cur.execute(f'''
                    SELECT * FROM trades {where_str}
                    ORDER BY id DESC LIMIT ?
                ''', params + [last_n])
            else:
                cur.execute(f'SELECT * FROM trades {where_str}', params)

            rows = [dict(r) for r in cur.fetchall()]
            if not rows:
                return {'total': 0, 'wins': 0, 'losses': 0, 'wr': 0, 'pf': 0,
                        'total_pnl': 0, 'avg_pnl': 0}

            wins = [r for r in rows if r['pnl_pct'] > 0]
            losses = [r for r in rows if r['pnl_pct'] <= 0]
            total_profit = sum(r['pnl_pct'] for r in wins)
            total_loss = abs(sum(r['pnl_pct'] for r in losses))
            pf = total_profit / total_loss if total_loss > 0 else 999

            tp_count = sum(1 for r in rows if r.get('close_reason') == 'TP')
            sl_count = sum(1 for r in rows if r.get('close_reason') == 'SL')
            time_count = sum(1 for r in rows if r.get('close_reason') == 'TIME')

            return {
                'total': len(rows),
                'wins': len(wins),
                'losses': len(losses),
                'wr': round(len(wins) / len(rows) * 100, 1) if rows else 0,
                'pf': round(pf, 2),
                'total_pnl': round(sum(r['pnl_pct'] for r in rows), 2),
                'total_pnl_usdt': round(sum(r['pnl_usdt'] for r in rows), 2),
                'avg_pnl': round(sum(r['pnl_pct'] for r in rows) / len(rows), 3),
                'avg_hold_sec': round(sum(r['hold_seconds'] for r in rows) / len(rows), 1),
                'tp_count': tp_count,
                'sl_count': sl_count,
                'time_count': time_count
            }

    # === PAIRS ===

    def save_pair(self, symbol: str, leader: str = 'BTC', reason: str = '') -> int:
        with self.get_cursor() as cur:
            cur.execute('''
            INSERT INTO pairs (symbol, leader, activated_at, reason, status)
            VALUES (?, ?, ?, ?, 'active')
            ''', (symbol, leader, get_gmt2_str(), reason))
            return cur.lastrowid

    def deactivate_pair(self, symbol: str, reason: str = ''):
        with self.get_cursor() as cur:
            cur.execute('''
            UPDATE pairs SET status='inactive', deactivated_at=?, reason=?
            WHERE symbol=? AND status='active'
            ''', (get_gmt2_str(), reason, symbol))

    def get_active_pairs(self) -> List[Dict]:
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM pairs WHERE status='active' ORDER BY activated_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def get_pairs_history(self, limit: int = 50) -> List[Dict]:
        with self.get_cursor() as cur:
            cur.execute('SELECT * FROM pairs ORDER BY id DESC LIMIT ?', (limit,))
            return [dict(r) for r in cur.fetchall()]

    def update_pair_stats_live(self, symbol: str, total_trades: int, wr: float, pf: float):
        with self.get_cursor() as cur:
            cur.execute('''
            UPDATE pairs SET total_trades=?, win_rate=?, profit_factor=?
            WHERE symbol=? AND status='active'
            ''', (total_trades, wr, pf, symbol))

    # === PAIR STATS (scanner snapshots) ===

    def save_pair_stat(self, stat: Dict):
        with self.get_cursor() as cur:
            cur.execute('''
            INSERT INTO pair_stats (symbol, timestamp, wr, pf, sharpe, lag_seconds,
                n_trades, avg_return, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stat['symbol'], get_gmt2_str(),
                stat.get('wr', 0), stat.get('pf', 0), stat.get('sharpe', 0),
                stat.get('lag_seconds', 0), stat.get('n_trades', 0),
                stat.get('avg_return', 0), stat.get('source', 'scanner')
            ))

    def get_pair_stats_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        with self.get_cursor() as cur:
            cur.execute('''
            SELECT * FROM pair_stats WHERE symbol=? ORDER BY id DESC LIMIT ?
            ''', (symbol, limit))
            return [dict(r) for r in cur.fetchall()]

    # === IMPULSES ===

    def save_impulse(self, impulse: Dict) -> int:
        with self.get_cursor() as cur:
            cur.execute('''
            INSERT INTO impulses (timestamp, leader, direction, magnitude,
                n_followers_entered, candle_open, candle_close,
                gap_seconds, gap_factor, quality, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                impulse.get('timestamp', get_gmt2_str()),
                impulse.get('leader', 'BTC'),
                impulse.get('direction'),
                impulse.get('magnitude'),
                impulse.get('n_followers_entered', 0),
                impulse.get('candle_open'),
                impulse.get('candle_close'),
                impulse.get('gap_seconds', 0),
                impulse.get('gap_factor', 1.0),
                impulse.get('quality', 0),
                impulse.get('status', 'accepted'),
            ))
            return cur.lastrowid

    def get_impulses(self, limit: int = 100) -> List[Dict]:
        with self.get_cursor() as cur:
            cur.execute('SELECT * FROM impulses ORDER BY id DESC LIMIT ?', (limit,))
            return [dict(r) for r in cur.fetchall()]

    def update_impulse_followers(self, impulse_id: int, n_followers: int):
        with self.get_cursor() as cur:
            cur.execute('UPDATE impulses SET n_followers_entered=? WHERE id=?',
                        (n_followers, impulse_id))

    # === STATE ===

    def get_state(self, key: str, default: str = None) -> Optional[str]:
        with self.get_cursor() as cur:
            cur.execute('SELECT value FROM state WHERE key=?', (key,))
            row = cur.fetchone()
            return row['value'] if row else default

    def set_state(self, key: str, value: str):
        with self.get_cursor() as cur:
            cur.execute('''
            INSERT INTO state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?
            ''', (key, value, get_gmt2_str(), value, get_gmt2_str()))

    # === LOGS ===

    def log(self, log_type: str, message: str, data: Any = None):
        with self.get_cursor() as cur:
            cur.execute('''
            INSERT INTO bot_logs (timestamp, type, message, data)
            VALUES (?, ?, ?, ?)
            ''', (get_gmt2_str(), log_type, message,
                  json.dumps(data, ensure_ascii=False) if data else None))

    def get_logs(self, log_type: str = None, limit: int = 100) -> List[Dict]:
        with self.get_cursor() as cur:
            if log_type:
                cur.execute(
                    'SELECT * FROM bot_logs WHERE type=? ORDER BY id DESC LIMIT ?',
                    (log_type, limit))
            else:
                cur.execute('SELECT * FROM bot_logs ORDER BY id DESC LIMIT ?', (limit,))
            return [dict(r) for r in cur.fetchall()]


# Singleton
db = Database()
