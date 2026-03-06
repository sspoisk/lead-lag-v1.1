"""
Lead-Lag Trader — Flask web application with background trading threads.

Background threads:
- ScannerThread: runs pair scanner every N minutes
- TraderThread: monitors BTC impulses, opens positions (1s loop)
- PriceThread: updates open position prices (1s loop)
"""

import json
import os
import sys
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Custom .env loader
def _load_dotenv(path='.env'):
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if not os.environ.get(key):
                    os.environ[key] = value
    except Exception:
        pass

_load_dotenv()

# Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

from flask import Flask, render_template, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

import ccxt

from database import db, get_gmt2_str, get_gmt2_time
from scanner import PairScanner
from pair_manager import PairManager
from trader import LeadLagTrader
from btc_ws import BtcWebSocket

# ============================================================
# Flask app
# ============================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


@app.after_request
def add_headers(response):
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


# ============================================================
# Global state
# ============================================================

class AppState:
    def __init__(self):
        self.exchange: Optional[ccxt.Exchange] = None
        self.scanner: Optional[PairScanner] = None
        self.pair_manager: Optional[PairManager] = None
        self.trader: Optional[LeadLagTrader] = None
        self.running = False
        self.lock = threading.Lock()

        # Scanner state
        self.last_scan_time: Optional[str] = None
        self.next_scan_time: Optional[str] = None
        self.scan_running = False

        # Health
        self.health = {
            'exchange': 'unknown',
            'scanner': 'unknown',
            'trader': 'unknown'
        }

        # Telegram
        self.telegram = None

state = AppState()


# ============================================================
# Telegram
# ============================================================

class TelegramBot:
    def __init__(self, token: str, chat_id: str, prefix: str = '[LeadLag]'):
        self.token = token
        self.chat_id = chat_id
        self.prefix = prefix
        self.enabled = bool(token and chat_id)

    def send(self, text: str):
        if not self.enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.chat_id,
                'text': f"{self.prefix} {text}",
                'parse_mode': 'HTML'
            }, timeout=10)
        except Exception as e:
            logger.error(f"[TG] Send error: {e}")


# ============================================================
# Exchange init
# ============================================================

def init_exchange() -> ccxt.Exchange:
    config = load_config()
    exchange_id = config.get('exchange', 'okx')

    if exchange_id == 'okx':
        ex = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY', ''),
            'secret': os.environ.get('OKX_SECRET_KEY', ''),
            'password': os.environ.get('OKX_PASSPHRASE', ''),
            'options': {'defaultType': 'swap'},
            'enableRateLimit': True,
        })
    else:
        ex = ccxt.binance({
            'apiKey': os.environ.get('BINANCE_API_KEY', ''),
            'secret': os.environ.get('BINANCE_SECRET_KEY', ''),
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })

    try:
        ex.load_markets()
        logger.info(f"[EXCHANGE] {exchange_id.upper()} connected, "
                    f"{len(ex.markets)} markets")
        state.health['exchange'] = 'ok'
    except Exception as e:
        logger.error(f"[EXCHANGE] Connection failed: {e}")
        state.health['exchange'] = f'error: {e}'

    return ex


def load_config() -> Dict:
    with open('config.json', 'r') as f:
        return json.load(f)


# ============================================================
# Background Threads
# ============================================================

class ScannerThread(threading.Thread):
    """Runs pair scanner periodically."""
    daemon = True

    def __init__(self):
        super().__init__(name='ScannerThread')

    def run(self):
        config = load_config()
        interval = config.get('scanner', {}).get('interval_min', 30) * 60

        # Initial delay to let exchange connect
        time.sleep(10)
        logger.info("[SCANNER-THREAD] Started")

        while state.running:
            try:
                self._run_scan()
            except Exception as e:
                logger.error(f"[SCANNER-THREAD] Error: {e}")
                state.health['scanner'] = f'error: {e}'

            # Wait for next scan
            config = load_config()
            interval = config.get('scanner', {}).get('interval_min', 30) * 60
            state.next_scan_time = (get_gmt2_time() +
                                     timedelta(seconds=interval)).strftime('%H:%M:%S')

            # Sleep in 5s chunks for responsive shutdown
            elapsed = 0
            while elapsed < interval and state.running:
                time.sleep(5)
                elapsed += 5

    def _run_scan(self):
        state.scan_running = True
        state.health['scanner'] = 'scanning'
        logger.info("[SCANNER-THREAD] Running scan...")

        results = state.scanner.run_scan()
        state.last_scan_time = get_gmt2_str()
        state.scan_running = False

        if results:
            added = state.pair_manager.evaluate_new(results)
            removed = state.pair_manager.evaluate_existing()

            state.health['scanner'] = 'ok'

            # Telegram notification
            if state.telegram and (added or removed):
                msg = f"Scan complete: {len(results)} pairs analyzed\n"
                if added:
                    msg += f"Added: {', '.join(added)}\n"
                if removed:
                    msg += f"Removed: {', '.join(removed)}\n"
                top3 = results[:3]
                for r in top3:
                    msg += f"  {r['short_symbol']}: WR={r['wr']}% PF={r['pf']}\n"
                state.telegram.send(msg)
        else:
            state.health['scanner'] = 'ok (no results)'


class TraderThread(threading.Thread):
    """
    V1.1 (fixed): BTC 1m candle polling — same as V1.
    Waits for candle close before entering, gives confirmed impulse.
    """
    daemon = True

    def __init__(self):
        super().__init__(name='TraderThread')

    def run(self):
        time.sleep(15)  # Wait for scanner first run
        logger.info("[TRADER-THREAD] Started (1m candle mode)")

        while state.running:
            try:
                if not state.trader.paused:
                    self._check_impulse()
            except Exception as e:
                logger.error(f"[TRADER-THREAD] Error: {e}")

            time.sleep(1)

    def _check_impulse(self):
        impulse = state.trader.check_btc_impulse()
        if not impulse:
            return

        impulse_id = db.save_impulse(impulse)

        if impulse.get('status') != 'accepted':
            return

        pairs = state.pair_manager.get_pairs_info()
        if not pairs:
            logger.info("[TRADER-THREAD] No active pairs, skipping impulse")
            db.update_impulse_followers(impulse_id, 0)
            return

        symbols = [(p['symbol'], p['full_symbol']) for p in pairs]
        direction = impulse['direction']

        opened = state.trader.open_positions(
            symbols, direction, impulse_id, impulse['magnitude'])

        db.update_impulse_followers(impulse_id, opened)

        logger.info(f"[TRADER-THREAD] Impulse {direction} "
                    f"{abs(impulse['magnitude'])*100:.2f}% "
                    f"→ opened {opened}/{len(symbols)} positions")

        if state.telegram and opened > 0:
            state.telegram.send(
                f"BTC {direction} {abs(impulse['magnitude'])*100:.2f}%\n"
                f"Opened {opened} positions: "
                f"{', '.join(s[0] for s in symbols[:opened])}")

        state.health['trader'] = 'ok'


class PriceThread(threading.Thread):
    """Updates prices for open positions and checks exits.
    When idle (no open positions), refreshes cache for all active pairs every
    IDLE_INTERVAL seconds so the next impulse enters at pre-reaction prices.
    """
    daemon = True
    IDLE_INTERVAL = 5  # seconds between idle cache refreshes

    def __init__(self):
        super().__init__(name='PriceThread')
        self._idle_ticks = 0

    def run(self):
        time.sleep(20)
        logger.info("[PRICE-THREAD] Started")

        while state.running:
            try:
                self._update_prices()
            except Exception as e:
                logger.error(f"[PRICE-THREAD] Error: {e}")

            time.sleep(1)

    def _update_prices(self):
        positions = state.trader.get_open_positions()

        if not positions:
            # Idle: refresh cache every IDLE_INTERVAL seconds so it doesn't go stale
            self._idle_ticks += 1
            if self._idle_ticks < self.IDLE_INTERVAL:
                return
            self._idle_ticks = 0
            active_full = state.pair_manager.get_active_pairs()
            if not active_full:
                return
            try:
                tickers = state.exchange.fetch_tickers(active_full)
                idle_prices = {}
                for sym, t in tickers.items():
                    price = t.get('last', 0)
                    if price:
                        short = sym.split('/')[0].replace(':USDT', '')
                        idle_prices[short] = price
                if idle_prices:
                    state.trader.prices.update(idle_prices)
            except Exception as e:
                logger.debug(f"[PRICE-THREAD] Idle cache refresh error: {e}")
            return

        self._idle_ticks = 0
        # Fetch tickers for all open position symbols
        symbols = list(set(p['symbol'] for p in positions))
        prices = {}

        try:
            # Batch fetch is more efficient
            tickers = state.exchange.fetch_tickers(symbols)
            for sym, t in tickers.items():
                price = t.get('last', 0)
                if price:
                    # Map to short symbol
                    short = sym.split('/')[0] if '/' in sym else sym
                    short = short.replace(':USDT', '')
                    prices[short] = price
        except Exception:
            # Fallback: fetch one by one
            for sym in symbols:
                try:
                    t = state.exchange.fetch_ticker(sym)
                    short = sym.split('/')[0] if '/' in sym else sym
                    short = short.replace(':USDT', '')
                    prices[short] = t.get('last', 0)
                except Exception:
                    pass

        if prices:
            # LIVE: detect positions closed by exchange SL/TP
            state.trader.sync_exchange_positions()

            # Check for closes
            before = set(state.trader.positions.keys())
            state.trader.update_prices(prices)
            after = set(state.trader.positions.keys())

            closed = before - after
            for sym in closed:
                # Find the trade in DB
                trades = db.get_trades(limit=1, symbol=sym)
                if trades:
                    t = trades[0]
                    state.pair_manager.record_trade(sym, t['pnl_pct'])

                    # Telegram
                    if state.telegram:
                        icon = '+' if t['pnl_pct'] > 0 else ''
                        state.telegram.send(
                            f"CLOSE {t['trade_id']} {sym} {t['side']}\n"
                            f"PnL: {icon}{t['pnl_pct']:.2f}% ({t['close_reason']})\n"
                            f"Hold: {t['hold_seconds']}s")


# ============================================================
# Flask Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    if not state.trader:
        return jsonify({'error': 'not initialized'}), 503

    status = state.trader.get_status()
    status['active_pairs'] = len(state.pair_manager.active_pairs) if state.pair_manager else 0
    status['last_scan'] = state.last_scan_time
    status['next_scan'] = state.next_scan_time
    status['scan_running'] = state.scan_running
    status['health'] = state.health
    return jsonify(status)


@app.route('/api/pairs')
def api_pairs():
    if not state.pair_manager:
        return jsonify([])
    return jsonify(state.pair_manager.get_pairs_info())


@app.route('/api/pairs/history')
def api_pairs_history():
    limit = request.args.get('limit', 50, type=int)
    return jsonify(db.get_pairs_history(limit))


@app.route('/api/trades')
def api_trades():
    limit = request.args.get('limit', 100, type=int)
    symbol = request.args.get('symbol')
    since = db.get_state('session_start')
    return jsonify(db.get_trades(limit, symbol, since=since))


@app.route('/api/trades/stats')
def api_trade_stats():
    symbol = request.args.get('symbol')
    since = db.get_state('session_start')
    stats = db.get_trade_stats(symbol=symbol, since=since)
    config = load_config()
    stats['max_positions'] = config.get('trading', {}).get('max_positions', 10)
    stats['max_active'] = config.get('follower', {}).get('max_active', 10)
    return jsonify(stats)


@app.route('/api/impulses')
def api_impulses():
    limit = request.args.get('limit', 100, type=int)
    since = db.get_state('session_start')
    return jsonify(db.get_impulses(limit, since=since))


@app.route('/api/positions')
def api_positions():
    if not state.trader:
        return jsonify([])
    return jsonify(state.trader.get_open_positions())


@app.route('/api/health')
def api_health():
    return jsonify(state.health)


@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    return jsonify(load_config())


@app.route('/api/settings', methods=['POST'])
def api_settings_post():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'no data'}), 400

        config = load_config()
        # Deep merge (2 levels)
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                for k2, v2 in value.items():
                    if isinstance(v2, dict) and isinstance(config[key].get(k2), dict):
                        config[key][k2].update(v2)
                    else:
                        config[key][k2] = v2
            else:
                config[key] = value

        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Reload in trader
        if state.trader:
            state.trader.reload_config()

        db.log('config', 'Settings updated', data)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pair/add', methods=['POST'])
def api_pair_add():
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    if not symbol:
        return jsonify({'error': 'symbol required'}), 400
    if state.pair_manager:
        state.pair_manager.add_pair_manual(symbol)
    return jsonify({'status': 'ok', 'symbol': symbol})


@app.route('/api/pairs/export')
def api_pairs_export():
    if not state.pair_manager:
        return jsonify([])
    pairs = state.pair_manager.get_pairs_info()
    return jsonify([{'symbol': p['symbol'], 'full_symbol': p['full_symbol']} for p in pairs])


@app.route('/api/pairs/import', methods=['POST'])
def api_pairs_import():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({'error': 'expected list of pairs'}), 400
    added = []
    for item in data:
        if isinstance(item, str):
            sym, full_sym = item.upper(), f"{item.upper()}/USDT:USDT"
        elif isinstance(item, dict):
            sym = item.get('symbol', '').upper()
            full_sym = item.get('full_symbol', f"{sym}/USDT:USDT")
        else:
            continue
        if sym and sym not in state.pair_manager.active_pairs:
            state.pair_manager.add_pair_manual(sym, full_sym)
            added.append(sym)
    logger.info(f"[IMPORT] Added {len(added)} pairs: {added}")
    return jsonify({'status': 'ok', 'added': added, 'count': len(added)})


@app.route('/api/pair/remove', methods=['POST'])
def api_pair_remove():
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    reason = data.get('reason', 'manual')
    if not symbol:
        return jsonify({'error': 'symbol required'}), 400
    if state.pair_manager:
        state.pair_manager.remove_pair(symbol, reason)
    return jsonify({'status': 'ok', 'symbol': symbol})


@app.route('/api/close_all', methods=['POST'])
def api_close_all():
    if state.trader:
        count = state.trader.close_all('manual')
        return jsonify({'status': 'ok', 'closed': count})
    return jsonify({'error': 'not initialized'}), 503


@app.route('/api/pause', methods=['POST'])
def api_pause():
    if state.trader:
        state.trader.paused = not state.trader.paused
        status = 'paused' if state.trader.paused else 'running'
        db.log('control', f'Trader {status}', {})
        return jsonify({'status': status})
    return jsonify({'error': 'not initialized'}), 503


@app.route('/api/reset', methods=['POST'])
def api_reset():
    if state.trader:
        state.trader.reset_stats()
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'not initialized'}), 503


@app.route('/api/scan/run', methods=['POST'])
def api_scan_run():
    """Trigger manual scan."""
    if state.scan_running:
        return jsonify({'status': 'already running'})

    def _run():
        try:
            state.scan_running = True
            results = state.scanner.run_scan()
            state.last_scan_time = get_gmt2_str()
            if results and state.pair_manager:
                state.pair_manager.evaluate_new(results)
                state.pair_manager.evaluate_existing()
        except Exception as e:
            logger.error(f"[MANUAL-SCAN] Error: {e}")
        finally:
            state.scan_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/api/logs')
def api_logs():
    log_type = request.args.get('type')
    limit = request.args.get('limit', 50, type=int)
    return jsonify(db.get_logs(log_type, limit))


@app.route('/api/env', methods=['GET'])
def api_env_get():
    """Return which keys are set (masked), never the actual values."""
    keys = ['OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_PASSPHRASE']
    return jsonify({k: bool(os.environ.get(k)) for k in keys})


@app.route('/api/env', methods=['POST'])
def api_env_post():
    """Save API keys to .env file and reload into os.environ."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'no data'}), 400

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

    # Read existing .env lines, update/add our keys
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

    allowed = {'OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_PASSPHRASE'}
    updates = {k: v for k, v in data.items() if k in allowed and v}

    # Replace existing keys in-place
    existing_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split('=', 1)[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}\n")
            existing_keys.add(key)
        else:
            new_lines.append(line)

    # Append new keys not yet in file
    for key, val in updates.items():
        if key not in existing_keys:
            new_lines.append(f"{key}={val}\n")

    with open(env_path, 'w') as f:
        f.writelines(new_lines)

    os.chmod(env_path, 0o600)

    # Reload into current process
    for key, val in updates.items():
        os.environ[key] = val

    db.log('config', 'API keys updated', {k: '***' for k in updates})
    logger.info(f"[ENV] Updated keys: {list(updates.keys())}")
    return jsonify({'status': 'ok', 'updated': list(updates.keys())})


@app.route('/api/restart', methods=['POST'])
def api_restart():
    """Restart the bot via systemd."""
    db.log('control', 'Restart requested from UI', {})
    def _restart():
        time.sleep(1)
        os.system('systemctl restart lead-lag-v1.1.service')
    threading.Thread(target=_restart, daemon=True).start()
    return jsonify({'status': 'restarting'})


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()
    port = config.get('port', 8086)

    logger.info("=" * 60)
    logger.info("  LEAD-LAG TRADER starting...")
    logger.info(f"  Mode: {config.get('trade_mode', 'PAPER')}")
    logger.info(f"  Exchange: {config.get('exchange', 'okx').upper()}")
    logger.info(f"  Port: {port}")
    logger.info("=" * 60)

    # Init exchange
    state.exchange = init_exchange()

    # Init components
    state.scanner = PairScanner(exchange=state.exchange)
    state.pair_manager = PairManager()
    state.trader = LeadLagTrader(exchange=state.exchange)

    # Telegram
    tg_cfg = config.get('telegram', {})
    token = tg_cfg.get('bot_token') or os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = tg_cfg.get('chat_id') or os.environ.get('TELEGRAM_CHAT_ID', '')
    prefix = tg_cfg.get('prefix', '[LeadLag]')
    state.telegram = TelegramBot(token, chat_id, prefix)
    if state.telegram.enabled:
        logger.info("[TG] Telegram notifications enabled")
        state.telegram.send("Lead-Lag Trader started")

    # Mark session start — UI shows only trades/impulses from this point
    db.set_state('session_start', get_gmt2_str())

    # LIVE: read real balance and sync open positions from exchange
    state.trader.sync_live_on_start()

    # Start background threads
    state.running = True

    scanner_thread = ScannerThread()
    trader_thread  = TraderThread()
    price_thread   = PriceThread()

    scanner_thread.start()
    trader_thread.start()
    price_thread.start()

    logger.info("[APP] Background threads started")

    # Run Flask
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("[APP] Shutting down...")
    finally:
        state.running = False
        if state.trader:
            state.trader.close_all('shutdown')
        logger.info("[APP] Stopped")


if __name__ == '__main__':
    main()
