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


class BtcWsThread(threading.Thread):
    """
    V1.1: BTC WebSocket monitor.
    Replaces the old REST polling TraderThread.
    Detects impulses in real-time (< 1 second from BTC move to all orders placed).
    """
    daemon = True

    def __init__(self):
        super().__init__(name='BtcWsThread')
        self._ws: Optional[BtcWebSocket] = None

    def run(self):
        # Wait for scanner to populate pairs
        time.sleep(15)
        logger.info("[WS-THREAD] Starting BTC WebSocket monitor...")

        self._ws = BtcWebSocket(
            config_getter=load_config,
            on_impulse=self._on_impulse,
            on_price=self._on_price,
        )
        self._ws.start()

        # Keep thread alive while running
        while state.running:
            time.sleep(1)

        if self._ws:
            self._ws.stop()

    def _on_price(self, price: float):
        """Update BTC price in trader state."""
        if state.trader:
            state.trader.btc_price = price

    def _on_impulse(self, direction: str, magnitude: float, price: float, ref_price: float):
        """Called by BtcWebSocket when impulse detected. Runs in WS thread."""
        if not state.trader or state.trader.paused:
            return

        cfg = load_config()

        # Direction filter
        trading_cfg = cfg.get('trading', {})
        force = trading_cfg.get('force_direction')
        if force in ('LONG', 'SHORT'):
            direction = force
        else:
            if direction == 'LONG' and not trading_cfg.get('enable_long', True):
                logger.info(f"[WS-THREAD] SKIP: LONG disabled")
                self._save_impulse(direction, magnitude, price, ref_price, 0, 'skip_long_disabled')
                return
            if direction == 'SHORT' and not trading_cfg.get('enable_short', True):
                logger.info(f"[WS-THREAD] SKIP: SHORT disabled")
                self._save_impulse(direction, magnitude, price, ref_price, 0, 'skip_short_disabled')
                return

        # Get active pairs
        pairs = state.pair_manager.get_pairs_info() if state.pair_manager else []
        if not pairs:
            logger.info("[WS-THREAD] No active pairs, skipping impulse")
            self._save_impulse(direction, magnitude, price, ref_price, 0, 'skip_no_pairs')
            return

        full_symbols = [p['full_symbol'] for p in pairs]
        symbol_map   = {p['full_symbol']: p['symbol'] for p in pairs}

        # ---------------------------------------------------------------
        # Fetch ALL alt prices in ONE API call
        # ---------------------------------------------------------------
        prefetched: Dict[str, float] = {}
        try:
            tickers = state.exchange.fetch_tickers(full_symbols)
            for full_sym, t in tickers.items():
                p = t.get('last') or t.get('close') or 0
                if p:
                    short = symbol_map.get(full_sym, full_sym.split('/')[0])
                    prefetched[short] = float(p)
        except Exception as e:
            logger.warning(f"[WS-THREAD] fetch_tickers failed ({e}), will use cached prices")

        # Save impulse to DB
        impulse_id = self._save_impulse(direction, magnitude, price, ref_price, 0, 'accepted')

        # Open all positions fast
        symbols = [(p['symbol'], p['full_symbol']) for p in pairs]
        opened = state.trader.open_positions_fast(
            symbols, direction, impulse_id, magnitude, prefetched
        )

        db.update_impulse_followers(impulse_id, opened)

        logger.info(
            f"[WS-THREAD] Impulse {direction} {abs(magnitude)*100:.3f}% "
            f"→ opened {opened}/{len(symbols)} positions"
        )
        state.health['trader'] = 'ok'

        if state.telegram and opened > 0:
            state.telegram.send(
                f"⚡ BTC {direction} {abs(magnitude)*100:.3f}%\n"
                f"Opened {opened} positions: "
                f"{', '.join(s[0] for s in symbols[:opened])}"
            )

    def _save_impulse(self, direction, magnitude, price, ref_price,
                      n_followers, status) -> int:
        impulse = {
            'timestamp': get_gmt2_str(),
            'leader': 'BTC',
            'direction': direction,
            'magnitude': round(magnitude, 6),
            'candle_open': round(ref_price, 2),
            'candle_close': round(price, 2),
            'n_followers_entered': n_followers,
            'gap_seconds': 0,
            'gap_factor': 1.0,
            'quality': round(abs(magnitude), 6),
            'status': status,
        }
        return db.save_impulse(impulse)


class PriceThread(threading.Thread):
    """Updates prices for open positions and checks exits."""
    daemon = True

    def __init__(self):
        super().__init__(name='PriceThread')

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
            return

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
    return jsonify(db.get_trades(limit, symbol))


@app.route('/api/trades/stats')
def api_trade_stats():
    symbol = request.args.get('symbol')
    stats = db.get_trade_stats(symbol=symbol)
    config = load_config()
    stats['max_positions'] = config.get('trading', {}).get('max_positions', 10)
    stats['max_active'] = config.get('follower', {}).get('max_active', 10)
    return jsonify(stats)


@app.route('/api/impulses')
def api_impulses():
    limit = request.args.get('limit', 100, type=int)
    return jsonify(db.get_impulses(limit))


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

    # Start background threads
    state.running = True

    scanner_thread = ScannerThread()
    ws_thread      = BtcWsThread()      # V1.1: WebSocket replaces REST polling
    price_thread   = PriceThread()

    scanner_thread.start()
    ws_thread.start()
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
