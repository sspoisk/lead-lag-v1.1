"""
BTC WebSocket monitor for OKX.

Streams real-time ticker updates and detects price impulses
using a rolling window — no need to wait for candle close.
"""

import json
import time
import threading
import logging
from collections import deque
from typing import Callable, Optional

import websocket

logger = logging.getLogger(__name__)


class BtcWebSocket:
    """
    Connects to OKX public WebSocket, subscribes to BTC-USDT-SWAP tickers.
    Detects impulse when BTC moves >= threshold over ws_window_sec seconds.

    Callbacks:
        on_impulse(direction, magnitude, price, ref_price) — impulse detected
        on_price(price)                                    — every tick
    """

    WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(self, config_getter: Callable, on_impulse: Callable, on_price: Callable):
        self.get_config = config_getter   # () -> dict, reads fresh config every time
        self.on_impulse = on_impulse
        self.on_price = on_price

        # Rolling price history: deque of (timestamp, price)
        self._history: deque = deque(maxlen=100_000)
        self._lock = threading.Lock()

        self.last_price: float = 0.0
        self.last_impulse_ts: float = 0.0

        self._ws: Optional[websocket.WebSocketApp] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run_loop, daemon=True, name='BtcWS')
        t.start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    def _run_loop(self):
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"[BTC-WS] Connection error: {e}")
            if self._running:
                logger.info("[BTC-WS] Reconnecting in 5s...")
                time.sleep(5)

    def _connect(self):
        ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = ws
        ws.run_forever(ping_interval=20, ping_timeout=10)

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    def _on_open(self, ws):
        logger.info("[BTC-WS] Connected to OKX")
        ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "tickers", "instId": "BTC-USDT-SWAP"}]
        }))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            event = data.get('event')
            if event == 'subscribe':
                logger.info("[BTC-WS] Subscribed: BTC-USDT-SWAP tickers")
                return
            if event == 'error':
                logger.error(f"[BTC-WS] Subscribe error: {data}")
                return

            if data.get('arg', {}).get('channel') != 'tickers':
                return

            ticker_list = data.get('data', [])
            if not ticker_list:
                return

            last_str = ticker_list[0].get('last')
            if not last_str:
                return

            price = float(last_str)
            now = time.time()

            self.last_price = price
            self.on_price(price)

            with self._lock:
                self.price_history_append(now, price)
                self._detect_impulse(now, price)

        except Exception as e:
            logger.error(f"[BTC-WS] Message error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"[BTC-WS] Error: {error}")

    def _on_close(self, ws, code, msg):
        logger.info(f"[BTC-WS] Closed: code={code}")

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def price_history_append(self, ts: float, price: float):
        """Append price tick and prune entries older than max window."""
        self._history.append((ts, price))
        # Prune: keep only last 5 minutes max
        cutoff = ts - 300
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    # ------------------------------------------------------------------
    # Impulse detection
    # ------------------------------------------------------------------

    def _detect_impulse(self, now: float, price: float):
        cfg = self.get_config().get('leader', {})
        threshold = cfg.get('impulse_threshold', 0.0018)
        cooldown = cfg.get('cooldown_seconds', 60)
        window_sec = cfg.get('ws_window_sec', 30)

        # Cooldown: don't fire too often
        if now - self.last_impulse_ts < cooldown:
            return

        if not self._history:
            return

        # Find reference price at the start of the window
        cutoff = now - window_sec
        ref_price = None
        for ts, p in self._history:
            if ts >= cutoff:
                ref_price = p
                break

        if ref_price is None or ref_price <= 0:
            return

        move = (price - ref_price) / ref_price

        if abs(move) < threshold:
            return

        direction = 'LONG' if move > 0 else 'SHORT'
        self.last_impulse_ts = now

        logger.info(
            f"[BTC-WS] IMPULSE {direction} {abs(move)*100:.3f}% "
            f"in {window_sec}s  (ref={ref_price:.1f} → {price:.1f})"
        )

        self.on_impulse(direction, move, price, ref_price)
