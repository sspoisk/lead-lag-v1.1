#!/usr/bin/env python3
"""
Lead-Lag — Бэктест с фильтром подтверждения альта.

После BTC-импульса входим ТОЛЬКО если альт сам уже начал двигаться
в ту же сторону (move за последние confirm_candles свечей >= confirm_pct).

Сравниваем:
  A) Без фильтра (текущая логика)
  B) С фильтром подтверждения

Запуск:
  python backtest_alt_confirm.py              # 72h, 60 пар
  python backtest_alt_confirm.py --hours 168  # 7 дней
"""
import argparse
import json
import os
import sys
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'backtest_cache')


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


# ─── BTC impulses ────────────────────────────────────────────────────────────

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


# ─── Simulation ──────────────────────────────────────────────────────────────

def simulate(follower: np.ndarray,
             impulses: List[Dict],
             tp_pct: float,
             sl_pct: float,
             hold_max_candles: int,
             fee: float,
             confirm_candles: int = 0,   # 0 = без фильтра
             confirm_pct: float = 0.0,   # минимальный сдвиг альта
             confirm_same_dir: bool = True,
             ) -> Optional[Dict]:
    """
    confirm_candles: смотреть N свечей ПЕРЕД входом (0 = выключено)
    confirm_pct:     мин. сдвиг альта в нужную сторону (0.001 = 0.1%)
    confirm_same_dir: True = альт должен двигаться В ТУ ЖЕ сторону что BTC
    """
    n = len(follower)
    pnls, reasons = [], []
    skipped = 0

    for imp in impulses:
        entry_idx = imp['index'] + 1
        if entry_idx >= n:
            continue

        direction = imp['direction']

        # ── Фильтр подтверждения альта ──────────────────────────────────
        # Реальный сценарий: BTC-импульс = закрытая свеча imp['index'].
        # В тот же момент закрыта такая же 1m-свеча альта (imp['index']).
        # Мы ЗНАЕМ её open и close ПЕРЕД тем как войти на следующей свече.
        if confirm_candles > 0 and confirm_pct > 0:
            # Смотрим N свечей альта, заканчивающихся на свече BTC-импульса
            imp_idx = imp['index']
            look_start = max(0, imp_idx - confirm_candles + 1)
            alt_open  = follower[look_start, 1]   # open первой смотровой свечи
            alt_close = follower[imp_idx, 4]       # close свечи BTC-импульса (известна до входа)
            if alt_open <= 0:
                continue
            alt_move = (alt_close - alt_open) / alt_open  # + = рост, - = падение

            if confirm_same_dir:
                if direction == 'SHORT' and alt_move > -confirm_pct:
                    skipped += 1
                    continue  # альт не падает достаточно → пропуск
                if direction == 'LONG'  and alt_move < confirm_pct:
                    skipped += 1
                    continue  # альт не растёт → пропуск
        # ────────────────────────────────────────────────────────────────

        entry = follower[entry_idx, 1]
        if entry <= 0:
            continue

        exit_price, reason = None, 'TIME'

        for j in range(entry_idx, min(entry_idx + hold_max_candles, n)):
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
            exit_price = follower[min(entry_idx + hold_max_candles - 1, n - 1), 4]

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
    tp_profit = float(wins.sum()) if len(wins) else 0
    sl_loss   = float(abs(losses.sum())) if len(losses) else 0
    return {
        'n':        len(pnls),
        'skipped':  skipped,
        'wins':     int(len(wins)),
        'losses':   int(len(losses)),
        'wr':       round(len(wins) / len(pnls) * 100, 1),
        'pf':       round(tp_profit / sl_loss, 2) if sl_loss > 0 else 999,
        'avg_pct':  round(float(arr.mean()) * 100, 3),
        'total_pct':round(float(arr.sum()) * 100, 2),
        'tp':       reasons.count('TP'),
        'sl':       reasons.count('SL'),
        'time':     reasons.count('TIME'),
    }


def aggregate(results_per_pair: Dict[str, Dict]) -> Dict:
    if not results_per_pair:
        return {}
    total_n = total_wins = total_losses = total_tp = total_sl = total_time = total_skipped = 0
    total_pnl = 0.0
    for s in results_per_pair.values():
        total_n       += s['n']
        total_wins    += s['wins']
        total_losses  += s['losses']
        total_tp      += s['tp']
        total_sl      += s['sl']
        total_time    += s['time']
        total_skipped += s.get('skipped', 0)
        total_pnl     += s['total_pct']
    wr = round(total_wins / total_n * 100, 1) if total_n else 0
    wins_sum  = sum(s['avg_pct'] * s['wins']   for s in results_per_pair.values() if s['avg_pct'] > 0)
    loss_sum  = abs(sum(s['avg_pct'] * s['losses'] for s in results_per_pair.values() if s['avg_pct'] <= 0))
    pf = round(wins_sum / loss_sum, 2) if loss_sum > 0 else 999
    return {
        'n': total_n, 'wins': total_wins, 'losses': total_losses,
        'wr': wr, 'pf': pf,
        'avg_pct': round(total_pnl / total_n, 3) if total_n else 0,
        'total_pct': round(total_pnl, 2),
        'tp': total_tp, 'sl': total_sl, 'time': total_time,
        'skipped': total_skipped,
        'pairs': len(results_per_pair),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_scenario(label: str,
                 btc: np.ndarray, follower_data: Dict[str, np.ndarray],
                 threshold: float, tp: float, sl: float, hold: int, fee: float,
                 force_dir: Optional[str],
                 confirm_candles: int, confirm_pct: float) -> Dict:
    impulses = find_impulses(btc, threshold, force_dir)
    results = {}
    for sym, candles in follower_data.items():
        s = simulate(candles, impulses, tp, sl, hold, fee,
                     confirm_candles, confirm_pct)
        if s:
            results[sym] = s
    agg = aggregate(results)
    agg['label'] = label
    agg['impulses_total'] = len(impulses)
    return agg


def print_scenario(r: Dict):
    if not r:
        print("  нет данных")
        return
    skipped_str = f"  пропущено по фильтру: {r.get('skipped',0)}" if r.get('skipped') else ""
    print(f"  Сделок: {r['n']}  Пар: {r['pairs']}  BTC-импульсов: {r['impulses_total']}{skipped_str}")
    print(f"  WR: {r['wr']}%   PF: {r['pf']}   Avg: {r['avg_pct']:+.3f}%   Total: {r['total_pct']:+.2f}%")
    print(f"  Выходы: TP={r['tp']}  SL={r['sl']}  TIME={r['time']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=72)
    ap.add_argument('--pairs', type=int, default=60)
    ap.add_argument('--min-vol', type=float, default=500000)
    args = ap.parse_args()

    with open(os.path.join(BASE_DIR, 'config.json')) as f:
        cfg = json.load(f)

    # Параметры "лучшей" комбинации из grid-поиска
    THRESHOLD = 0.005   # 0.50% — лучший из grid
    TP        = 0.005   # 0.50%
    SL        = 0.003   # 0.30%
    HOLD      = 3       # 3 минуты
    FEE       = cfg['trading'].get('fee_pct', 0.05) / 100
    FORCE_DIR = cfg['trading'].get('force_direction')
    if FORCE_DIR not in ('LONG', 'SHORT'):
        FORCE_DIR = None

    print(f"\n{'='*65}")
    print(f"  Lead-Lag — Бэктест с фильтром подтверждения альта")
    print(f"{'='*65}")
    print(f"  Параметры: threshold={THRESHOLD*100:.2f}%  TP={TP*100:.2f}%  SL={SL*100:.2f}%  hold={HOLD}m")
    print(f"  direction={FORCE_DIR or 'BTC-signal'}  lookback={args.hours}h  pairs={args.pairs}")
    print(f"{'='*65}\n")

    exchange = create_exchange()

    logger.info("Загружаю BTC свечи...")
    btc = fetch_candles(exchange, 'BTC/USDT:USDT', args.hours)
    if btc is None:
        print("Ошибка: не удалось загрузить BTC свечи")
        sys.exit(1)
    logger.info(f"BTC: {len(btc)} свечей")

    logger.info(f"Загружаю топ-{args.pairs} пар...")
    pairs = fetch_top_pairs(exchange, args.pairs, args.min_vol)
    follower_data = {}
    for i, symbol in enumerate(pairs):
        candles = fetch_candles(exchange, symbol, args.hours)
        if candles is not None and len(candles) >= 30:
            follower_data[symbol.split('/')[0]] = candles
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(pairs)} пар...")
        time.sleep(0.02)
    logger.info(f"Загружено {len(follower_data)} пар\n")

    # ── Блок 1: threshold=0.50% (лучший из grid), только ОБА направления ─
    print(f"\n{'─'*65}")
    print(f"  БЛОК 1: threshold=0.50%, TP=0.50%, SL=0.30%, hold=3m")
    print(f"{'─'*65}")

    # ── Сценарий A: без фильтра (baseline) ──────────────────────────────
    print("[ A ] Без фильтра (baseline — лучший из grid)")
    r_a = run_scenario("no_filter", btc, follower_data,
                       THRESHOLD, TP, SL, HOLD, FEE, FORCE_DIR, 0, 0.0)
    print_scenario(r_a)

    # ── Сценарий B: альт должен сдвинуться на 0.05% за 1 свечу ─────────
    print("\n[ B ] Фильтр: альт сдвинулся на ≥0.05% за 1 свечу")
    r_b = run_scenario("confirm_1c_005", btc, follower_data,
                       THRESHOLD, TP, SL, HOLD, FEE, FORCE_DIR, 1, 0.0005)
    print_scenario(r_b)

    # ── Сценарий C: 0.10% за 1 свечу ────────────────────────────────────
    print("\n[ C ] Фильтр: альт сдвинулся на ≥0.10% за 1 свечу")
    r_c = run_scenario("confirm_1c_010", btc, follower_data,
                       THRESHOLD, TP, SL, HOLD, FEE, FORCE_DIR, 1, 0.001)
    print_scenario(r_c)

    # ── Сценарий D: 0.05% за 2 свечи ────────────────────────────────────
    print("\n[ D ] Фильтр: альт сдвинулся на ≥0.05% за 2 свечи")
    r_d = run_scenario("confirm_2c_005", btc, follower_data,
                       THRESHOLD, TP, SL, HOLD, FEE, FORCE_DIR, 2, 0.0005)
    print_scenario(r_d)

    # ── Сценарий E: 0.15% за 2 свечи ────────────────────────────────────
    print("\n[ E ] Фильтр: альт сдвинулся на ≥0.15% за 2 свечи")
    r_e = run_scenario("confirm_2c_015", btc, follower_data,
                       THRESHOLD, TP, SL, HOLD, FEE, FORCE_DIR, 2, 0.0015)
    print_scenario(r_e)

    # ── Сводная таблица ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  СВОДНАЯ ТАБЛИЦА")
    print(f"{'='*65}")
    print(f"  {'Сценарий':<30} {'Сделок':>7} {'WR%':>6} {'PF':>6} {'Avg%':>7} {'Total%':>8}")
    print(f"  {'-'*62}")

    for r in [r_a, r_b, r_c, r_d, r_e]:
        if not r or not r.get('n'):
            continue
        label_map = {
            'no_filter':       'A) Без фильтра',
            'confirm_1c_005':  'B) ≥0.05% за 1 свечу',
            'confirm_1c_010':  'C) ≥0.10% за 1 свечу',
            'confirm_2c_005':  'D) ≥0.05% за 2 свечи',
            'confirm_2c_015':  'E) ≥0.15% за 2 свечи',
        }
        name = label_map.get(r['label'], r['label'])
        sign = '+' if r['total_pct'] >= 0 else ''
        avg_sign = '+' if r['avg_pct'] >= 0 else ''
        print(f"  {name:<30} {r['n']:>7} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"{avg_sign}{r['avg_pct']:>6.3f}% {sign}{r['total_pct']:>7.2f}%")

    print(f"\n  Вывод:")
    label_map2 = {
        'no_filter':       'Без фильтра',
        'confirm_1c_005':  'Фильтр ≥0.05% за 1 свечу',
        'confirm_1c_010':  'Фильтр ≥0.10% за 1 свечу',
        'confirm_2c_005':  'Фильтр ≥0.05% за 2 свечи',
        'confirm_2c_015':  'Фильтр ≥0.15% за 2 свечи',
    }
    scenarios = [r for r in [r_a, r_b, r_c, r_d, r_e] if r and r.get('n', 0) >= 5]
    if scenarios:
        best = max(scenarios, key=lambda x: x['avg_pct'])
        improvement = best['avg_pct'] - r_a['avg_pct']
        print(f"  Лучший: {label_map2.get(best['label'])} "
              f"(avg: {best['avg_pct']:+.3f}%  baseline: {r_a['avg_pct']:+.3f}%  "
              f"улучшение: {improvement:+.3f}%/сделку)")
        if best['avg_pct'] < 0:
            print(f"  ⚠ Все варианты убыточны")
        else:
            print(f"  ✓ Фильтр делает стратегию прибыльной")

    # ── Блок 2: threshold=0.18% (текущий конфиг), большая выборка ──────
    CUR_THRESH = cfg['leader'].get('impulse_threshold', 0.0018)
    CUR_TP     = cfg['trading'].get('tp_pct', 0.5) / 100
    CUR_SL     = cfg['trading'].get('sl_pct', 0.2) / 100
    CUR_HOLD   = max(1, round(cfg['trading'].get('hold_max_sec', 300) / 60))
    print(f"\n{'─'*65}")
    print(f"  БЛОК 2: текущий конфиг  threshold={CUR_THRESH*100:.2f}%  "
          f"TP={CUR_TP*100:.2f}%  SL={CUR_SL*100:.2f}%  hold={CUR_HOLD}m")
    print(f"{'─'*65}")

    print("[ A2 ] Без фильтра")
    r_a2 = run_scenario("no_filter", btc, follower_data,
                        CUR_THRESH, CUR_TP, CUR_SL, CUR_HOLD, FEE, FORCE_DIR, 0, 0.0)
    print_scenario(r_a2)

    print("\n[ B2 ] Фильтр ≥0.05% за 1 свечу")
    r_b2 = run_scenario("confirm_1c_005", btc, follower_data,
                        CUR_THRESH, CUR_TP, CUR_SL, CUR_HOLD, FEE, FORCE_DIR, 1, 0.0005)
    print_scenario(r_b2)

    print("\n[ C2 ] Фильтр ≥0.10% за 1 свечу")
    r_c2 = run_scenario("confirm_1c_010", btc, follower_data,
                        CUR_THRESH, CUR_TP, CUR_SL, CUR_HOLD, FEE, FORCE_DIR, 1, 0.001)
    print_scenario(r_c2)

    print("\n[ D2 ] Фильтр ≥0.15% за 2 свечи")
    r_d2 = run_scenario("confirm_2c_015", btc, follower_data,
                        CUR_THRESH, CUR_TP, CUR_SL, CUR_HOLD, FEE, FORCE_DIR, 2, 0.0015)
    print_scenario(r_d2)

    print(f"\n{'='*65}")
    print(f"  БЛОК 2 — СВОДНАЯ")
    print(f"  {'Сценарий':<30} {'Сделок':>7} {'WR%':>6} {'PF':>6} {'Avg%':>7} {'Total%':>8}")
    print(f"  {'-'*62}")
    for label, r in [("A2) Без фильтра", r_a2), ("B2) ≥0.05%/1св", r_b2),
                     ("C2) ≥0.10%/1св", r_c2), ("D2) ≥0.15%/2св", r_d2)]:
        if r and r.get('n'):
            s = '+' if r['total_pct'] >= 0 else ''
            a = '+' if r['avg_pct'] >= 0 else ''
            print(f"  {label:<30} {r['n']:>7} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
                  f"{a}{r['avg_pct']:>6.3f}% {s}{r['total_pct']:>7.2f}%")

    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
