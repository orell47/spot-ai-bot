import os
import sys
import json
import logging
import time
import sqlite3
from datetime import datetime, timezone, timedelta

import yaml
import numpy as np
import pandas as pd
import feedparser
import requests

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("spot-ai")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB = os.path.join(DATA_DIR, "bot.db")
UTC = timezone.utc


def now():
    return datetime.now(UTC)


def iso(dt=None):
    return (dt or now()).isoformat()


def pct(a, b):
    return (a / b - 1) * 100 if b else 0.0


def money(x):
    return f"${x:,.2f}"


def price_text(x):
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


class Store:
    def __init__(self):
        self.db = sqlite3.connect(DB, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS candidates(
          id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, score REAL, decision TEXT,
          price REAL, change_pct REAL, volume_ratio REAL, breakout REAL,
          risk REAL, reasons TEXT, raw_json TEXT);
        CREATE TABLE IF NOT EXISTS trades(
          id INTEGER PRIMARY KEY, ts_open TEXT, ts_close TEXT, symbol TEXT,
          side TEXT, entry REAL, exit REAL, qty REAL, stop REAL, target REAL,
          pnl_pct REAL, pnl_usd REAL, score REAL, reason TEXT, outcome TEXT,
          position_usd REAL, high_watermark REAL, partial_closed INTEGER DEFAULT 0,
          close_reason TEXT);
        CREATE TABLE IF NOT EXISTS portfolio(
          id INTEGER PRIMARY KEY, ts TEXT, cash REAL, positions_value REAL,
          equity REAL, peak_equity REAL, drawdown_pct REAL, daily_pnl_usd REAL,
          open_positions INTEGER, kill_switch INTEGER, note TEXT);
        CREATE TABLE IF NOT EXISTS observations(
          id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, data_json TEXT);
        CREATE TABLE IF NOT EXISTS settings(
          key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(ts_close);
        CREATE INDEX IF NOT EXISTS idx_candidates_ts ON candidates(ts);
        CREATE INDEX IF NOT EXISTS idx_observations_symbol_ts ON observations(symbol, ts);
        """)
        self._migrate_trades()
        self.db.commit()

    def _migrate_trades(self):
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(trades)").fetchall()}
        additions = {
            "entry_qty": "REAL",
            "realized_pnl_usd": "REAL DEFAULT 0",
            "tp1_price": "REAL",
            "tp2_price": "REAL",
            "tp3_price": "REAL",
            "target_pct": "REAL",
            "tp_quality": "REAL",
            "thesis": "TEXT",
            "exchange": "TEXT",
            "last_thesis": "TEXT",
        }
        for name, typ in additions.items():
            if name not in cols:
                self.db.execute(f"ALTER TABLE trades ADD COLUMN {name} {typ}")

    def candidate(self, x):
        self.db.execute(
            """INSERT INTO candidates
            (ts,symbol,score,decision,price,change_pct,volume_ratio,breakout,risk,reasons,raw_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(), x["symbol"], x["score"], x["decision"], x["price"],
             x["change_pct"], x["volume_ratio"], x["breakout"], x["risk"],
             json.dumps(x.get("reasons", [])), json.dumps(x)))
        self.db.commit()
        return self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def open_trades(self):
        return pd.read_sql_query(
            "SELECT * FROM trades WHERE ts_close IS NULL ORDER BY id", self.db)

    def add_trade(self, x):
        self.db.execute(
            """INSERT INTO trades
            (ts_open,symbol,side,entry,qty,stop,target,score,reason,outcome,
             position_usd,high_watermark,partial_closed,close_reason,entry_qty,
             realized_pnl_usd,tp1_price,tp2_price,tp3_price,target_pct,tp_quality,thesis,exchange,last_thesis)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(), x["symbol"], "BUY", x["entry"], x["qty"], x["invalidation_price"],
             x["tp1_price"], x["score"], json.dumps(x.get("reasons", [])), "OPEN",
             x["entry"] * x["qty"], x["entry"], 0, None, x["qty"], 0.0,
             x["tp1_price"], x["tp2_price"], x["tp3_price"], x["target_pct"],
             x["tp_quality"], x["thesis"], "bybit_spot", "ACTIVE"))
        self.db.commit()
        return self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def close_trade(self, trade_id, exit_price, qty=None, reason="EXIT"):
        row = self.db.execute(
            "SELECT * FROM trades WHERE id=? AND ts_close IS NULL", (trade_id,)
        ).fetchone()
        if not row:
            return None
        entry = float(row["entry"])
        old_qty = float(row["qty"])
        close_qty = min(old_qty, qty if qty is not None else old_qty)
        pnl_pct = pct(exit_price, entry)
        pnl_usd = (exit_price - entry) * close_qty
        old_realized = float(row["realized_pnl_usd"] or 0)
        total_realized = old_realized + pnl_usd

        if close_qty < old_qty - 1e-12:
            new_qty = old_qty - close_qty
            self.db.execute(
                """UPDATE trades SET qty=?, position_usd=?, partial_closed=1,
                realized_pnl_usd=?, high_watermark=?, close_reason=? WHERE id=?""",
                (new_qty, new_qty * entry, total_realized,
                 max(float(row["high_watermark"] or entry), exit_price), reason, trade_id))
            fully_closed = False
            total_pnl = total_realized
        else:
            outcome = "WIN" if total_realized > 0 else ("LOSS" if total_realized < 0 else "FLAT")
            self.db.execute(
                """UPDATE trades SET ts_close=?, exit=?, pnl_pct=?, pnl_usd=?,
                outcome=?, close_reason=?, realized_pnl_usd=? WHERE id=?""",
                (iso(), exit_price, pnl_pct, total_realized, outcome, reason,
                 total_realized, trade_id))
            fully_closed = True
            total_pnl = total_realized
        self.db.commit()
        return {
            "symbol": row["symbol"],
            "entry": entry,
            "exit": exit_price,
            "pnl_usd": pnl_usd,
            "total_pnl_usd": total_pnl,
            "pnl_pct": pnl_pct,
            "qty": close_qty,
            "remaining_qty": old_qty - close_qty,
            "reason": reason,
            "fully_closed": fully_closed,
        }

    def update_trade_stop(self, trade_id, stop, high_watermark):
        self.db.execute(
            "UPDATE trades SET stop=?, high_watermark=? WHERE id=? AND ts_close IS NULL",
            (stop, high_watermark, trade_id))
        self.db.commit()

    def update_thesis(self, trade_id, state):
        self.db.execute(
            "UPDATE trades SET last_thesis=? WHERE id=? AND ts_close IS NULL",
            (state, trade_id))
        self.db.commit()

    def set(self, key, value):
        self.db.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))
        self.db.commit()

    def get(self, key, default=None):
        r = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def record_portfolio(self, cash, positions_value, equity, peak, daily_pnl, kill, note=""):
        dd = max(0, (peak - equity) / peak * 100) if peak else 0
        self.db.execute(
            """INSERT INTO portfolio
            (ts,cash,positions_value,equity,peak_equity,drawdown_pct,daily_pnl_usd,
             open_positions,kill_switch,note)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (iso(), cash, positions_value, equity, peak, dd, daily_pnl,
             len(self.open_trades()), int(kill), note))
        self.db.commit()

    def realized_today(self):
        day = now().date().isoformat()
        r = self.db.execute(
            """SELECT COALESCE(SUM(pnl_usd),0) p FROM trades
            WHERE ts_close IS NOT NULL AND substr(ts_close,1,10)=?""", (day,)
        ).fetchone()
        return float(r["p"])

    def closed_count(self):
        return int(self.db.execute(
            "SELECT COUNT(*) c FROM trades WHERE ts_close IS NOT NULL"
        ).fetchone()["c"])

    def observation_exists(self, candidate_id, stage):
        pattern = f'"candidate_id": {int(candidate_id)}'
        stage_pattern = f'"stage": "{stage}"'
        r = self.db.execute(
            "SELECT 1 FROM observations WHERE data_json LIKE ? AND data_json LIKE ? LIMIT 1",
            (f"%{pattern}%", f"%{stage_pattern}%")
        ).fetchone()
        return r is not None

    def add_observation(self, data):
        self.db.execute(
            "INSERT INTO observations(ts,symbol,data_json) VALUES(?,?,?)",
            (iso(), data["symbol"], json.dumps(data)))
        self.db.commit()

    def due_candidates(self):
        since = (now() - timedelta(minutes=75)).isoformat()
        return self.db.execute(
            "SELECT id,ts,symbol,price,score,decision,raw_json FROM candidates WHERE ts>=? ORDER BY id",
            (since,)).fetchall()


class BybitMarket:
    """Bybit V5 public Spot market data. No private API key is used."""
    BASES = ["https://api.bybit.com", "https://api.bytick.com"]
    CATEGORY = "spot"

    STABLE_BASES = {
        "USDT", "USDC", "USD1", "USDE", "USDS", "DAI", "FDUSD", "TUSD",
        "USDD", "PYUSD", "FRAX", "RLUSD", "EURC", "EURT", "USTC", "BUSD"
    }

    LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "spot-ai-paper-bot/0.6"})
        self.instruments_cache = None
        self.base = None

    def _get(self, path, params=None, retries=3):
        last = None
        bases = [self.base] + [b for b in self.BASES if b != self.base] if self.base else self.BASES
        for base in bases:
            for attempt in range(retries):
                try:
                    r = self.s.get(base + path, params=params, timeout=15)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    data = r.json()
                    if data.get("retCode", 0) != 0:
                        raise RuntimeError(f"Bybit {data.get('retCode')}: {data.get('retMsg')}")
                    self.base = base
                    return data["result"]
                except Exception as e:
                    last = e
                    if attempt < retries - 1:
                        time.sleep(1.0 + attempt)
        raise RuntimeError(str(last))

    def instruments(self):
        if self.instruments_cache is not None:
            return self.instruments_cache
        result = self._get("/v5/market/instruments-info", {"category": self.CATEGORY, "limit": 1000})
        rows = result.get("list", [])
        out = {}
        for x in rows:
            symbol = str(x.get("symbol", "")).upper()
            base = str(x.get("baseCoin", "")).upper()
            quote = str(x.get("quoteCoin", "")).upper()
            status = str(x.get("status", ""))
            if quote != "USDT" or status != "Trading":
                continue
            if base in self.STABLE_BASES:
                continue
            if any(base.endswith(s) for s in self.LEVERAGED_SUFFIXES):
                continue
            out[symbol] = x
        self.instruments_cache = out
        return out

    def tickers(self):
        self.instruments()
        result = self._get("/v5/market/tickers", {"category": self.CATEGORY})
        out = {}
        for x in result.get("list", []):
            symbol = str(x.get("symbol", "")).upper()
            if symbol not in self.instruments_cache:
                continue
            last = float(x.get("lastPrice") or 0)
            if last <= 0:
                continue
            out[symbol] = {
                "symbol": symbol,
                "lastPrice": last,
                "volume24h": float(x.get("volume24h") or 0),
                "turnover24h": float(x.get("turnover24h") or 0),
                "price24hPcnt": float(x.get("price24hPcnt") or 0) * 100,
                "high24h": float(x.get("highPrice24h") or last),
                "low24h": float(x.get("lowPrice24h") or last),
                "bid": float(x.get("bid1Price") or 0),
                "ask": float(x.get("ask1Price") or 0),
                "bid_size": float(x.get("bid1Size") or 0),
                "ask_size": float(x.get("ask1Size") or 0),
            }
        return out

    def candles(self, symbol, interval, limit=200):
        result = self._get("/v5/market/kline", {
            "category": self.CATEGORY,
            "symbol": symbol,
            "interval": str(interval),
            "limit": min(int(limit), 1000),
        })
        rows = result.get("list", [])
        if not rows:
            raise RuntimeError(f"no kline data for {symbol}")
        # Bybit returns newest first.
        rows = list(reversed(rows))
        df = pd.DataFrame(rows, columns=[
            "ts", "open", "high", "low", "close", "volume", "turnover"
        ])
        for c in ["ts", "open", "high", "low", "close", "volume", "turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("ts").reset_index(drop=True)

    def orderbook(self, symbol):
        result = self._get("/v5/market/orderbook", {
            "category": self.CATEGORY,
            "symbol": symbol,
            "limit": 25,
        })
        bids = np.array([[float(x[0]), float(x[1])] for x in result.get("b", [])])
        asks = np.array([[float(x[0]), float(x[1])] for x in result.get("a", [])])
        if len(bids) == 0 or len(asks) == 0:
            raise ValueError("empty orderbook")
        mid = (bids[0, 0] + asks[0, 0]) / 2
        spread = (asks[0, 0] - bids[0, 0]) / mid * 100 if mid else 99
        depth = (bids[:, 0] * bids[:, 1]).sum() + (asks[:, 0] * asks[:, 1]).sum()
        return {"spread_pct": float(spread), "depth_usd": float(depth)}


class News:
    FEEDS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss"
    ]

    def __init__(self):
        self.items = []
        for url in self.FEEDS:
            try:
                f = feedparser.parse(url)
                for e in f.entries[:100]:
                    self.items.append({
                        "title": getattr(e, "title", ""),
                        "summary": getattr(e, "summary", ""),
                        "link": getattr(e, "link", ""),
                    })
            except Exception as ex:
                log.warning("news feed: %s", ex)

    def recent(self, symbol):
        base = symbol.replace("USDT", "").lower()
        hits = []
        for e in self.items:
            text = (e["title"] + " " + e["summary"]).lower()
            if base and base in text:
                hits.append(e)
        return hits[:10]


class Telegram:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)

    def send(self, text):
        if not self.enabled:
            log.warning("telegram is not configured")
            return False
        ok = True
        chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]
        for chunk in chunks:
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": chunk},
                    timeout=15)
                r.raise_for_status()
            except Exception as ex:
                ok = False
                log.warning("telegram send failed: %s", ex)
        return ok


class Analyzer:
    def __init__(self, market, news):
        self.m = market
        self.n = news

    @staticmethod
    def _volume_ratio(series, n=10):
        if len(series) < n + 1:
            return 0.0
        avg = series.iloc[-n-1:-1].mean()
        return float(series.iloc[-1] / avg) if avg > 0 else 0.0

    def analyze(self, symbol, ticker, fast_df=None, deep_orderbook=False):
        df = self.m.candles(symbol, CFG["exchange"]["primary_interval"], limit=200)
        c = df.iloc[-1]
        prev = df.iloc[-2]

        if len(df) < 100:
            raise RuntimeError("not enough 15m candles")

        ret_24 = pct(c.close, df.iloc[-97].close)
        hour_ret = pct(c.close, df.iloc[-5].close)
        hour_turnover = float(df.iloc[-4:]["turnover"].sum())
        prior_hours = []
        for end in range(len(df) - 4, 4, -4):
            block = df.iloc[end-4:end]
            if len(block) == 4:
                prior_hours.append(float(block["turnover"].sum()))
        hour_avg = float(np.mean(prior_hours[-12:])) if prior_hours else 0.0
        hour_volume_ratio = hour_turnover / hour_avg if hour_avg else 0.0

        vol15 = self._volume_ratio(df["turnover"], 10)

        lookback = int(CFG["strategy"]["breakout_lookback"])
        recent = df.iloc[-lookback-1:-1]
        recent_high = float(recent.high.max())
        recent_low = float(recent.low.min())
        breakout = max(0.0, pct(c.close, recent_high))
        breakout_raw = c.close > recent_high * (1 + CFG["strategy"]["breakout_buffer_pct"] / 100)

        retest = bool(
            breakout_raw and (
                c.low <= recent_high * (1 + CFG["strategy"]["retest_tolerance_pct"] / 100)
                or prev.low <= recent_high * (1 + CFG["strategy"]["retest_tolerance_pct"] / 100)
            ) and c.close > recent_high
        )

        ma7 = float(df.close.rolling(7).mean().iloc[-1])
        ma14 = float(df.close.rolling(14).mean().iloc[-1])
        ma28 = float(df.close.rolling(28).mean().iloc[-1])
        ma20 = float(df.close.rolling(20).mean().iloc[-1])
        trend_stack = c.close > ma7 > ma14 > ma28

        tr = pd.concat([
            df.high - df.low,
            (df.high - df.close.shift()).abs(),
            (df.low - df.close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(CFG["strategy"]["atr_period"]).mean().iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            raise RuntimeError("invalid ATR")
        atr_pct = float(atr / c.close * 100)

        fast_ret = 0.0
        fast_vr = 0.0
        fast_15m_ret = 0.0
        if fast_df is not None and len(fast_df) >= 20:
            fc = fast_df.iloc[-1]
            fast_ret = pct(fc.close, fc.open)
            fast_15m_ret = pct(fc.close, fast_df.iloc[-4].close) if len(fast_df) >= 4 else fast_ret
            fast_vr = self._volume_ratio(fast_df["turnover"], 10)

        # Find the nearest visible resistance above price in the recent 7-day window.
        window = df.iloc[max(0, len(df)-672):-4]
        above = window.high[window.high > c.close * 1.002] if len(window) else pd.Series(dtype=float)
        if len(above):
            resistance = float(above.min())
            room_pct = pct(resistance, c.close)
        else:
            resistance = float(c.close)
            room_pct = 5.0

        support = recent_low
        support_distance_pct = pct(c.close, support) if support > 0 else 0

        # Liquidity: ticker-level spread/depth first; deep orderbook only for stronger candidates.
        bid = float(ticker.get("bid") or 0)
        ask = float(ticker.get("ask") or 0)
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
            top_depth = bid * float(ticker.get("bid_size") or 0) + ask * float(ticker.get("ask_size") or 0)
        else:
            spread_pct = 99.0
            top_depth = 0.0
        depth_usd = top_depth
        if deep_orderbook:
            try:
                ob = self.m.orderbook(symbol)
                spread_pct = ob["spread_pct"]
                depth_usd = ob["depth_usd"]
            except Exception:
                pass

        news = self.n.recent(symbol)
        reasons = []
        score = 0.0
        risk = 0.0

        # 1h momentum: favors active movers without forcing a chase.
        if hour_ret >= 4:
            score += 15
            reasons.append("strong 1h momentum")
        elif hour_ret >= 2:
            score += 12
            reasons.append("positive 1h momentum")
        elif hour_ret >= 0.75:
            score += 8
            reasons.append("positive 1h momentum")
        elif hour_ret > 0:
            score += 4
            reasons.append("mild 1h momentum")
        elif hour_ret < -2:
            risk += 6
            reasons.append("1h momentum negative")

        if 0 < ret_24 <= 20:
            score += 8
        elif ret_24 > 20:
            score += 5
            risk += 4
            reasons.append("extended 24h move")
        elif ret_24 < -10:
            risk += 8
            reasons.append("strong recent decline")

        # Volume quality, not just volume size.
        if vol15 >= 5:
            score += 16
            reasons.append(f"15m volume {vol15:.1f}x")
        elif vol15 >= 3:
            score += 13
            reasons.append(f"15m volume {vol15:.1f}x")
        elif vol15 >= 2:
            score += 10
            reasons.append(f"15m volume {vol15:.1f}x")
        elif vol15 >= 1.3:
            score += 6
            reasons.append(f"15m volume {vol15:.1f}x")

        if hour_volume_ratio >= 3:
            score += 12
            reasons.append(f"1h volume {hour_volume_ratio:.1f}x")
        elif hour_volume_ratio >= 1.7:
            score += 9
            reasons.append(f"1h volume {hour_volume_ratio:.1f}x")
        elif hour_volume_ratio >= 1.2:
            score += 5

        if fast_vr >= 3:
            score += 8
            reasons.append(f"5m volume {fast_vr:.1f}x")
        elif fast_vr >= 1.8:
            score += 6
            reasons.append(f"5m volume {fast_vr:.1f}x")
        elif fast_vr >= 1.3:
            score += 3

        # Price response to volume.
        if (vol15 >= 3 or hour_volume_ratio >= 2) and hour_ret <= 0.25:
            risk += 8
            reasons.append("high volume without strong price response")
        if vol15 >= 8 and hour_ret < 1:
            risk += 8
            reasons.append("possible absorption/manipulation")

        # 5m confirmation.
        if fast_ret >= 1 and fast_vr >= 1.8:
            score += 9
            reasons.append("5m momentum + volume confirmation")
        elif fast_ret > 0 and fast_vr >= 1.3:
            score += 5
            reasons.append("5m confirmation")
        elif fast_ret < -1.5:
            risk += 5
            reasons.append("5m momentum weakening")

        # Structure / breakout / retest.
        if breakout_raw:
            score += 10
            reasons.append("breakout confirmed")
        if retest:
            score += 9
            reasons.append("breakout/retest hold")
        elif breakout_raw:
            risk += 3
            reasons.append("breakout without clean retest")

        if trend_stack:
            score += 6
            reasons.append("bullish MA structure")
        elif c.close > ma20:
            score += 3
            reasons.append("above 20-period mean")

        # Room to next resistance.
        if room_pct >= 5:
            score += 6
            reasons.append(f"room to resistance {room_pct:.1f}%")
        elif room_pct >= 2:
            score += 4
            reasons.append(f"room to resistance {room_pct:.1f}%")
        elif room_pct < 1:
            risk += 8
            reasons.append("resistance very close")

        if c.close > prev.close:
            score += 2
        if fast_15m_ret > 0:
            score += 2

        # Liquidity.
        min_liq = float(CFG["strategy"]["min_liquidity_usd"])
        if spread_pct <= float(CFG["strategy"]["max_spread_pct"]) and depth_usd >= min_liq:
            score += 8
            reasons.append("liquidity good")
        elif spread_pct <= float(CFG["strategy"]["max_spread_pct"]) * 1.8 and depth_usd >= min_liq * 0.25:
            score += 3
            reasons.append("liquidity acceptable")
        else:
            risk += 12
            reasons.append("liquidity/spread concern")

        if news:
            score += min(4, len(news))
            reasons.append(f"{len(news)} relevant news")

        # Chase protection is a penalty, not an absolute ban.
        if hour_ret > 6 and not retest:
            risk += 10
            reasons.append("chase risk after fast 1h move")
        if ret_24 > 50:
            risk += 12
            reasons.append("very extended 24h move")

        # Dynamic invalidation: below structure or ~2 ATR, whichever is farther but not below emergency floor.
        structural_stop = min(support, c.close - 1.8 * atr)
        invalidation_price = max(c.close * (1 - CFG["risk"]["emergency_loss_pct"] / 100), structural_stop)
        if invalidation_price >= c.close:
            invalidation_price = c.close * 0.92

        # TP ladder based on volatility + visible resistance.
        base_move = max(atr * float(CFG["strategy"]["tp_atr_multiple"]), c.close * 0.025)
        tp1 = max(c.close + base_move, c.close * (1 + CFG["strategy"]["tp1_min_pct"] / 100))
        if resistance > c.close and resistance < tp1:
            tp1 = c.close + max(atr * 0.9, c.close * 0.02)
        tp2 = max(tp1 + atr * 0.8, c.close * (1 + CFG["strategy"]["tp2_min_pct"] / 100))
        tp3 = max(tp2 + atr, c.close * (1 + CFG["strategy"]["tp3_min_pct"] / 100))
        if resistance > c.close and resistance < tp2:
            tp2 = min(tp2, resistance * 1.01)
        target_pct = pct(tp2, c.close)

        rr = (tp1 - c.close) / max(c.close - invalidation_price, c.close * 0.005)
        tp_quality = max(1.0, min(10.0,
            5 + min(3, max(0, rr - 1)) + (2 if room_pct >= target_pct else 0)
        ))

        confirmations = sum([
            hour_ret > 0 or breakout_raw,
            vol15 >= 1.3 or fast_vr >= 1.3,
            fast_ret > 0,
            breakout_raw or trend_stack,
            spread_pct <= float(CFG["strategy"]["max_spread_pct"]) * 1.8,
        ])

        min_score = float(CFG["strategy"]["min_score"])
        max_risk = float(CFG["safety"]["max_manipulation_risk"])
        rejection_reasons = []
        if score < min_score:
            rejection_reasons.append(f"score {score:.0f} < {min_score:.0f}")
        if risk > max_risk:
            rejection_reasons.append(f"risk {risk:.0f} > {max_risk:.0f}")
        if confirmations < int(CFG["strategy"]["min_confirmations"]):
            rejection_reasons.append(f"only {confirmations}/{CFG['strategy']['min_confirmations']} confirmations")
        if hour_ret < -3 and not breakout_raw:
            rejection_reasons.append("negative short-term momentum")

        decision = "BUY" if not rejection_reasons else "NO_TRADE"
        thesis = " | ".join(reasons[:6])

        return {
            "symbol": symbol,
            "score": float(min(100, score)),
            "decision": decision,
            "price": float(c.close),
            "change_pct": float(ret_24),
            "hour_change_pct": float(hour_ret),
            "hour_volume_usd": float(hour_turnover),
            "hour_volume_ratio": float(hour_volume_ratio),
            "hour_score": float(min(100, max(0, score - risk * 0.35))),
            "volume_ratio": float(vol15),
            "fast_change_pct": float(fast_ret),
            "fast_15m_change_pct": float(fast_15m_ret),
            "fast_volume_ratio": float(fast_vr),
            "breakout": float(breakout),
            "breakout_confirmed": bool(breakout_raw),
            "retest": bool(retest),
            "risk": float(risk),
            "confirmations": int(confirmations),
            "spread_pct": float(spread_pct),
            "depth_usd": float(depth_usd),
            "atr_pct": float(atr_pct),
            "support": float(support),
            "resistance": float(resistance),
            "room_to_resistance_pct": float(room_pct),
            "invalidation_price": float(invalidation_price),
            "stop": float(invalidation_price),
            "tp1_price": float(tp1),
            "tp2_price": float(tp2),
            "tp3_price": float(tp3),
            "target": float(tp2),
            "target_pct": float(target_pct),
            "tp_quality": float(tp_quality),
            "news": news,
            "reasons": reasons,
            "rejection_reasons": rejection_reasons,
            "thesis": thesis,
        }


class Portfolio:
    def __init__(self, store):
        self.db = store
        saved = store.get("paper_cash")
        self.cash = float(saved) if saved is not None else float(CFG["challenge"]["starting_equity_usd"])
        self.peak = float(store.get("paper_peak_equity", self.cash))
        self.kill = store.get("kill_switch", "0") == "1"

    def mark(self, prices):
        pos = 0.0
        for r in self.db.open_trades().to_dict("records"):
            pos += float(r["qty"]) * float(prices.get(r["symbol"], r["entry"]))
        eq = self.cash + pos
        self.peak = max(self.peak, eq)
        dd = (self.peak - eq) / self.peak * 100 if self.peak else 0
        daily = self.db.realized_today()
        if dd >= CFG["risk"]["max_drawdown_pct"] or daily <= -abs(self.peak) * CFG["risk"]["max_daily_loss_pct"] / 100:
            self.kill = True
        self.db.set("paper_cash", f"{self.cash:.12f}")
        self.db.set("paper_peak_equity", f"{self.peak:.12f}")
        self.db.set("kill_switch", "1" if self.kill else "0")
        self.db.record_portfolio(self.cash, pos, eq, self.peak, daily, self.kill,
                                 "kill switch" if self.kill else "")
        return eq, pos, dd, daily

    def reserve_for_buy(self, usd):
        if self.kill or usd > self.cash:
            return False
        if self.cash - usd < self.peak * CFG["risk"]["reserve_cash_pct"] / 100:
            return False
        self.cash -= usd
        self.db.set("paper_cash", f"{self.cash:.12f}")
        return True

    def credit(self, usd):
        self.cash += usd
        self.db.set("paper_cash", f"{self.cash:.12f}")


class Risk:
    def position_size(self, portfolio, a):
        equity, _, _, _ = portfolio.mark({a["symbol"]: a["price"]})
        if portfolio.kill:
            return 0
        # User-defined trade notional: normally $10-$15 per entry.
        # The exact size scales with score, but never exceeds the configured maximum.
        min_usd = float(CFG["strategy"]["trade_size_min_usd"])
        max_usd = float(CFG["strategy"]["trade_size_max_usd"])
        score_floor = float(CFG["strategy"]["min_score"])
        score_ceiling = float(CFG["strategy"]["exceptional_score"])
        score_ratio = 0.0 if score_ceiling <= score_floor else max(0.0, min(1.0, (float(a["score"]) - score_floor) / (score_ceiling - score_floor)))
        desired_usd = min_usd + (max_usd - min_usd) * score_ratio

        # Keep the old structural-risk guard. If a setup cannot safely support at least
        # the requested minimum notional, skip it rather than silently trading smaller.
        risk_usd = equity * CFG["risk"]["risk_per_trade_pct"] / 100
        per_unit = max(a["price"] - a["invalidation_price"], a["price"] * 0.01)
        risk_limited_usd = (risk_usd / per_unit) * a["price"]
        allowed_usd = min(desired_usd, risk_limited_usd)

        total_exposure = sum(float(x["position_usd"]) for x in portfolio.db.open_trades().to_dict("records"))
        max_total = equity * CFG["risk"]["max_total_exposure_pct"] / 100
        exposure_room = max(0.0, max_total - total_exposure)
        allowed_usd = min(allowed_usd, exposure_room)
        if allowed_usd < min_usd:
            return 0
        return allowed_usd / a["price"]


class PaperBroker:
    def __init__(self, store, portfolio, tg, market):
        self.db = store
        self.p = portfolio
        self.tg = tg
        self.m = market

    def _thesis(self, symbol):
        try:
            df = self.m.candles(symbol, 15, limit=120)
            c = df.iloc[-1]
            ma20 = df.close.rolling(20).mean().iloc[-1]
            support = df.low.iloc[-21:-1].min()
            vr = Analyzer._volume_ratio(df.turnover, 10)
            if c.close < support:
                return "BROKEN — support lost"
            if c.close < ma20 and vr >= 2:
                return "WEAK — below MA20 with active volume"
            if c.close >= ma20:
                return "ACTIVE — structure intact"
            return "WATCH — structure neutral"
        except Exception:
            return "UNKNOWN"

    def manage_exits(self, ticks):
        results = []
        for r in self.db.open_trades().to_dict("records"):
            try:
                sym = r["symbol"]
                price = float(ticks.get(sym, {}).get("lastPrice") or 0)
                if not price:
                    continue
                entry = float(r["entry"])
                qty = float(r["qty"])
                target = float(r["target"] or 0)
                tp2 = float(r["tp2_price"] or target)
                tp3 = float(r["tp3_price"] or target * 1.05)
                high = max(float(r["high_watermark"] or entry), price)
                self.db.update_trade_stop(r["id"], float(r["stop"] or entry * 0.92), high)

                gain = pct(price, entry)
                thesis = self._thesis(sym)
                self.db.update_thesis(r["id"], thesis)

                # Emergency protection only. The old 1.9% hard stop is gone.
                emergency = -float(CFG["risk"]["emergency_loss_pct"])
                if gain <= emergency:
                    out = self.db.close_trade(r["id"], price, reason="EMERGENCY_RISK")
                    if out:
                        self.p.credit(price * qty)
                        results.append(out)
                    continue

                if thesis.startswith("BROKEN") and gain <= float(CFG["strategy"]["structure_exit_max_loss_pct"]):
                    out = self.db.close_trade(r["id"], price, reason="THESIS_BROKEN")
                    if out:
                        self.p.credit(price * qty)
                        results.append(out)
                    continue

                partial_done = int(r["partial_closed"] or 0)
                if price >= target and partial_done == 0:
                    part = qty * float(CFG["strategy"]["partial_take_profit_pct"])
                    out = self.db.close_trade(r["id"], price, qty=part, reason="TP1")
                    if out:
                        self.p.credit(price * part)
                        results.append(out)
                        # After TP1, protect the remaining position around breakeven rather than using the old 1.9% stop.
                        self.db.update_trade_stop(r["id"], max(float(r["stop"] or entry * 0.92), entry), high)
                    continue

                # Profit protection is allowed only after the trade has already moved meaningfully in our favor.
                stored_stop = float(r["stop"] or entry * 0.92)
                if gain >= float(CFG["strategy"]["trailing_activation_pct"]) and stored_stop > entry and price <= stored_stop:
                    out = self.db.close_trade(r["id"], price, reason="PROFIT_TRAIL")
                    if out:
                        self.p.credit(price * qty)
                        results.append(out)
                    continue

                if price >= tp2 and partial_done == 1:
                    part = float(r["qty"]) * float(CFG["strategy"]["tp2_close_pct"])
                    out = self.db.close_trade(r["id"], price, qty=part, reason="TP2")
                    if out:
                        self.p.credit(price * part)
                        results.append(out)
                    continue

                # Final TP only for the remaining position; otherwise thesis-based trailing.
                if price >= tp3 and partial_done == 1:
                    out = self.db.close_trade(r["id"], price, reason="TP3")
                    if out:
                        self.p.credit(price * qty)
                        results.append(out)
                    continue

                # Once a trade is meaningfully profitable, protect profit with a soft trailing floor.
                if gain >= float(CFG["strategy"]["trailing_activation_pct"]):
                    trail = price * (1 - float(CFG["strategy"]["soft_trailing_pct"]) / 100)
                    current_stop = float(r["stop"] or entry * 0.92)
                    if trail > current_stop:
                        self.db.update_trade_stop(r["id"], trail, high)
            except Exception as ex:
                log.warning("exit %s: %s", r["symbol"], ex)
        return results

    def buy(self, a):
        qty = Risk().position_size(self.p, a)
        usd = qty * a["price"]
        if qty <= 0 or usd < float(CFG["risk"]["min_trade_usd"]):
            return None
        if not self.p.reserve_for_buy(usd):
            return None
        tid = self.db.add_trade(dict(a, qty=qty, entry=a["price"]))
        return tid


def adaptive_min_score(store):
    base = float(CFG["strategy"]["min_score"])
    n = store.closed_count()
    if not CFG["learning"]["adaptive_thresholds_enabled"] or n < CFG["learning"]["min_closed_trades_before_calibration"]:
        return base
    rows = store.db.execute(
        "SELECT pnl_usd FROM trades WHERE ts_close IS NOT NULL ORDER BY id DESC LIMIT ?",
        (CFG["learning"]["calibration_window"],)).fetchall()
    if not rows:
        return base
    wins = sum(1 for r in rows if float(r["pnl_usd"]) > 0)
    wr = wins / len(rows)
    if wr < 0.40:
        return min(CFG["learning"]["adaptive_threshold_ceiling"], base + 3)
    if wr > 0.60:
        return max(CFG["learning"]["adaptive_threshold_floor"], base - 2)
    return base


def capture_observations(store, ticks):
    """Create T+15/T+30/T+60 snapshots for recent candidates. observations was empty before this version."""
    current = now()
    for row in store.due_candidates():
        try:
            created = datetime.fromisoformat(row["ts"])
            age = (current - created).total_seconds() / 60
            stage = None
            if 12 <= age < 23:
                stage = "T+15m"
            elif 27 <= age < 42:
                stage = "T+30m"
            elif 55 <= age < 75:
                stage = "T+60m"
            if not stage or store.observation_exists(row["id"], stage):
                continue
            sym = row["symbol"]
            price_now = float(ticks.get(sym, {}).get("lastPrice") or 0)
            if not price_now:
                continue
            entry_price = float(row["price"])
            data = json.loads(row["raw_json"] or "{}")
            store.add_observation({
                "candidate_id": int(row["id"]),
                "stage": stage,
                "ts_candidate": row["ts"],
                "ts_observed": iso(),
                "symbol": sym,
                "decision": row["decision"],
                "score": float(row["score"]),
                "candidate_price": entry_price,
                "observed_price": price_now,
                "forward_return_pct": pct(price_now, entry_price),
                "hour_change_at_candidate": float(data.get("hour_change_pct", 0)),
                "hour_volume_ratio_at_candidate": float(data.get("hour_volume_ratio", 0)),
                "volume_ratio_at_candidate": float(data.get("volume_ratio", 0)),
                "risk_at_candidate": float(data.get("risk", 0)),
            })
        except Exception as ex:
            log.warning("observation: %s", ex)


def format_open_positions(store, ticks):
    rows = store.open_trades().to_dict("records")
    if not rows:
        return ["📂 OPEN POSITIONS", "━━━━━━━━━━━━━━", "אין עסקאות פתוחות כרגע."]
    lines = ["📂 OPEN POSITIONS", "━━━━━━━━━━━━━━"]
    for r in rows:
        sym = r["symbol"]
        entry = float(r["entry"])
        qty = float(r["qty"])
        current = float(ticks.get(sym, {}).get("lastPrice") or entry)
        pnl_pct = pct(current, entry)
        pnl_usd = (current - entry) * qty + float(r["realized_pnl_usd"] or 0)
        current_value = current * qty
        target = float(r["target"] or 0)
        target_pct = pct(target, entry) if target else 0
        thesis = str(r["last_thesis"] or "UNKNOWN")
        lines.extend([
            f"\n🪙 {sym}",
            f"📍 Entry: {price_text(entry)}",
            f"💵 Bought: {qty:.8f}  |  {money(float(r['position_usd']))}",
            f"💹 Now: {price_text(current)}",
            f"📦 Value: {money(current_value)}",
            f"📈 P&L: {pnl_pct:+.2f}%  |  {money(pnl_usd)}",
            f"🎯 TP2: {price_text(target)}  |  {target_pct:+.2f}%",
            f"🧠 Thesis: {thesis}",
        ])
    return lines


def send_exit(tg, x, row):
    tg.send(
        "🔴 PAPER EXIT\n"
        "━━━━━━━━━━━━━━\n\n"
        f"🪙 {x['symbol']}\n"
        f"📍 Bought: {price_text(x['entry'])}\n"
        f"📍 Sold: {price_text(x['exit'])}\n"
        f"📦 Qty sold: {x['qty']:.8f}\n"
        f"💰 This exit: {x['pnl_pct']:+.2f}% | {money(x['pnl_usd'])}\n"
        f"💵 Total trade P&L: {money(x['total_pnl_usd'])}\n"
        f"🧠 Reason: {x['reason']}"
    )


def send_buy(tg, a, qty, usd):
    tg.send(
        "🟢 PAPER BUY\n"
        "━━━━━━━━━━━━━━\n\n"
        f"🪙 {a['symbol']}\n"
        f"💵 Bought: {qty:.8f}\n"
        f"💰 Position: {money(usd)}\n"
        f"📍 Entry: {price_text(a['price'])}\n\n"
        f"🎯 Target attempt: +{a['target_pct']:.2f}%\n"
        f"🎯 TP1: {price_text(a['tp1_price'])} ({pct(a['tp1_price'], a['price']):+.2f}%)\n"
        f"🎯 TP2: {price_text(a['tp2_price'])} ({pct(a['tp2_price'], a['price']):+.2f}%)\n"
        f"🎯 TP3: {price_text(a['tp3_price'])} ({pct(a['tp3_price'], a['price']):+.2f}%)\n"
        f"⭐ TP Quality: {a['tp_quality']:.1f}/10\n\n"
        f"⭐ Trade Score: {a['score']:.0f}/100\n"
        f"⚠️ Risk: {a['risk']:.0f}/100\n"
        f"🔊 15m Vol: {a['volume_ratio']:.1f}x\n"
        f"🔊 1h Vol: {a['hour_volume_ratio']:.1f}x\n"
        f"⚡ 5m: {a['fast_change_pct']:+.2f}% | Vol {a['fast_volume_ratio']:.1f}x\n"
        f"🚀 Breakout: {'YES' if a['breakout_confirmed'] else 'NO'} | Retest: {'YES' if a['retest'] else 'NO'}\n"
        f"💧 Liquidity: spread {a['spread_pct']:.2f}%\n\n"
        "🧠 Why:\n" + "\n".join(f"• {r}" for r in a["reasons"][:9])
    )


def run_once():
    store = Store()
    market = BybitMarket()
    news = News()
    tg = Telegram()
    portfolio = Portfolio(store)
    broker = PaperBroker(store, portfolio, tg, market)

    try:
        syms_all = market.instruments()
        ticks = market.tickers()
    except Exception as e:
        log.error("BYBIT MARKET DATA UNAVAILABLE - NO TRADE: %s", e)
        tg.send("⚠️ BYBIT DATA UNAVAILABLE\nNo trade was made.\nBot will retry next scan.")
        return

    # Balanced universe: liquid leaders + fast movers, while excluding stablecoins.
    eligible = [s for s in ticks if s in syms_all and ticks[s]["turnover24h"] >= CFG["exchange"]["min_24h_turnover_usd"]]
    top_liquid = sorted(eligible, key=lambda s: ticks[s]["turnover24h"], reverse=True)[:CFG["exchange"]["liquid_bucket"]]
    top_movers = sorted(eligible, key=lambda s: ticks[s]["price24hPcnt"], reverse=True)[:CFG["exchange"]["mover_bucket"]]
    alt_movers = sorted(
        [s for s in eligible if ticks[s]["turnover24h"] < CFG["exchange"]["large_cap_turnover_usd"]],
        key=lambda s: ticks[s]["price24hPcnt"], reverse=True
    )[:CFG["exchange"]["altcoin_bucket"]]
    syms = list(dict.fromkeys(top_liquid + top_movers + alt_movers))[:CFG["exchange"]["universe_limit"]]

    # Always manage old open paper positions, even if a symbol later leaves the new universe.
    exits = broker.manage_exits(ticks)
    for x in exits:
        row = store.db.execute("SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT 1", (x["symbol"],)).fetchone()
        send_exit(tg, x, row)

    # Build 5m cache for every scanned symbol. 80-ish symbols is intentional: more altcoin coverage.
    fast_cache = {}
    for s in syms:
        try:
            fast_cache[s] = market.candles(s, CFG["exchange"]["fast_interval"], limit=120)
        except Exception as e:
            log.warning("5m %s: %s", s, e)

    min_score = adaptive_min_score(store)
    old_min = CFG["strategy"]["min_score"]
    CFG["strategy"]["min_score"] = min_score
    ranked = []
    bought_ids = set()

    try:
        analyzer = Analyzer(market, news)
        for s in syms:
            try:
                # Deep orderbook only for candidates already showing meaningful momentum/volume.
                a = analyzer.analyze(s, ticks[s], fast_cache.get(s), deep_orderbook=False)
                if a["score"] >= CFG["strategy"]["deep_orderbook_score"] or a["hour_change_pct"] >= CFG["strategy"]["deep_orderbook_1h_pct"]:
                    a = analyzer.analyze(s, ticks[s], fast_cache.get(s), deep_orderbook=True)

                cid = store.candidate(a)
                a["candidate_id"] = cid
                ranked.append(a)

                log.info(
                    "🔎 CHECK %s | score=%.0f | risk=%.0f | 1h=%+.2f%% | 1h vol=%.1fx | 15m vol=%.1fx | 5m=%+.2f%% | %s",
                    s, a["score"], a["risk"], a["hour_change_pct"],
                    a["hour_volume_ratio"], a["volume_ratio"], a["fast_change_pct"], a["decision"]
                )

                if a["decision"] == "BUY" and len(store.open_trades()) < int(CFG["strategy"]["max_open_positions"]) and not portfolio.kill:
                    tid = broker.buy(a)
                    if tid:
                        bought_ids.add(tid)
                        qty = float(store.db.execute("SELECT qty FROM trades WHERE id=?", (tid,)).fetchone()["qty"])
                        send_buy(tg, a, qty, qty * a["price"])
                
            except Exception as e:
                log.warning("%s: %s", s, e)
    finally:
        CFG["strategy"]["min_score"] = old_min

    # Observations are captured after candidates have been stored.
    capture_observations(store, ticks)

    eq, pos, dd, daily = portfolio.mark({s: ticks[s]["lastPrice"] for s in ticks})

    # Open position report on every scan.
    open_lines = format_open_positions(store, ticks)
    tg.send("\n".join(open_lines))

    top5 = sorted(ranked, key=lambda x: (x["hour_score"], x["score"], x["hour_volume_ratio"]), reverse=True)[:5]
    top_lines = [
        "📊 TOP 5 — CURRENT OPPORTUNITIES",
        "━━━━━━━━━━━━━━",
        "Bybit Spot only • Altcoins included • no stablecoins",
        "",
    ]
    for i, x in enumerate(top5, 1):
        top_lines.extend([
            f"{i}. {x['symbol']}",
            f"⭐ Score: {x['score']:.0f}  |  ⚠️ Risk: {x['risk']:.0f}",
            f"📈 1h: {x['hour_change_pct']:+.2f}%  |  24h: {x['change_pct']:+.2f}%",
            f"🔊 1h Vol: {x['hour_volume_ratio']:.1f}x  |  15m: {x['volume_ratio']:.1f}x",
            f"⚡ 5m: {x['fast_change_pct']:+.2f}%  |  Vol {x['fast_volume_ratio']:.1f}x",
            f"🚀 Breakout: {'🟢 YES' if x['breakout_confirmed'] else '⚪ NO'}  |  Retest: {'🟢 YES' if x['retest'] else '⚪ NO'}",
            f"💧 Spread: {x['spread_pct']:.2f}%  |  Room: {x['room_to_resistance_pct']:.1f}%",
            f"📌 Decision: {x['decision']}",
            "",
        ])
    tg.send("\n".join(top_lines))

    start = float(CFG["challenge"]["starting_equity_usd"])
    target = float(CFG["challenge"]["target_equity_usd"])
    progress = (eq / start - 1) * 100
    tg.send(
        "🤖 SCAN COMPLETE\n"
        "━━━━━━━━━━━━━━\n"
        f"💰 Equity: {money(eq)}\n"
        f"📈 Progress: {progress:+.2f}%\n"
        f"🎯 Target: {money(target)}\n"
        f"📂 Open: {len(store.open_trades())}\n"
        f"📉 Drawdown: {dd:.2f}%\n"
        f"💵 Daily PnL: {money(daily)}\n"
        f"🔎 Scanned: {len(ranked)} Bybit Spot coins\n"
        f"🧠 Observations stored: {store.db.execute('SELECT COUNT(*) c FROM observations').fetchone()['c']}\n"
        f"🛑 Kill switch: {'ON' if portfolio.kill else 'OFF'}\n"
        f"⭐ Min score: {min_score:.0f}"
    )


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_once()
    else:
        while True:
            try:
                run_once()
            except Exception as e:
                log.exception(e)
            time.sleep(300)
