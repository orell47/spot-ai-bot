import os, sys, json, logging, time, sqlite3
from datetime import datetime, timezone
import yaml, numpy as np, pandas as pd, feedparser, requests
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
        """)
        self.db.commit()

    def candidate(self, x):
        self.db.execute(
            """INSERT INTO candidates
            (ts,symbol,score,decision,price,change_pct,volume_ratio,breakout,risk,reasons,raw_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(), x["symbol"], x["score"], x["decision"], x["price"],
             x["change_pct"], x["volume_ratio"], x["breakout"], x["risk"],
             json.dumps(x["reasons"]), json.dumps(x)))
        self.db.commit()

    def open_trades(self):
        return pd.read_sql_query(
            "SELECT * FROM trades WHERE ts_close IS NULL ORDER BY id", self.db
        )

    def add_trade(self, x):
        self.db.execute(
            """INSERT INTO trades
            (ts_open,symbol,side,entry,qty,stop,target,score,reason,outcome,
             position_usd,high_watermark,partial_closed,close_reason)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(), x["symbol"], "BUY", x["entry"], x["qty"], x["stop"],
             x["target"], x["score"], json.dumps(x["reasons"]), "OPEN",
             x["entry"] * x["qty"], x["entry"], 0, None))
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
        pnl_pct = (exit_price / entry - 1) * 100
        pnl_usd = (exit_price - entry) * close_qty

        if close_qty < old_qty - 1e-12:
            new_qty = old_qty - close_qty
            self.db.execute(
                "UPDATE trades SET qty=?, position_usd=?, partial_closed=1, high_watermark=? WHERE id=?",
                (new_qty, new_qty * entry,
                 max(float(row["high_watermark"] or entry), exit_price), trade_id))
        else:
            outcome = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "FLAT")
            self.db.execute(
                """UPDATE trades SET ts_close=?, exit=?, pnl_pct=?, pnl_usd=?,
                outcome=?, close_reason=? WHERE id=?""",
                (iso(), exit_price, pnl_pct, pnl_usd, outcome, reason, trade_id))
        self.db.commit()
        return {
            "symbol": row["symbol"], "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
            "qty": close_qty, "reason": reason,
            "fully_closed": close_qty >= old_qty - 1e-12
        }

    def update_trade_stop(self, trade_id, stop, high_watermark):
        self.db.execute(
            "UPDATE trades SET stop=?, high_watermark=? WHERE id=? AND ts_close IS NULL",
            (stop, high_watermark, trade_id))
        self.db.commit()

    def set(self, key, value):
        self.db.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))
        self.db.commit()

    def get(self, key, default=None):
        r = self.db.execute(
            "SELECT value FROM settings WHERE key=?", (key,)).fetchone()
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


class Market:
    """Kraken public Spot market data. No API key and no private endpoints."""
    BASE = "https://api.kraken.com/0/public"

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "spot-ai-paper-bot/0.5"})
        self.pairs = None

    def _get(self, path, params=None, retries=3):
        last = None
        for attempt in range(retries):
            try:
                r = self.s.get(self.BASE + path, params=params, timeout=20)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.json()
                if data.get("error"):
                    raise RuntimeError("; ".join(data["error"]))
                return data["result"]
            except Exception as e:
                last = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(str(last))

    def symbols(self):
        if self.pairs is None:
            raw = self._get("/AssetPairs")
            out = {}
            for key, x in raw.items():
                if x.get("status") != "online":
                    continue
                quote = str(x.get("quote", "")).upper()
                alt = x.get("altname", key)
                wsname = x.get("wsname", "")
                if quote in ("USDT", "ZUSDT") or wsname.upper().endswith("/USDT"):
                    # Exclude leveraged-token style products.
                    upper = alt.upper()
                    if any(k in upper for k in ("UP", "DOWN", "BULL", "BEAR")):
                        continue
                    out[alt] = {"key": key, "alt": alt, "wsname": wsname}
            self.pairs = out
        return list(self.pairs.keys())

    def tickers(self):
        self.symbols()
        result = {}
        names = list(self.pairs.keys())

        # Kraken accepts multiple pairs. Keep batches small for reliability.
        for i in range(0, len(names), 40):
            batch = names[i:i + 40]
            data = self._get("/Ticker", {"pair": ",".join(batch)})
            for key, x in data.items():
                alt = self.pairs.get(key, {}).get("alt", key)
                last = float(x["c"][0])
                vol24 = float(x["v"][1])
                result[alt] = {
                    "symbol": alt,
                    "lastPrice": last,
                    "volume24h": vol24,
                    "turnover24h": last * vol24,
                    "raw": x
                }
        return result

    def candles(self, symbol, interval, limit=180):
        pair = self.pairs[symbol]["alt"] if symbol in self.pairs else symbol
        result = self._get("/OHLC", {"pair": pair, "interval": int(interval)})
        key = next((k for k in result.keys() if k != "last"), None)
        rows = result.get(key, [])
        if not rows:
            raise RuntimeError(f"no OHLC data for {symbol}")
        # Kraken returns: time, open, high, low, close, vwap, volume, count
        rows = rows[-limit:]
        df = pd.DataFrame(rows, columns=[
            "ts", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c])
        return df.sort_values("ts").reset_index(drop=True)

    def orderbook(self, symbol):
        pair = self.pairs[symbol]["alt"] if symbol in self.pairs else symbol
        result = self._get("/Depth", {"pair": pair, "count": 25})
        key = next(iter(result.keys()))
        bids = np.array([[float(x[0]), float(x[1])] for x in result[key]["bids"]])
        asks = np.array([[float(x[0]), float(x[1])] for x in result[key]["asks"]])
        if len(bids) == 0 or len(asks) == 0:
            raise ValueError("empty orderbook")
        mid = (bids[0, 0] + asks[0, 0]) / 2
        spread = (asks[0, 0] - bids[0, 0]) / mid * 100
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
                for e in f.entries[:80]:
                    self.items.append({
                        "title": getattr(e, "title", ""),
                        "summary": getattr(e, "summary", ""),
                        "link": getattr(e, "link", ""),
                        "feed": url
                    })
            except Exception as ex:
                log.warning("news feed: %s", ex)

    def recent(self, symbol):
        base = symbol.replace("USDT", "").replace("XBT", "BTC").lower()
        hits = []
        for e in self.items:
            t = (e["title"] + " " + e["summary"]).lower()
            if base and base in t:
                hits.append(e)
        return hits[:10]


class Telegram:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)

    def send(self, text):
        if not self.enabled:
            return
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=15)
            r.raise_for_status()
        except Exception as ex:
            log.warning("telegram: %s", ex)


class Analyzer:
    def __init__(self, market, news):
        self.m = market
        self.n = news

    def analyze(self, symbol, ticker, fast_df=None):
        df = self.m.candles(symbol, CFG["exchange"]["primary_interval"])
        ob = self.m.orderbook(symbol)
        c = df.iloc[-1]
        prev = df.iloc[-2]

        ret_24 = (c.close / df.iloc[-97].close - 1) * 100 if len(df) > 97 else (
            c.close / df.iloc[0].close - 1) * 100)

        lookback = CFG["strategy"]["breakout_lookback"]
        recent_high = df.iloc[-(lookback + 1):-1].high.max()
        breakout = max(0, (c.close / recent_high - 1) * 100)

        vol_ma = df.volume.iloc[-21:-1].mean()
        vr = c.volume / vol_ma if vol_ma else 0

        tr = pd.concat([
            df.high - df.low,
            (df.high - df.close.shift()).abs(),
            (df.low - df.close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(CFG["strategy"]["atr_period"]).mean().iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            raise RuntimeError("invalid ATR")

        atr_pct = atr / c.close * 100
        fast_ret = 0
        fast_vr = 0

        if fast_df is not None and len(fast_df) > 30:
            fc = fast_df.iloc[-1]
            fast_ret = (fc.close / fast_df.iloc[-25].close - 1) * 100
            fvm = fast_df.volume.iloc[-21:-1].mean()
            fast_vr = fc.volume / fvm if fvm else 0

        reasons = []
        score = 0
        risk = 0

        if 0 < ret_24 <= CFG["strategy"]["max_price_change_pct"]:
            score += 15
            reasons.append("positive momentum")
        elif ret_24 > CFG["strategy"]["max_price_change_pct"]:
            reasons.append("price already up >50%")
            risk += 60
        elif ret_24 < -10:
            risk += 15
            reasons.append("strong recent decline")

        if vr >= CFG["strategy"]["min_volume_ratio"]:
            score += 15
            reasons.append(f"15m volume {vr:.1f}x")

        if fast_ret > 0 and fast_vr >= 1.5:
            score += 10
            reasons.append("5m confirmation")

        if breakout > 0:
            score += 20
            reasons.append("resistance breakout")

        if ob["spread_pct"] <= 0.30 and ob["depth_usd"] >= CFG["strategy"]["min_liquidity_usd"]:
            score += 15
            reasons.append("liquidity acceptable")
        else:
            risk += 15
            reasons.append("liquidity/spread concern")

        if c.close > df.close.rolling(20).mean().iloc[-1]:
            score += 10
            reasons.append("above 20-period mean")

        if c.close > prev.close:
            score += 5
            reasons.append("positive candle")

        news = self.n.recent(symbol)
        if news:
            score += 5
            reasons.append(f"{len(news)} relevant news items")
        else:
            reasons.append("no relevant news found")

        if vr > 10 and ob["depth_usd"] < CFG["strategy"]["min_liquidity_usd"] * 2:
            risk += 25
            reasons.append("possible pump/manipulation pattern")

        decision = (
            "BUY"
            if score >= CFG["strategy"]["min_score"]
            and risk <= CFG["safety"]["max_manipulation_risk"]
            and ret_24 <= CFG["strategy"]["max_price_change_pct"]
            else "NO_TRADE"
        )

        stop = max(
            c.close - CFG["strategy"]["stop_atr_multiple"] * atr,
            c.close * 0.90
        )
        target = c.close + (c.close - stop) * CFG["strategy"]["final_take_profit_rr"]

        return {
            "symbol": symbol,
            "score": float(score),
            "decision": decision,
            "price": float(c.close),
            "change_pct": float(ret_24),
            "volume_ratio": float(vr),
            "fast_change_pct": float(fast_ret),
            "fast_volume_ratio": float(fast_vr),
            "breakout": float(breakout),
            "risk": float(risk),
            "spread_pct": ob["spread_pct"],
            "depth_usd": ob["depth_usd"],
            "atr_pct": float(atr_pct),
            "stop": float(stop),
            "target": float(target),
            "news": news,
            "reasons": reasons
        }


class Portfolio:
    def __init__(self, store):
        self.db = store
        saved = store.get("paper_cash")
        self.cash = float(saved) if saved is not None else float(
            CFG["challenge"]["starting_equity_usd"])
        self.peak = float(store.get("paper_peak_equity", self.cash))
        self.kill = store.get("kill_switch", "0") == "1"

    def mark(self, prices):
        pos = 0
        for r in self.db.open_trades().to_dict("records"):
            pos += float(r["qty"]) * float(prices.get(r["symbol"], r["entry"]))
        eq = self.cash + pos
        self.peak = max(self.peak, eq)
        dd = (self.peak - eq) / self.peak * 100 if self.peak else 0
        daily = self.db.realized_today()

        if (
            dd >= CFG["risk"]["max_drawdown_pct"]
            or dd >= CFG["challenge"]["max_total_drawdown_pct"]
            or daily <= -abs(self.peak) * CFG["risk"]["max_daily_loss_pct"] / 100
        ):
            self.kill = True

        self.db.set("paper_cash", f"{self.cash:.12f}")
        self.db.set("paper_peak_equity", f"{self.peak:.12f}")
        self.db.set("kill_switch", "1" if self.kill else "0")
        self.db.record_portfolio(
            self.cash, pos, eq, self.peak, daily, self.kill,
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

        base_pct = CFG["strategy"]["normal_position_max_pct"]
        if a["score"] >= CFG["strategy"]["exceptional_score"] and a["risk"] <= 20:
            base_pct = CFG["strategy"]["exceptional_position_max_pct"]
        elif a["score"] < CFG["strategy"]["exceptional_score"]:
            base_pct = CFG["strategy"]["base_position_pct"]

        cap = equity * min(base_pct, CFG["risk"]["max_position_pct"]) / 100
        risk_usd = equity * CFG["risk"]["risk_per_trade_pct"] / 100
        per_unit = max(a["price"] - a["stop"], a["price"] * 0.005)
        qty = min(cap / a["price"], risk_usd / per_unit)

        total_exposure = sum(
            float(x["position_usd"])
            for x in portfolio.db.open_trades().to_dict("records"))
        max_total = equity * CFG["risk"]["max_total_exposure_pct"] / 100
        qty = min(qty, max(0, max_total - total_exposure) / a["price"])
        return max(0, qty)


class PaperBroker:
    def __init__(self, store, portfolio, tg):
        self.db = store
        self.p = portfolio
        self.tg = tg

    def manage_exits(self, market, ticks):
        results = []
        for r in self.db.open_trades().to_dict("records"):
            try:
                sym = r["symbol"]
                price = float(ticks.get(sym, {}).get("lastPrice") or 0)
                if not price:
                    continue

                entry = float(r["entry"])
                qty = float(r["qty"])
                stop = float(r["stop"])
                target = float(r["target"])
                high = max(float(r["high_watermark"] or entry), price)

                gain = (price / entry - 1) * 100
                if gain >= CFG["strategy"]["trailing_activation_pct"]:
                    trail = max(
                        stop,
                        price - CFG["strategy"]["trailing_atr_multiple"]
                        * entry * 0.01)
                    if trail > stop:
                        stop = trail
                        self.db.update_trade_stop(r["id"], stop, high)

                if price <= stop:
                    out = self.db.close_trade(r["id"], price, reason="STOP/TRAIL")
                    if out:
                        self.p.credit(price * qty)
                        results.append(out)
                elif price >= target and int(r["partial_closed"] or 0) == 0:
                    part = qty * CFG["strategy"]["partial_take_profit_pct"]
                    out = self.db.close_trade(
                        r["id"], price, qty=part, reason="PARTIAL_TP")
                    if out:
                        self.p.credit(price * part)
                        results.append(out)
                        self.db.update_trade_stop(
                            r["id"], max(stop, entry), high)
                elif price >= target * 1.05:
                    out = self.db.close_trade(r["id"], price, reason="FINAL_TP")
                    if out:
                        self.p.credit(price * qty)
                        results.append(out)
            except Exception as ex:
                log.warning("exit %s: %s", r["symbol"], ex)
        return results

    def buy(self, a):
        qty = Risk().position_size(self.p, a)
        usd = qty * a["price"]
        if qty <= 0 or usd < 5:
            return None
        if not self.p.reserve_for_buy(usd):
            return None
        tid = self.db.add_trade(dict(a, qty=qty, entry=a["price"]))
        log.info(
            "PAPER BUY %s $%.2f entry=%.8f stop=%.8f target=%.8f score=%.0f",
            a["symbol"], usd, a["price"], a["stop"], a["target"], a["score"])
        return tid


def adaptive_min_score(store):
    base = float(CFG["strategy"]["min_score"])
    n = store.closed_count()
    if not CFG["learning"]["adaptive_thresholds_enabled"] or n < CFG["learning"]["min_closed_trades_before_calibration"]:
        return base

    rows = store.db.execute(
        "SELECT pnl_usd FROM trades WHERE ts_close IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (CFG["learning"]["calibration_window"],)).fetchall()
    if not rows:
        return base

    wins = sum(1 for r in rows if float(r["pnl_usd"]) > 0)
    wr = wins / len(rows)
    if wr < 0.40:
        return min(CFG["learning"]["adaptive_threshold_ceiling"], base + 4)
    if wr > 0.60:
        return max(CFG["learning"]["adaptive_threshold_floor"], base - 2)
    return base


def run_once():
    store = Store()
    market = Market()
    news = News()
    tg = Telegram()
    portfolio = Portfolio(store)
    broker = PaperBroker(store, portfolio, tg)

    try:
        syms = market.symbols()
        ticks = market.tickers()
    except Exception as e:
        log.error("MARKET DATA UNAVAILABLE - NO TRADE: %s", e)
        tg.send("⚠️ MARKET DATA UNAVAILABLE\nNo trade was made.\nScanner will retry next run.")
        return

    ranked = []
    syms = sorted(
        [s for s in syms if s in ticks],
        key=lambda s: float(ticks[s].get("turnover24h", 0) or 0),
        reverse=True
    )[:CFG["exchange"]["universe_limit"]]

    exits = broker.manage_exits(market, ticks)
    for x in exits:
        tg.send(
            f"🔔 PAPER EXIT\n{x['symbol']}\nReason: {x['reason']}\n"
            f"PnL: ${x['pnl_usd']:.2f} ({x['pnl_pct']:.2f}%)")

    fast_cache = {}
    fast_syms = syms[:CFG["exchange"]["fast_confirm_limit"]]
    for s in fast_syms:
        try:
            fast_cache[s] = market.candles(
                s, CFG["exchange"]["fast_interval"], limit=100)
        except Exception as e:
            log.warning("5m %s: %s", s, e)

    min_score = adaptive_min_score(store)
    old_min = CFG["strategy"]["min_score"]
    CFG["strategy"]["min_score"] = min_score

    try:
        analyzer = Analyzer(market, news)
        for s in syms:
            try:
                a = analyzer.analyze(s, ticks[s], fast_cache.get(s))
                store.candidate(a)
                ranked.append(a)

                if (
                    a["decision"] == "BUY"
                    and len(store.open_trades()) < CFG["strategy"]["max_open_positions"]
                    and not portfolio.kill
                ):
                    tid = broker.buy(a)
                    if tid:
                        tg.send(
                            f"🟢 PAPER BUY\n{s}\n"
                            f"Score: {a['score']:.0f} | Risk: {a['risk']:.0f}\n"
                            f"Entry: {a['price']:.8f}\n"
                            f"Stop: {a['stop']:.8f}\n"
                            f"Target: {a['target']:.8f}\n"
                            f"5m: {a['fast_change_pct']:.2f}% | "
                            f"Vol: {a['volume_ratio']:.1f}x\n"
                            f"Reasons: {', '.join(a['reasons'])}")
            except Exception as e:
                log.warning("%s: %s", s, e)
    finally:
        CFG["strategy"]["min_score"] = old_min

    eq, pos, dd, daily = portfolio.mark({
        s: float(ticks[s].get("lastPrice") or 0) for s in ticks
    })

    ranked = sorted(
        ranked,
        key=lambda x: (x["decision"] == "BUY", x["score"]),
        reverse=True)

    print("\nTOP CANDIDATES")
    for x in ranked[:15]:
        print(
            x["symbol"], x["decision"],
            f"score={x['score']:.0f}",
            f"risk={x['risk']:.0f}",
            f"24h={x['change_pct']:.1f}%",
            f"5m={x['fast_change_pct']:.1f}%",
            f"vol={x['volume_ratio']:.1f}x"
        )

    start = CFG["challenge"]["starting_equity_usd"]
    target = CFG["challenge"]["target_equity_usd"]
    progress = (eq / start - 1) * 100

    tg.send(
        f"🤖 SCAN COMPLETE\n"
        f"Equity: ${eq:.2f}\n"
        f"Progress: {progress:+.1f}%\n"
        f"Target: ${target:.0f}\n"
        f"Open: {len(store.open_trades())}\n"
        f"Drawdown: {dd:.2f}%\n"
        f"Daily PnL: ${daily:.2f}\n"
        f"Scanned: {len(ranked)}\n"
        f"Kill switch: {'ON' if portfolio.kill else 'OFF'}\n"
        f"Min score: {min_score:.0f}"
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
