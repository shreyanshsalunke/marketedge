#!/usr/bin/env python3
"""
MarketEdge — Nightly Scanner
Feeds market_data.json consumed by marketedge.html

Scans:
  Bullish  : VCP, High Tight Flag, EP/EGU, Base Breakout, Stage 2, Pocket Pivot,
             EMA Pullback, NR7, RS New High
  Bearish  : Breakdown, Stage 4, Failed Breakout, Short EP, Distribution Top
  Choppy   : Darvas Box, 200MA Bounce, Oversold Bounce, Support Bounce, Vol Squeeze
  Long Term: New Highs+Earnings, Leader Pullback, Emerging Leader,
             Inst Accum, Industry Leader

Universe : Full US (Polygon) + TSX top-300 (yfinance .TO)
Schedule : Run nightly via cron/launchd after 4pm ET market close

Usage:
    python3 scanner_market.py           # full scan
    python3 scanner_market.py --test    # 120-stock quick test
"""
from __future__ import annotations
import os, sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CFG = {
    "min_price":       3.0,
    "min_avg_vol":     100_000,
    "top_n":           50,          # max results per scan category
    "chart_bars":      60,          # OHLCV bars sent to frontend
    "output_file":     "market_data.json",
    "batch_size":      400,
    "sector_cache":    str(Path.home() / ".marketedge_sectors.json"),
}

# Index & sector tickers
INDEX_TICKERS = {
    "SPY": ("S&P 500",     "US", "index"),
    "QQQ": ("NASDAQ 100",  "US", "index"),
    "IWM": ("Russell 2000","US", "index"),
    "DIA": ("Dow Jones",   "US", "index"),
    "VIX": ("CBOE VIX",    "US", "vix"),
    "XIU.TO": ("TSX 60",   "CA", "index"),
    "BTC-USD": ("Bitcoin", "BTC","crypto"),
    "ETH-USD": ("Ethereum","ETH","crypto"),
}

SECTOR_TICKERS = {
    "XLK":  ("Technology",     "💻"),
    "XLC":  ("Communication",  "📡"),
    "XLF":  ("Financials",     "🏦"),
    "XLV":  ("Health Care",    "🏥"),
    "XLI":  ("Industrials",    "⚙️"),
    "XLY":  ("Cons. Discret.", "🛍️"),
    "XLP":  ("Cons. Staples",  "🛒"),
    "XLE":  ("Energy",         "⚡"),
    "XLB":  ("Materials",      "🪨"),
    "XLRE": ("Real Estate",    "🏢"),
    "XLU":  ("Utilities",      "💡"),
    "XBI":  ("Biotech",        "🧬"),
}

# Canadian top-300 seeds (yfinance .TO suffix)
TSX_SEEDS = [
    "SHOP","RY","TD","BNS","BMO","CM","MFC","SLF","ENB","TRP",
    "CNQ","SU","ABX","WPM","AEM","CNR","CP","ATD","QSR","MG",
    "BCE","T","RCI-B","POW","GWO","IAG","SLF","FFH","BAM","BN",
    "FTS","H","EMA","AQN","NPI","INE","GFL","WCN","RBA","BYD",
    "CCO","LUN","FM","CS","HBM","NXE","DML","URE","GGD","CG",
    "AGI","K","OR","PG","IMG","TGZ","MAG","USA","EMX","NVO",
    "AC","WJA","CAE","MDA","BB","QTRH","PHO","KXS","TOI","DSG",
    "LSPD","DCBO","TCS","REAL","WELL","CHAL","DR","GUD","PAT","AFN",
    "ERF","PEY","MEG","ATH","BTE","TVE","FRU","CPG","GEI","SGY",
]

# ─── CONFIG LOADER ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    cfg = {}
    f = Path.home() / ".marketedge_config"
    # Also accept old SwingEdge config for API key
    for fname in [f, Path.home() / ".qscanner_config"]:
        if fname.exists():
            for line in fname.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    key = os.environ.get("POLYGON_API_KEY", cfg.get("POLYGON_API_KEY",""))
    if not key:
        print("  ✗  No POLYGON_API_KEY found.")
        print("     Add to ~/.marketedge_config: POLYGON_API_KEY=your_key")
        sys.exit(1)
    return cfg

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def sma(s, n): return s.rolling(n, min_periods=max(1, int(n*.6))).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr(d, n=14):
    h, l, c = d["High"], d["Low"], d["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def poly(endpoint, params={}, key=""):
    try:
        r = requests.get(f"https://api.polygon.io{endpoint}",
                         params={**params, "apiKey": key}, timeout=20)
        return r.json() if r.ok else {}
    except: return {}

def ohlc_chart(d: pd.DataFrame, bars: int) -> list:
    """Return last `bars` candles as [{t,o,h,l,c,v}, ...]"""
    return [
        {"t": int(ts.timestamp()),
         "o": round(float(r["Open"]),   2),
         "h": round(float(r["High"]),   2),
         "l": round(float(r["Low"]),    2),
         "c": round(float(r["Close"]),  2),
         "v": int(r["Volume"])}
        for ts, r in d.tail(bars).iterrows()
    ]

def rs_line(daily: pd.DataFrame, spy: pd.DataFrame, bars: int) -> list:
    """Normalized RS line vs SPY (1.0 = flat vs SPY on day 0)"""
    if spy is None or len(spy) == 0: return []
    try:
        tail = daily.tail(bars)
        spy_a = spy["Close"].reindex(tail.index, method="ffill").dropna()
        stock_a = tail["Close"].reindex(spy_a.index)
        raw = stock_a.values / spy_a.values
        base = raw[0]
        if base == 0: return []
        return [round(float(v/base), 4) for v in raw]
    except: return []

def rs_raw(close):
    if len(close) < 120: return 0.0
    try:
        c = close.iloc[-252:] if len(close) >= 252 else close
        n = len(c); q = max(1, n//4)
        qs = [c.iloc[max(0,i*q):min(n,(i+1)*q)] for i in range(4)]
        g = [(qs[i].iloc[-1]/qs[i].iloc[0]-1) if len(qs[i])>1 else 0 for i in range(4)]
        return g[0]*.2 + g[1]*.2 + g[2]*.2 + g[3]*.4
    except: return 0.0

def pct_rank(arr, v):
    return float(np.sum(arr <= v) / len(arr) * 100) if len(arr) else 0.0

def weeks_tight(daily: pd.DataFrame, threshold=0.015) -> int:
    try:
        weekly = daily.resample("W").agg({"Close":"last"}).dropna()
        if len(weekly) < 2: return 0
        closes = weekly["Close"].values
        count = 0
        for i in range(len(closes)-1, 0, -1):
            if closes[i-1] <= 0: break
            if abs(closes[i]-closes[i-1])/closes[i-1] <= threshold: count += 1
            else: break
        return count
    except: return 0

def basic_ok(d: pd.DataFrame) -> bool:
    if len(d) < 60: return False
    price = float(d["Close"].iloc[-1])
    vol   = float(d["Volume"].rolling(20).mean().iloc[-1])
    return price >= CFG["min_price"] and vol >= CFG["min_avg_vol"]

# ─── UNIVERSE ─────────────────────────────────────────────────────────────────
def get_us_tickers(key: str, test_mode: bool) -> list[str]:
    if test_mode:
        tickers = [
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","AMD","CRM",
            "ADBE","NOW","PANW","SNPS","CDNS","ANET","TTD","DDOG","CRWD","NET",
            "ZS","PLTR","AXON","CELH","MU","MRVL","DELL","CIEN","DOCN","APP",
            "MSTR","IONQ","RGTI","LLY","UNH","ABBV","REGN","VRTX","ISRG","DXCM",
            "JPM","GS","V","MA","COIN","SOFI","FSLR","ENPH","XOM","CVX",
            "NFLX","SPOT","RDDT","CAVA","BROS","WING","DUOL","LULU","ONON","DECK",
            "MELI","BKNG","UBER","GE","CAT","DE","SAIA","ODFL","XPO","ARM",
            "SMCI","HOOD","AFRM","BILL","SE","PYPL","SQ","INTC","WBA","CVS",
            "GLD","UNH","PG","JNJ","KO","VZ","XLE","XLRE","XLK","XLF",
            "SPY","QQQ","IWM",
        ]
        print(f"  ⚡  TEST MODE — {len(tickers)} US tickers")
        return list(dict.fromkeys(tickers))

    print("  ↓  Fetching US universe from Polygon…")
    tickers, url = [], "https://api.polygon.io/v3/reference/tickers"
    params = {"market":"stocks","type":"CS","active":"true","limit":1000}
    page = 0
    while True:
        try:
            r = requests.get(url, params={**params,"apiKey":key}, timeout=20)
            data = r.json()
            tickers.extend(t["ticker"] for t in data.get("results",[]) if t.get("ticker"))
            next_url = data.get("next_url","")
            if not next_url: break
            url, params = next_url, {}
            page += 1
            if page % 10 == 0: print(f"     … {len(tickers)} tickers")
            time.sleep(0.12)
        except Exception as e:
            print(f"     ! {e}"); break
    print(f"  ✓  {len(tickers)} US tickers from Polygon")
    return tickers

def get_ca_tickers(test_mode: bool) -> list[str]:
    seeds = [t+".TO" for t in TSX_SEEDS]
    if test_mode: return seeds[:30]
    # Expand: try fetching additional TSX tickers via yfinance screener approach
    return seeds

# ─── DOWNLOAD OHLCV ───────────────────────────────────────────────────────────
def download_ohlcv(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    batches = [tickers[i:i+CFG["batch_size"]] for i in range(0, len(tickers), CFG["batch_size"])]
    print(f"  ↓  Downloading {len(tickers)} tickers in {len(batches)} batch(es)…")
    for bi, batch in enumerate(batches):
        print(f"     Batch {bi+1}/{len(batches)}: {len(batch)} tickers…")
        try:
            raw = yf.download(batch, period="400d", interval="1d",
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty: continue
            for t in batch:
                try:
                    df = raw.xs(t, axis=1, level=1) if len(batch)>1 else raw.copy()
                    df = df.dropna()
                    if len(df) >= 60: out[t] = df
                except: pass
        except Exception as e:
            print(f"     ! Batch error: {e}")
    print(f"  ✓  Got data for {len(out)} tickers")
    return out

# ─── INDEX & SECTOR DATA ──────────────────────────────────────────────────────
def fetch_indexes(data: dict) -> list:
    results = []
    for sym, (name, flag, typ) in INDEX_TICKERS.items():
        try:
            # Try from already-downloaded data first
            key = sym
            df = data.get(key)
            if df is None or len(df) < 5:
                df = yf.download(sym, period="30d", interval="1d",
                                 auto_adjust=True, progress=False)
            if df is None or len(df) < 2: continue
            price = float(df["Close"].iloc[-1])
            prev1 = float(df["Close"].iloc[-2]) if len(df)>=2 else price
            prev5 = float(df["Close"].iloc[-6]) if len(df)>=6 else price
            prev20= float(df["Close"].iloc[-21]) if len(df)>=21 else price
            d1  = (price/prev1 - 1)*100
            d7  = (price/prev5 - 1)*100
            d30 = (price/prev20- 1)*100
            results.append({
                "sym": sym.replace("-USD","").replace(".TO",""),
                "name": name, "flag": flag, "type": typ,
                "price": round(price, 4 if price < 1 else 2),
                "d1": round(d1, 2), "d7": round(d7, 2), "d30": round(d30, 2),
            })
        except Exception as e:
            print(f"     ! Index {sym}: {e}")
    return results

def compute_rrg_position(sector_df: pd.DataFrame, spy_df: pd.DataFrame, period: int = 14) -> dict:
    """
    JdK RS-Ratio and RS-Momentum calculation.
    RS-Ratio  : 14-period EMA of (sector/SPY relative performance) normalized to 100
    RS-Momentum: 14-period EMA of RS-Ratio rate-of-change, normalized to 100
    """
    try:
        sec_close = sector_df["Close"]
        spy_close = spy_df["Close"].reindex(sec_close.index, method="ffill")
        ratio = sec_close / spy_close
        # Normalize ratio to 100-based scale
        ratio_norm = ratio / ratio.rolling(63, min_periods=20).mean() * 100
        rs_ratio = float(ema(ratio_norm, period).iloc[-1])
        # Momentum: rate of change of rs_ratio
        rs_ratio_series = ema(ratio_norm, period)
        roc = rs_ratio_series / rs_ratio_series.shift(1) * 100
        rs_momentum = float(ema(roc, period).iloc[-1])
        return {"ratio": round(rs_ratio, 2), "mom": round(rs_momentum, 2)}
    except:
        return {"ratio": 100.0, "mom": 100.0}

def fetch_sectors(data: dict, spy_df: pd.DataFrame) -> list:
    results = []
    TF_LOOKBACKS = {"d1":2,"d7":6,"d30":22,"d90":65,"d180":130,"d365":252}
    for sym, (name, emoji) in SECTOR_TICKERS.items():
        df = data.get(sym)
        if df is None or len(df) < 30:
            try:
                df = yf.download(sym, period="400d", interval="1d",
                                 auto_adjust=True, progress=False)
            except: continue
        if df is None or len(df) < 30: continue
        price = float(df["Close"].iloc[-1])
        changes = {}
        for key, lb in TF_LOOKBACKS.items():
            if len(df) >= lb+1:
                changes[key] = round((price / float(df["Close"].iloc[-lb-1]) - 1)*100, 2)
            else:
                changes[key] = 0.0
        # RRG per timeframe — use different lookback windows for normalization
        rrg = {}
        for tf, lb in [("d1",5),("d7",10),("d30",22),("d90",65),("d180",130),("d365",252)]:
            if spy_df is not None and len(df)>=lb and len(spy_df)>=lb:
                sub_sec = df.tail(max(lb*2, 60))
                sub_spy = spy_df.tail(max(lb*2, 60))
                rrg[tf] = compute_rrg_position(sub_sec, sub_spy, period=14)
            else:
                rrg[tf] = {"ratio":100.0,"mom":100.0}
        results.append({
            "sym": sym, "name": name, "emoji": emoji,
            **changes, "rrg": rrg
        })
    return results

# ─── BREADTH ──────────────────────────────────────────────────────────────────
def compute_breadth(data: dict, valid: list) -> dict:
    breadth = {"mcclellan": None, "pct_above_50sma": None, "vix": None, "fear_greed": None,
               "adv": 0, "dec": 0, "new_highs": 0, "new_lows": 0}
    # VIX
    try:
        vix_df = data.get("VIX")
        if vix_df is None or len(vix_df) < 2:
            vix_df = yf.download("^VIX", period="5d", interval="1d",
                                 auto_adjust=True, progress=False)
        if vix_df is not None and len(vix_df) > 0:
            breadth["vix"] = round(float(vix_df["Close"].iloc[-1]), 2)
    except: pass

    above50 = 0
    total   = 0
    adv_today = 0
    dec_today = 0
    new_highs = 0
    new_lows  = 0
    direction_cols = {}

    for t in valid:
        if t not in data: continue
        df = data[t]
        if len(df) < 51: continue
        total += 1
        close = df["Close"]
        # Above 50 SMA
        s50 = sma(close, 50)
        if float(close.iloc[-1]) > float(s50.iloc[-1]): above50 += 1
        # Today's advance/decline
        if len(close) >= 2:
            if float(close.iloc[-1]) > float(close.iloc[-2]): adv_today += 1
            elif float(close.iloc[-1]) < float(close.iloc[-2]): dec_today += 1
        # 52-week high/low
        high52 = float(df["High"].rolling(min(252,len(df))).max().iloc[-1])
        low52  = float(df["Low"].rolling(min(252,len(df))).min().iloc[-1])
        price  = float(close.iloc[-1])
        if price >= high52 * 0.99: new_highs += 1
        if price <= low52  * 1.01: new_lows  += 1
        # A-D direction series for McClellan
        tail = close.tail(56)
        direction = tail.diff().apply(lambda x: 1 if x>0 else (-1 if x<0 else 0)).dropna()
        direction_cols[t] = direction

    if total > 0:
        breadth["pct_above_50sma"] = round(above50/total*100, 1)
        breadth["adv"]   = adv_today
        breadth["dec"]   = dec_today
        breadth["new_highs"] = new_highs
        breadth["new_lows"]  = new_lows

    # McClellan Oscillator
    try:
        if len(direction_cols) >= 30:
            ad_df  = pd.DataFrame(direction_cols).fillna(0)
            ad_net = (ad_df==1).sum(axis=1) - (ad_df==-1).sum(axis=1)
            if len(ad_net) >= 20:
                ema19 = float(ad_net.ewm(span=19, adjust=False).mean().iloc[-1])
                ema39 = float(ad_net.ewm(span=39, adjust=False).mean().iloc[-1])
                breadth["mcclellan"] = round(ema19-ema39, 1)
    except: pass

    # Fear & Greed proxy (composite of VIX, %>50SMA, MCO)
    try:
        vix_score  = max(0, min(100, (30-(breadth["vix"] or 20))/30*100)) if breadth["vix"] else 50
        pct_score  = breadth["pct_above_50sma"] or 50
        mco_score  = max(0, min(100, 50+(breadth["mcclellan"] or 0))) if breadth["mcclellan"] else 50
        breadth["fear_greed"] = round(vix_score*.35 + pct_score*.40 + mco_score*.25, 0)
    except: pass

    return breadth

# ─── EARNINGS CALENDAR ────────────────────────────────────────────────────────
def fetch_earnings_week(tickers: list[str]) -> list:
    """Get earnings scheduled in the next 7 days for tickers in our universe."""
    results = []
    ticker_set = set(tickers)
    today = datetime.now()
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    checked = 0
    for t in list(ticker_set)[:200]:  # limit API calls
        try:
            info = yf.Ticker(t).calendar
            if info is None: continue
            if hasattr(info, 'get'):
                earn_date = info.get("Earnings Date", [])
                if isinstance(earn_date, list) and len(earn_date) > 0:
                    ed = earn_date[0]
                    if isinstance(ed, pd.Timestamp):
                        days_out = (ed.date()-today.date()).days
                        if 0 <= days_out <= 7:
                            results.append({
                                "ticker": t,
                                "day": day_map.get(ed.weekday(),"?"),
                                "est_eps": info.get("EPS Estimate","—"),
                                "sector": ""
                            })
            checked += 1
            if checked % 50 == 0: print(f"     … earnings check {checked}/200")
            time.sleep(0.1)
        except: pass
    return results[:20]  # cap at 20

# ─── FUNDAMENTALS CACHE ───────────────────────────────────────────────────────
def load_sector_cache() -> dict:
    f = Path(CFG["sector_cache"])
    if not f.exists(): return {}
    try:
        data = json.loads(f.read_text())
        age = (datetime.now()-datetime.fromisoformat(data.get("ts","2000-01-01"))).days
        if age < 7:
            print(f"  ✓  Fundamentals cache: {len(data.get('s',{}))} tickers ({age}d old)")
            return data.get("s", {})
    except: pass
    return {}

def save_sector_cache(s: dict):
    Path(CFG["sector_cache"]).write_text(
        json.dumps({"ts":datetime.now().isoformat(),"s":s}, indent=2))

def enrich_fundamentals(tickers: list[str], cache: dict) -> dict:
    needed = [t for t in tickers if t not in cache]
    if not needed: return cache
    print(f"  ↓  Fetching fundamentals for {len(needed)} scan hits…")
    out = dict(cache)
    for i, t in enumerate(needed):
        try:
            info = yf.Ticker(t).info
            eps_g  = info.get("earningsGrowth", None)
            rev_g  = info.get("revenueGrowth", None)
            out[t] = {
                "name":        info.get("shortName", t),
                "sector":      info.get("sector",""),
                "industry":    info.get("industry",""),
                "mktcap":      info.get("marketCap",0),
                "eps_growth":  round((eps_g or 0)*100, 1),
                "rev_growth":  round((rev_g or 0)*100, 1),
                "pe":          info.get("trailingPE", None),
                "inst_own":    round((info.get("heldPercentInstitutions") or 0)*100, 1),
                "short_float": round((info.get("shortPercentOfFloat") or 0)*100, 1),
            }
            time.sleep(0.3)
        except:
            out[t] = {"name":t,"sector":"","industry":"","mktcap":0,
                      "eps_growth":0,"rev_growth":0,"pe":None,"inst_own":0,"short_float":None}
        if i>0 and i%20==0:
            print(f"     … {i}/{len(needed)}")
            save_sector_cache(out)
    save_sector_cache(out)
    return out

# ─── RS RANKING ───────────────────────────────────────────────────────────────
def compute_rs_ranks(data: dict, valid: list) -> dict:
    rs_map = {t: rs_raw(data[t]["Close"]) for t in valid if t in data}
    rs_arr = np.array(list(rs_map.values()))
    return {t: round(pct_rank(rs_arr, v), 1) for t,v in rs_map.items()}

# ─── BASE RESULT ──────────────────────────────────────────────────────────────
_SPY: pd.DataFrame = pd.DataFrame()

def base_result(ticker, daily, rs_pctile, score, status, tags, extra, scan, cache) -> dict:
    si = cache.get(ticker, {})
    wt = weeks_tight(daily)
    rs_l = rs_line(daily, _SPY, CFG["chart_bars"])
    return {
        "ticker":      ticker,
        "name":        si.get("name", ticker),
        "sector":      si.get("sector",""),
        "industry":    si.get("industry",""),
        "price":       round(float(daily["Close"].iloc[-1]), 2),
        "rs_pctile":   rs_pctile,
        "rs":          rs_pctile,
        "eps_growth":  si.get("eps_growth", 0),
        "rev_growth":  si.get("rev_growth", 0),
        "eps":         si.get("eps_growth", 0),
        "rev":         si.get("rev_growth", 0),
        "inst":        si.get("inst_own", 0),
        "short_float": si.get("short_float", None),
        "sf":          si.get("short_float", None),
        "score":       score,
        "status":      status,
        "tags":        tags,
        "chart":       ohlc_chart(daily, CFG["chart_bars"]),
        "rs_line":     rs_l,
        "weeks_tight": wt,
        "wt":          wt,
        "scan":        scan,
        "scanned_at":  datetime.now(timezone.utc).isoformat(),
        **extra,
    }

def canslim_ok(ticker, rs_pctile, cache) -> bool:
    if rs_pctile < 80: return False
    si = cache.get(ticker, {})
    return (si.get("eps_growth",0) or 0) >= 20 or (si.get("rev_growth",0) or 0) >= 20

# ══════════════════════════════════════════════════════════════════════════════
#  BULLISH SCANS
# ══════════════════════════════════════════════════════════════════════════════

def scan_vcp(ticker, daily, rs_pctile, cache) -> dict | None:
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(close) < 150: return None
    s50 = float(sma(close,50).iloc[-1]); s150 = float(sma(close,150).iloc[-1])
    s200 = float(sma(close,200).iloc[-1]) if len(close)>=200 else s150*.98
    if not (price > s150 > s200): return None
    if price < s50 * 0.94: return None
    high52 = float(daily["High"].rolling(min(252,len(daily)),min_periods=50).max().iloc[-1])
    from_high = (high52-price)/high52*100
    if from_high > 20: return None
    bars = daily.tail(80)
    def range_pct(n):
        sl = bars.tail(n); mid = float(sl["Close"].mean())
        return (float(sl["High"].max())-float(sl["Low"].min()))/mid*100 if mid>0 else 99
    r10,r20,r40 = range_pct(10),range_pct(20),range_pct(40)
    if not (r10 < r20 < r40): return None
    if r10 > 12 or r40 < 15: return None
    vol10 = float(bars["Volume"].tail(10).mean()); vol40 = float(bars["Volume"].tail(40).mean())
    vol_dry = vol10/max(vol40,1)
    if vol_dry > 0.90: return None
    contractions = 2 if r20 < r40*.75 else 1
    if r10 < r20*.70: contractions += 1
    score = round(rs_pctile*.35 + max(0,100-from_high*3)*.25 +
                  max(0,100-r10*6)*.20 + max(0,(1-vol_dry)*100)*.10 +
                  min(100,contractions*33)*.10, 1)
    status = ("READY" if score>=68 and from_high<=5 and contractions>=2
              else "GOOD" if score>=55 and from_high<=12 else "DEVELOPING")
    tags = [f"{contractions} contractions", f"Range {r10:.0f}% tight"]
    if vol_dry < 0.6: tags.append("Vol drying up")
    if from_high <= 5: tags.append("Near pivot")
    return base_result(ticker,daily,rs_pctile,score,status,tags,
        {"strategy":"VCP","pivot":round(high52,2),"from_high_pct":round(from_high,1),
         "range_10d":round(r10,1),"contractions":contractions,"vol_ratio":round(vol_dry,2),
         "sma50":round(s50,2)},"VCP",cache)

def scan_high_tight_flag(ticker, daily, rs_pctile, cache) -> dict | None:
    """O'Neil High Tight Flag: 100%+ gain in 4-8 weeks, then 10-25% pullback."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    # Look for 100%+ gain in 4-8 weeks (20-40 bars)
    for pole_len in range(20, 42, 2):
        if len(daily) < pole_len + 5: continue
        pole = daily.iloc[-(pole_len+10):-10]
        if len(pole) < 15: continue
        pole_lo = float(pole["Low"].min()); pole_hi = float(pole["High"].max())
        if pole_lo <= 0: continue
        pole_gain = (pole_hi-pole_lo)/pole_lo*100
        if pole_gain < 100: continue
        # Flag: last 10 bars, 10-25% pullback from pole high, tight
        flag = daily.tail(10)
        flag_lo = float(flag["Low"].min()); flag_hi = float(flag["High"].max())
        pullback = (pole_hi-price)/pole_hi*100
        flag_range = (flag_hi-flag_lo)/flag_lo*100 if flag_lo>0 else 99
        if not (10 <= pullback <= 25 and flag_range <= 15): continue
        vol_flag = float(flag["Volume"].mean()); vol_pole = float(pole["Volume"].mean())
        vol_ratio = vol_flag/max(vol_pole,1)
        if vol_ratio > 0.7: continue
        score = round(rs_pctile*.30 + min(100,pole_gain/2)*.25 +
                      max(0,100-pullback*3)*.25 + max(0,(1-vol_ratio)*100)*.20, 1)
        status = "READY" if score>=72 else "GOOD" if score>=58 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"Pole +{pole_gain:.0f}%","Vol drying up","Near pivot"],
            {"strategy":"HTF","pole_gain_pct":round(pole_gain,1),
             "pullback_pct":round(pullback,1),"flag_range_pct":round(flag_range,1),
             "vol_ratio":round(vol_ratio,2),"pivot":round(pole_hi,2)},"HTF",cache)
    return None

def scan_episodic_pivot(ticker, daily, rs_pctile, cache) -> dict | None:
    """Qullamaggie EP/EGU: gap up 10%+ on volume 2x+, holds gap, RS leads."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 30: return None
    # Find largest gap in last 60 bars
    for i in range(len(daily)-2, max(0,len(daily)-62), -1):
        gap_pct = (float(daily["Open"].iloc[i]) / float(daily["Close"].iloc[i-1]) - 1)*100
        if gap_pct < 10: continue
        vol_ratio = float(daily["Volume"].iloc[i]) / max(float(daily["Volume"].iloc[i-10:i].mean()),1)
        if vol_ratio < 2.0: continue
        # Price must still be above gap open (holding the gap)
        gap_open = float(daily["Open"].iloc[i])
        if price < gap_open * 0.97: continue
        # Must be within 30 bars of the gap
        bars_since = len(daily) - 1 - i
        if bars_since > 30: continue
        gain_since_gap = (price/gap_open-1)*100
        score = round(rs_pctile*.35 + min(100,gap_pct*4)*.25 +
                      min(100,vol_ratio*20)*.20 + max(0,100-bars_since*3)*.20, 1)
        status = "READY" if score>=70 and bars_since<=10 else "GOOD" if score>=55 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"Gap +{gap_pct:.0f}%",f"Vol {vol_ratio:.1f}x avg","Holding gap"],
            {"strategy":"EP","gap_pct":round(gap_pct,1),"vol_ratio":round(vol_ratio,2),
             "bars_since_gap":bars_since,"gap_open":round(gap_open,2),
             "from_high_pct":round(max(0,-gain_since_gap),1)},"EP/EGU",cache)
    return None

def scan_base_breakout(ticker, daily, rs_pctile, cache) -> dict | None:
    """O'Neil/Weinstein base breakout: 6+ weeks flat, volume contraction, near pivot."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(close) < 120: return None
    s50 = float(sma(close,50).iloc[-1])
    if price < s50 * 0.97: return None
    weekly = daily.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if len(weekly) < 10: return None
    weekly["rng"] = (weekly["High"]-weekly["Low"])/weekly["Close"]*100
    weekly["s50w"] = sma(weekly["Close"],10)
    base_weeks = 0
    for i in range(len(weekly)-1, max(len(weekly)-52,0)-1, -1):
        wc = float(weekly["Close"].iloc[i]); wr = float(weekly["rng"].iloc[i])
        ws = float(weekly["s50w"].iloc[i]) if not pd.isna(weekly["s50w"].iloc[i]) else wc*.9
        if wr <= 12 and wc >= ws*.96: base_weeks += 1
        else: break
    if base_weeks < 6: return None
    base_sl = weekly.tail(base_weeks)
    base_hi = float(base_sl["High"].max()); base_lo = float(base_sl["Low"].min())
    base_mid = float(base_sl["Close"].mean())
    if base_mid <= 0: return None
    base_range = (base_hi-base_lo)/base_mid*100
    if base_range > 15: return None
    high52 = float(daily["High"].rolling(min(252,len(daily)),min_periods=50).max().iloc[-1])
    from_high = (high52-price)/high52*100
    if from_high > 7: return None
    vol_base = float(weekly["Volume"].tail(base_weeks).mean())
    vol_prior = float(weekly["Volume"].iloc[-(base_weeks+6):-base_weeks].mean()) if len(weekly)>base_weeks+6 else vol_base
    vol_ratio = vol_base/max(vol_prior,1)
    score = round(rs_pctile*.28 + max(0,100-base_range*5)*.22 +
                  min(100,base_weeks/20*100)*.18 + max(0,(1-vol_ratio)*80)*.15 +
                  max(0,100-from_high*12)*.17, 1)
    status = ("READY" if score>=68 and from_high<=3 else "GOOD" if score>=54 and from_high<=6 else "DEVELOPING")
    tags = [f"{base_weeks}w base",f"Range {base_range:.0f}%"]
    if vol_ratio < 0.65: tags.append("Vol dried up")
    if from_high <= 3: tags.append("At pivot")
    elif from_high <= 5: tags.append("Near pivot")
    return base_result(ticker,daily,rs_pctile,score,status,tags,
        {"strategy":"BASE","pivot":round(high52,2),"from_high_pct":round(from_high,1),
         "base_weeks":base_weeks,"base_range_pct":round(base_range,1),
         "vol_ratio":round(vol_ratio,2),"sma50":round(s50,2)},"Base Breakout",cache)

def scan_stage2(ticker, daily, rs_pctile, cache) -> dict | None:
    """Weinstein Stage 2: price crosses 30-week SMA upward from Stage 1 base."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 160: return None
    weekly = daily.resample("W").agg({"Close":"last","Volume":"sum"}).dropna()
    if len(weekly) < 30: return None
    w30 = sma(weekly["Close"],30)
    if pd.isna(w30.iloc[-1]) or pd.isna(w30.iloc[-5]): return None
    sma30_now = float(w30.iloc[-1]); sma30_4w = float(w30.iloc[-5])
    price_w = float(weekly["Close"].iloc[-1])
    # Stage 2 criteria: price > 30wk SMA AND 30wk SMA rising
    if price_w <= sma30_now * 1.0: return None
    if sma30_now <= sma30_4w: return None   # must be rising
    # Must have just crossed or be in early Stage 2 (within 15 weeks)
    crossed_weeks = 0
    for i in range(len(weekly)-1, max(0,len(weekly)-20), -1):
        if float(weekly["Close"].iloc[i]) > float(w30.iloc[i]):
            crossed_weeks += 1
        else: break
    if crossed_weeks > 15 or crossed_weeks < 1: return None
    # Volume should be expanding
    vol_recent = float(weekly["Volume"].tail(4).mean())
    vol_prior  = float(weekly["Volume"].iloc[-12:-4].mean()) if len(weekly)>=12 else vol_recent
    vol_ratio  = vol_recent/max(vol_prior,1)
    score = round(rs_pctile*.35 + max(0,(sma30_now/sma30_4w-1)*1000)*.25 +
                  max(0,100-crossed_weeks*5)*.20 + min(100,vol_ratio*60)*.20, 1)
    status = "READY" if score>=65 and crossed_weeks<=6 else "GOOD" if score>=52 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        [f"Stage 2 week {crossed_weeks}","30wk SMA rising",f"Vol ratio {vol_ratio:.1f}x"],
        {"strategy":"STAGE2","sma30w":round(sma30_now,2),"crossed_weeks":crossed_weeks,
         "vol_ratio":round(vol_ratio,2),"pivot":round(price*1.03,2)},"Stage 2",cache)

def scan_pocket_pivot(ticker, daily, rs_pctile, cache) -> dict | None:
    """Gil/Morales PP: up-day volume exceeds max down-day volume in prior 10 days."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    s50 = sma(close,50); s50_val = float(s50.iloc[-1])
    if price < s50_val * 0.96: return None
    # Check last 3 bars for pocket pivot
    for lookback in range(1,4):
        idx = -lookback
        if abs(idx)+10 > len(daily): continue
        bar = daily.iloc[idx]; up_day = float(bar["Close"]) >= float(bar["Open"])
        if not up_day: continue
        vol_today = float(bar["Volume"])
        # Down-day volumes in prior 10 bars
        prior = daily.iloc[idx-10:idx]
        down_vols = [float(prior["Volume"].iloc[i]) for i in range(len(prior))
                     if float(prior["Close"].iloc[i]) < float(prior["Open"].iloc[i])]
        if len(down_vols) < 3: continue
        max_down_vol = max(down_vols)
        if vol_today <= max_down_vol: continue
        # Valid pocket pivot
        vol_ratio = vol_today/max_down_vol
        score = round(rs_pctile*.35 + min(100,vol_ratio*30)*.25 +
                      min(100,(price/s50_val-1)*200)*.20 + max(0,100-lookback*20)*.20, 1)
        status = "READY" if score>=65 and lookback==1 else "GOOD" if score>=52 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"PP vol {vol_ratio:.1f}x down avg",f"Above 50 SMA","Strong accumulation"],
            {"strategy":"PP","vol_ratio":round(vol_ratio,2),"bars_ago":lookback,
             "sma50":round(s50_val,2),"pivot":round(price*1.05,2)},"Pocket Pivot",cache)
    return None

def scan_ema_pullback(ticker, daily, rs_pctile, cache) -> dict | None:
    """Classic EMA pullback: uptrending stock pulling back to 9 or 21 EMA."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    e9  = ema(close,9);  e9_val  = float(e9.iloc[-1])
    e21 = ema(close,21); e21_val = float(e21.iloc[-1])
    s50 = sma(close,50); s50_val = float(s50.iloc[-1])
    # Must be in uptrend: price > 50 SMA, 9 EMA > 21 EMA
    if not (price > s50_val * 0.98 and e9_val > e21_val): return None
    # Currently touching or just above 9 or 21 EMA
    dist_9  = (price-e9_val)/e9_val*100
    dist_21 = (price-e21_val)/e21_val*100
    if not ((-2 <= dist_9 <= 4) or (-2 <= dist_21 <= 5)): return None
    # Volume should be below average (healthy pullback)
    vol_today = float(daily["Volume"].iloc[-1])
    vol_avg   = float(daily["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = vol_today/max(vol_avg,1)
    if vol_ratio > 1.2: return None  # not a high-volume selloff
    # Prior uptrend: price higher than 4 weeks ago
    if len(close) >= 20:
        prior_price = float(close.iloc[-21])
        prior_gain = (price-prior_price)/prior_price*100
        if prior_gain < 5: return None
    score = round(rs_pctile*.35 + max(0,100-abs(dist_9)*15)*.25 +
                  max(0,(1-vol_ratio)*80)*.20 + max(0,100-(price/s50_val-1)*100)*.20, 1)
    ema_type = "9 EMA" if abs(dist_9) < abs(dist_21) else "21 EMA"
    status = "READY" if score>=65 else "GOOD" if score>=52 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        [f"Pulling back to {ema_type}","Low volume pullback","Uptrend intact"],
        {"strategy":"EMA_PULL","ema9":round(e9_val,2),"ema21":round(e21_val,2),
         "sma50":round(s50_val,2),"dist_ema_pct":round(min(abs(dist_9),abs(dist_21)),1),
         "pivot":round(float(close.rolling(20).max().iloc[-1]),2)},"EMA Pullback",cache)

def scan_nr7(ticker, daily, rs_pctile, cache) -> dict | None:
    """Connors NR7: narrowest daily range in 7 days — coiling before expansion."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    s50 = float(sma(close,50).iloc[-1])
    if price < s50 * 0.95: return None
    # NR7: today's range is smallest of last 7 days
    ranges = [(float(daily["High"].iloc[i])-float(daily["Low"].iloc[i]))
              for i in range(-7,0)]
    today_range = ranges[-1]
    if today_range > min(ranges[:-1]): return None   # not the narrowest
    # Must be in uptrend context (above 50 SMA, RS > 60)
    if rs_pctile < 55: return None
    nr7_count = sum(1 for i in range(-7,0)
                    if (float(daily["High"].iloc[i])-float(daily["Low"].iloc[i])) == today_range
                    or (float(daily["High"].iloc[i])-float(daily["Low"].iloc[i])) < ranges[-1]*1.1)
    pivot = float(daily["High"].rolling(20).max().iloc[-1])
    from_pivot = (pivot-price)/pivot*100
    score = round(rs_pctile*.35 + max(0,(1-today_range/max(ranges[:-1]))*100)*.30 +
                  max(0,100-from_pivot*8)*.20 + 15, 1)
    status = "READY" if score>=65 and from_pivot<=5 else "GOOD" if score>=52 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        ["NR7 coiling","Low volatility compression","Breakout pending"],
        {"strategy":"NR7","range_pct":round(today_range/price*100,2),
         "pivot":round(pivot,2),"from_high_pct":round(from_pivot,1)},"NR7",cache)

def scan_rs_new_high(ticker, daily, rs_pctile, cache) -> dict | None:
    """O'Neil leading indicator: RS line making new high before or with price."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60 or len(_SPY) < 60: return None
    if rs_pctile < 75: return None
    # Compute RS line
    rs_l = rs_line(daily, _SPY, min(len(daily), 252))
    if len(rs_l) < 20: return None
    rs_arr = np.array(rs_l)
    rs_current = rs_arr[-1]
    rs_52w_high = np.max(rs_arr[:-2]) if len(rs_arr)>2 else rs_current
    # RS at or very near new high
    if rs_current < rs_52w_high * 0.995: return None
    # Price need not be at new high (RS leading = the signal)
    pivot = float(daily["High"].rolling(min(252,len(daily))).max().iloc[-1])
    from_pivot = (pivot-price)/pivot*100
    rs_lead_bars = 0  # how many bars RS has been above prior high
    for i in range(len(rs_arr)-1, max(0,len(rs_arr)-20), -1):
        if rs_arr[i] >= rs_52w_high*0.99: rs_lead_bars += 1
        else: break
    score = round(rs_pctile*.40 + max(0,(rs_current/rs_52w_high-1)*1000)*.30 +
                  max(0,100-from_pivot*8)*.30, 1)
    status = "READY" if score>=68 and from_pivot<=8 else "GOOD" if score>=55 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        ["RS at new high","Leading the market","Strong relative momentum"],
        {"strategy":"RS_HIGH","rs_lead_bars":rs_lead_bars,"pivot":round(pivot,2),
         "from_high_pct":round(from_pivot,1)},"RS New High",cache)

# ══════════════════════════════════════════════════════════════════════════════
#  BEARISH SCANS
# ══════════════════════════════════════════════════════════════════════════════

def scan_breakdown(ticker, daily, rs_pctile, cache) -> dict | None:
    """Break of key support on volume — bearish continuation."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    if rs_pctile > 45: return None  # only weak stocks
    s50  = float(sma(close,50).iloc[-1])
    s200 = float(sma(close,200).iloc[-1]) if len(close)>=200 else s50*.9
    # Must be below 50 AND 200 SMA
    if price > s50 * 1.02: return None
    # Recent break: was above support, now below
    was_above = float(close.iloc[-15]) > s50*1.02 if len(close)>=15 else False
    if not was_above: return None
    # High volume on breakdown
    vol_today = float(daily["Volume"].iloc[-1])
    vol_avg   = float(daily["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = vol_today/max(vol_avg,1)
    dist_below_200 = (s200-price)/s200*100 if price < s200 else 0
    score = round((100-rs_pctile)*.35 + min(100,vol_ratio*40)*.25 +
                  min(100,dist_below_200*8)*.20 + 20, 1)
    status = "READY" if score>=65 else "GOOD" if score>=52 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        ["Break below 50 SMA","Volume expansion","Bearish"],
        {"strategy":"BREAKDOWN","sma50":round(s50,2),"sma200":round(s200,2),
         "vol_ratio":round(vol_ratio,2),"dist_below_pct":round(dist_below_200,1),
         "from_high_pct":round(dist_below_200,1)},"Breakdown",cache)

def scan_stage4(ticker, daily, rs_pctile, cache) -> dict | None:
    """Weinstein Stage 4: break below 30-week SMA into downtrend."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 160 or rs_pctile > 35: return None
    weekly = daily.resample("W").agg({"Close":"last","Volume":"sum"}).dropna()
    if len(weekly) < 30: return None
    w30 = sma(weekly["Close"],30)
    if pd.isna(w30.iloc[-1]): return None
    sma30 = float(w30.iloc[-1]); price_w = float(weekly["Close"].iloc[-1])
    if price_w > sma30 * 0.98: return None  # must be below 30wk SMA
    # SMA30 must be declining
    sma30_4w = float(w30.iloc[-5]) if len(w30)>=5 else sma30
    if sma30 >= sma30_4w: return None
    # How deep into Stage 4
    dist = (sma30-price_w)/sma30*100
    vol_recent = float(weekly["Volume"].tail(4).mean())
    vol_prior  = float(weekly["Volume"].iloc[-12:-4].mean()) if len(weekly)>=12 else vol_recent
    score = round((100-rs_pctile)*.40 + min(100,dist*5)*.30 +
                  max(0,(sma30_4w/sma30-1)*2000)*.30, 1)
    status = "READY" if score>=65 else "GOOD" if score>=52 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        ["Stage 4 breakdown","30wk SMA declining","Institutional distribution"],
        {"strategy":"STAGE4","sma30w":round(sma30,2),"dist_pct":round(dist,1),
         "from_high_pct":round(dist,1)},"Stage 4",cache)

def scan_failed_breakout(ticker, daily, rs_pctile, cache) -> dict | None:
    """Stock breaks out above resistance, reverses within 1-3 weeks — high-prob short."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60 or rs_pctile > 50: return None
    # Find a recent local high (potential failed breakout) in last 5-15 bars
    for lookback in range(5, 16):
        if lookback >= len(daily): continue
        peak_bar = daily.iloc[-lookback]
        peak_price = float(peak_bar["High"])
        # Was this bar a breakout? It should exceed prior 20-bar resistance
        prior_high = float(daily["High"].iloc[-lookback-20:-lookback].max()) if lookback+20 < len(daily) else peak_price
        if peak_price < prior_high * 1.01: continue  # wasn't a breakout
        # Now below the prior resistance (failed)
        if price > prior_high * 0.98: continue
        # Reversal should have volume
        vol_peak = float(peak_bar["Volume"])
        vol_avg  = float(daily["Volume"].rolling(20).mean().iloc[-1])
        if vol_peak < vol_avg * 1.3: continue
        reversal_pct = (peak_price-price)/peak_price*100
        score = round((100-rs_pctile)*.35 + min(100,reversal_pct*6)*.30 +
                      min(100,vol_peak/max(vol_avg,1)*20)*.20 + 15, 1)
        status = "READY" if score>=65 and lookback<=8 else "GOOD" if score>=52 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"Failed breakout -{reversal_pct:.0f}%","Reversed on volume","Bearish"],
            {"strategy":"FAILED_BO","peak_price":round(peak_price,2),
             "reversal_pct":round(reversal_pct,1),"bars_since":lookback,
             "from_high_pct":round(reversal_pct,1)},"Failed Breakout",cache)
    return None

def scan_short_ep(ticker, daily, rs_pctile, cache) -> dict | None:
    """Gap down on bad catalyst, holds gap, continues lower."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 30 or rs_pctile > 40: return None
    for i in range(len(daily)-2, max(0,len(daily)-32), -1):
        gap_pct = (float(daily["Open"].iloc[i]) / float(daily["Close"].iloc[i-1]) - 1)*100
        if gap_pct > -8: continue  # must be -8% or worse
        vol_ratio = float(daily["Volume"].iloc[i]) / max(float(daily["Volume"].iloc[i-10:i].mean()),1)
        if vol_ratio < 1.8: continue
        # Must still be below the gap open (holding the gap down)
        gap_open = float(daily["Open"].iloc[i])
        if price > gap_open * 1.03: continue
        bars_since = len(daily)-1-i
        if bars_since > 25: continue
        score = round((100-rs_pctile)*.35 + min(100,-gap_pct*5)*.25 +
                      min(100,vol_ratio*20)*.20 + max(0,100-bars_since*4)*.20, 1)
        status = "READY" if score>=65 and bars_since<=10 else "GOOD" if score>=52 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"Gap down {gap_pct:.0f}%",f"Vol {vol_ratio:.1f}x","Holding gap down"],
            {"strategy":"SHORT_EP","gap_pct":round(gap_pct,1),"vol_ratio":round(vol_ratio,2),
             "gap_open":round(gap_open,2),"from_high_pct":round((gap_open-price)/gap_open*100+5,1)},"Short EP",cache)
    return None

def scan_distribution_top(ticker, daily, rs_pctile, cache) -> dict | None:
    """O'Neil: 4+ distribution days in 4-5 weeks after 25%+ run = institutional exit."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 80 or rs_pctile > 45: return None
    # Prior run: must have gained 25%+ in last 6 months
    prior_low = float(daily["Low"].iloc[-130:-20].min()) if len(daily)>=130 else float(daily["Low"].min())
    peak = float(daily["High"].rolling(60).max().iloc[-1])
    prior_run = (peak-prior_low)/prior_low*100 if prior_low>0 else 0
    if prior_run < 25: return None
    # Price near or below peak (distribution happening)
    from_peak = (peak-price)/peak*100
    if from_peak > 20 or from_peak < 2: return None
    # Count distribution days in last 25 bars (O'Neil: up to close but high volume down)
    dist_days = 0
    vol_avg = float(daily["Volume"].rolling(50).mean().iloc[-1])
    for i in range(-25, 0):
        if abs(i) > len(daily): continue
        d_close = float(daily["Close"].iloc[i])
        d_open  = float(daily["Open"].iloc[i])
        d_vol   = float(daily["Volume"].iloc[i])
        if d_close < d_open and d_vol > vol_avg * 1.1:  # down day on above-avg volume
            dist_days += 1
    if dist_days < 4: return None
    score = round((100-rs_pctile)*.30 + min(100,dist_days*10)*.30 +
                  min(100,from_peak*8)*.20 + min(100,prior_run/50*100)*.20, 1)
    status = "READY" if score>=65 and dist_days>=5 else "GOOD" if score>=52 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        [f"{dist_days} distribution days","Institutional selling","Top forming"],
        {"strategy":"DIST_TOP","dist_days":dist_days,"prior_run_pct":round(prior_run,1),
         "from_peak_pct":round(from_peak,1),"from_high_pct":round(from_peak,1)},"Distribution Top",cache)

# ══════════════════════════════════════════════════════════════════════════════
#  CHOPPY SCANS
# ══════════════════════════════════════════════════════════════════════════════

def scan_darvas_box(ticker, daily, rs_pctile, cache) -> dict | None:
    """Darvas Box: defined price range, coiling for breakout."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 40: return None
    # Find box: last 15-35 bars where high and low are well defined
    for box_len in range(15, 36, 5):
        if box_len >= len(daily): continue
        box = daily.tail(box_len)
        box_hi = float(box["High"].max()); box_lo = float(box["Low"].min())
        box_mid = float(box["Close"].mean())
        if box_mid <= 0: continue
        box_range = (box_hi-box_lo)/box_mid*100
        if box_range > 12: continue  # must be tight
        if box_range < 3:  continue  # must have some range
        # Price must be in upper half of box (coiling near top)
        midpoint = (box_hi+box_lo)/2
        if price < midpoint: continue
        # Volume declining inside box
        vol_box   = float(box["Volume"].mean())
        vol_prior = float(daily["Volume"].iloc[-(box_len+10):-box_len].mean()) if box_len+10 < len(daily) else vol_box
        vol_ratio = vol_box/max(vol_prior,1)
        score = round(rs_pctile*.30 + max(0,100-box_range*5)*.25 +
                      max(0,(price-midpoint)/(box_hi-midpoint)*50 if box_hi>midpoint else 0)*.25 +
                      max(0,(1-vol_ratio)*80)*.20, 1)
        status = "READY" if score>=62 else "GOOD" if score>=50 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"Darvas box {box_range:.1f}% wide","Near box top","Vol contracting"],
            {"strategy":"DARVAS","box_top":round(box_hi,2),"box_bot":round(box_lo,2),
             "box_range_pct":round(box_range,1),"vol_ratio":round(vol_ratio,2),
             "pivot":round(box_hi,2),"from_high_pct":round((box_hi-price)/box_hi*100,1)},"Darvas Box",cache)
    return None

def scan_200ma_bounce(ticker, daily, rs_pctile, cache) -> dict | None:
    """Price tests 200 SMA from above, reversal candle with volume confirmation."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 220: return None
    s200 = sma(close,200); s200_val = float(s200.iloc[-1])
    # Price must be at or just above 200 SMA (within 5%)
    dist = (price-s200_val)/s200_val*100
    if not (-2 <= dist <= 5): return None
    # 200 SMA must be flat or rising (uptrend)
    s200_4w = float(s200.iloc[-21]) if len(s200)>=21 else s200_val
    if s200_val < s200_4w * 0.995: return None
    # Recent bar should be a reversal (close > open, close near high)
    recent = daily.tail(3)
    has_reversal = False
    for i in range(len(recent)):
        c, o = float(recent["Close"].iloc[i]), float(recent["Open"].iloc[i])
        h, l = float(recent["High"].iloc[i]),  float(recent["Low"].iloc[i])
        body_pct = abs(c-o)/(h-l) if (h-l)>0 else 0
        if c > o and body_pct > 0.5: has_reversal = True
    if not has_reversal: return None
    vol_today = float(daily["Volume"].iloc[-1]); vol_avg = float(daily["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = vol_today/max(vol_avg,1)
    score = round(rs_pctile*.30 + max(0,(5-abs(dist))*15)*.25 +
                  min(100,vol_ratio*50)*.25 + (20 if s200_val>s200_4w else 5)*.20, 1)
    status = "READY" if score>=62 else "GOOD" if score>=50 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        ["200 SMA support test","Reversal candle","Vol confirmation"],
        {"strategy":"200_BOUNCE","sma200":round(s200_val,2),"dist_pct":round(dist,1),
         "vol_ratio":round(vol_ratio,2),"pivot":round(price*1.06,2),
         "from_high_pct":round(dist,1)},"200MA Bounce",cache)

def scan_oversold_bounce(ticker, daily, rs_pctile, cache) -> dict | None:
    """RSI < 30 + price >12% below 200 SMA + reversal candle (not Stage 4)."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    # RSI calculation
    delta = close.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    rs_rsi = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, 1e-10)
    rsi = float((100 - 100/(1+rs_rsi)).iloc[-1])
    if rsi > 32: return None  # must be oversold
    # Price well below 200 SMA
    if len(close) >= 200:
        s200_val = float(sma(close,200).iloc[-1])
        dist_below = (s200_val-price)/s200_val*100
        if dist_below < 12: return None
        # NOT in Stage 4 (200 SMA must not be sharply declining)
        s200_old = float(sma(close,200).iloc[-42]) if len(close)>=242 else s200_val
        if s200_val < s200_old * 0.90: return None  # too steep a decline
    else: return None
    # Reversal candle (hammer or bullish engulf)
    c = float(daily["Close"].iloc[-1]); o = float(daily["Open"].iloc[-1])
    h = float(daily["High"].iloc[-1]);  l = float(daily["Low"].iloc[-1])
    lower_shadow = o-l if c>o else c-l
    if c <= o or lower_shadow < abs(c-o): return None  # needs hammer
    vol_today = float(daily["Volume"].iloc[-1]); vol_avg = float(daily["Volume"].rolling(20).mean().iloc[-1])
    score = round((100-rs_pctile)*.20 + max(0,(32-rsi)*5)*.30 +
                  max(0,dist_below*4)*.25 + min(100,vol_today/max(vol_avg,1)*40)*.25, 1)
    status = "READY" if score>=62 else "GOOD" if score>=50 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        [f"RSI {rsi:.0f} — oversold",f"-{dist_below:.0f}% from 200 SMA","Hammer reversal"],
        {"strategy":"OB_BOUNCE","rsi":round(rsi,1),"dist_below_200":round(dist_below,1),
         "sma200":round(s200_val,2),"pivot":round(price*1.08,2),
         "from_high_pct":round(dist_below,1)},"Oversold Bounce",cache)

def scan_support_bounce(ticker, daily, rs_pctile, cache) -> dict | None:
    """Price holds key horizontal support with volume confirmation."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    # Find horizontal support: a level that's been tested 2+ times in last 60 bars
    lows = daily["Low"].tail(60).values
    for supp_idx in range(5, 55):
        supp_price = float(lows[supp_idx])
        # Count how many bars touched this level (within 1.5%)
        touches = sum(1 for l in lows if abs(l-supp_price)/supp_price <= 0.015)
        if touches < 2: continue
        # Current price near support (within 3%)
        dist = (price-supp_price)/supp_price*100
        if not (-1 <= dist <= 3): continue
        # Volume on today's reversal
        vol_today = float(daily["Volume"].iloc[-1]); vol_avg = float(daily["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = vol_today/max(vol_avg,1)
        if vol_ratio < 0.8: continue
        score = round(rs_pctile*.28 + min(100,touches*25)*.25 +
                      max(0,(3-dist)*20)*.25 + min(100,vol_ratio*50)*.22, 1)
        status = "READY" if score>=60 and touches>=3 else "GOOD" if score>=48 else "DEVELOPING"
        return base_result(ticker,daily,rs_pctile,score,status,
            [f"Support tested {touches}x","Holding key level",f"Vol {vol_ratio:.1f}x avg"],
            {"strategy":"SUPPORT","support_level":round(supp_price,2),"touches":touches,
             "dist_pct":round(dist,1),"vol_ratio":round(vol_ratio,2),
             "pivot":round(price*1.06,2),"from_high_pct":round(dist,1)},"Support Bounce",cache)
    return None

def scan_vol_squeeze(ticker, daily, rs_pctile, cache) -> dict | None:
    """John Carter: Bollinger Band squeeze at 6-month low + ATR compression."""
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 60: return None
    # Bollinger Band width (20-period, 2 std)
    roll_mean = close.rolling(20).mean(); roll_std  = close.rolling(20).std()
    bb_width  = (roll_std * 4) / roll_mean * 100   # band width as % of price
    current_bbw = float(bb_width.iloc[-1])
    min_bbw_6m  = float(bb_width.tail(120).min()) if len(bb_width)>=120 else current_bbw
    # Must be at or near 6-month low BB width
    if current_bbw > min_bbw_6m * 1.15: return None
    # ATR also compressed
    atr_now  = float(atr(daily, 14).iloc[-1])
    atr_6m   = float(atr(daily, 14).tail(120).mean()) if len(daily)>=120 else atr_now
    atr_ratio= atr_now/max(atr_6m,1)
    if atr_ratio > 0.75: return None  # must be well compressed
    # Price above 50 SMA preferred (bullish squeeze)
    s50_val = float(sma(close,50).iloc[-1])
    above_50 = price > s50_val
    score = round(rs_pctile*.30 + max(0,(1-current_bbw/max(min_bbw_6m,1))*50)*.25 +
                  max(0,(1-atr_ratio)*100)*.25 + (20 if above_50 else 5)*.20, 1)
    status = "READY" if score>=60 else "GOOD" if score>=48 else "DEVELOPING"
    return base_result(ticker,daily,rs_pctile,score,status,
        [f"BB width at 6-month low","ATR compressed","Breakout imminent"],
        {"strategy":"VOL_SQUEEZE","bb_width_pct":round(current_bbw,2),"atr_ratio":round(atr_ratio,2),
         "above_50sma":above_50,"pivot":round(price*1.05,2),
         "from_high_pct":round(max(0,(float(close.rolling(20).max().iloc[-1])-price)/price*100),1)},"Vol Squeeze",cache)

# ══════════════════════════════════════════════════════════════════════════════
#  LONG TERM (CANSLIM)
# ══════════════════════════════════════════════════════════════════════════════

def scan_canslim(ticker, daily, rs_pctile, cache) -> list[dict]:
    """Run all 5 CANSLIM sub-scans, return list of matches (0 or 1 per sub-scan)."""
    if not basic_ok(daily): return []
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 100: return []
    si = cache.get(ticker, {})
    eps_g = si.get("eps_growth", 0) or 0
    rev_g = si.get("rev_growth", 0) or 0
    inst  = si.get("inst_own", 0) or 0
    results = []

    # 1. New Highs + Earnings (C+A+N criteria)
    if eps_g >= 20 and rs_pctile >= 80:
        high52 = float(daily["High"].rolling(min(252,len(daily))).max().iloc[-1])
        from_high = (high52-price)/high52*100
        if from_high <= 8:
            score = round(rs_pctile*.30 + min(100,eps_g/2)*.25 + min(100,rev_g/2)*.20 +
                          max(0,100-from_high*8)*.25, 1)
            results.append(base_result(ticker,daily,rs_pctile,score,
                "READY" if score>=75 else "GOOD",
                [f"EPS +{eps_g:.0f}%",f"RS {rs_pctile:.0f}th","Near 52w high"],
                {"strategy":"NH_EARNINGS","canslim":score,"pattern":"New High",
                 "from_high_pct":round(from_high,1),"inst":inst},"CANSLIM",cache))

    # 2. Market Leader Pullback (L+N criteria — first pullback to 50d)
    if rs_pctile >= 75 and eps_g >= 15:
        s50 = sma(close,50); s50_val = float(s50.iloc[-1])
        dist_50 = (price-s50_val)/s50_val*100
        # First pullback: was well above 50 SMA 4 weeks ago
        s50_4w = float(s50.iloc[-21]) if len(s50)>=21 else s50_val
        was_above = float(close.iloc[-21]) > s50_4w*1.05 if len(close)>=21 else False
        if was_above and -2 <= dist_50 <= 5:
            score = round(rs_pctile*.35 + max(0,(5-abs(dist_50))*12)*.25 +
                          min(100,eps_g/2)*.20 + min(100,inst)*.20, 1)
            results.append(base_result(ticker,daily,rs_pctile,score,
                "READY" if score>=72 else "GOOD",
                [f"First 50d pullback",f"RS {rs_pctile:.0f}th",f"EPS +{eps_g:.0f}%"],
                {"strategy":"LEADER_PULLBACK","canslim":score,"pattern":"50d SMA Test",
                 "from_high_pct":round(dist_50,1),"inst":inst},"CANSLIM",cache))

    # 3. Emerging Leader (small/mid cap + accelerating EPS + early Stage 2)
    mktcap = si.get("mktcap",0) or 0
    if eps_g >= 25 and rs_pctile >= 70 and mktcap < 10e9:
        s200 = float(sma(close,200).iloc[-1]) if len(close)>=200 else price*.9
        if price > s200:
            score = round(rs_pctile*.30 + min(100,eps_g/1.5)*.30 +
                          min(100,rev_g/1.5)*.20 + (20 if mktcap<2e9 else 10)*.20, 1)
            results.append(base_result(ticker,daily,rs_pctile,score,
                "READY" if score>=70 else "GOOD",
                [f"Small cap leader",f"EPS +{eps_g:.0f}%",f"Stage 2 uptrend"],
                {"strategy":"EMERGING","canslim":score,"pattern":"Emerging Stage 2",
                 "from_high_pct":5.0,"inst":inst},"CANSLIM",cache))

    # 4. Institutional Accumulation (I criteria — rising fund ownership)
    if inst >= 50 and rs_pctile >= 65 and eps_g >= 10:
        score = round(rs_pctile*.30 + min(100,inst)*.30 +
                      min(100,eps_g/2)*.20 + min(100,rev_g/2)*.20, 1)
        if score >= 60:
            results.append(base_result(ticker,daily,rs_pctile,score,
                "GOOD" if score>=68 else "DEVELOPING",
                [f"Inst own {inst:.0f}%",f"EPS +{eps_g:.0f}%","Big money accumulating"],
                {"strategy":"INST_ACCUM","canslim":score,"pattern":"Accumulation Base",
                 "from_high_pct":8.0,"inst":inst},"CANSLIM",cache))

    # 5. Industry Group Leader (#1 or #2 by RS in their group)
    if rs_pctile >= 85 and eps_g >= 15:
        score = round(rs_pctile*.40 + min(100,eps_g/1.5)*.30 + min(100,rev_g/2)*.30, 1)
        results.append(base_result(ticker,daily,rs_pctile,score,
            "READY" if score>=78 else "GOOD",
            [f"RS {rs_pctile:.0f}th percentile",f"EPS +{eps_g:.0f}%","Sector leadership"],
            {"strategy":"INDUSTRY_LEADER","canslim":score,"pattern":"Market Leader",
             "from_high_pct":5.0,"inst":inst},"CANSLIM",cache))

    return results

# ══════════════════════════════════════════════════════════════════════════════
#  GITHUB UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def upload_to_github(data: dict, cfg: dict):
    token = cfg.get("GITHUB_TOKEN","")
    repo  = cfg.get("GITHUB_REPO","")
    if not token or not repo:
        print("  !  No GITHUB_TOKEN/GITHUB_REPO — skipping upload")
        print("     Add to ~/.marketedge_config")
        return
    try:
        from github import Github
        print(f"  ↑  Uploading market_data.json → {repo}…")
        g = Github(token); gh_repo = g.get_repo(repo)
        content_str = json.dumps(data, indent=2)
        try:
            existing = gh_repo.get_contents("market_data.json", ref="main")
            gh_repo.update_file("market_data.json",
                f"MarketEdge scan — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content_str, existing.sha, branch="main")
        except:
            gh_repo.create_file("market_data.json",
                f"MarketEdge scan — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content_str, branch="main")
        site = repo.split("/")[1] if "/" in repo else repo
        print(f"  ✓  Live → https://{site}.vercel.app\n")
    except Exception as e:
        print(f"  !  GitHub upload failed: {e}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════==========
def run(test_mode=False):
    print("\n"+"═"*60)
    print("  MarketEdge — Nightly Scanner")
    print("  US + Canadian Markets | All Strategies")
    print("═"*60)
    cfg = load_config()
    KEY = cfg.get("POLYGON_API_KEY","")
    us_tickers  = get_us_tickers(KEY, test_mode)
    ca_tickers  = get_ca_tickers(test_mode)
    all_tickers = list(dict.fromkeys(us_tickers + ca_tickers +
                   list(SECTOR_TICKERS.keys()) + ["SPY","QQQ","VIX"]))
    data = download_ohlcv(all_tickers)
    global _SPY
    _SPY = data.get("SPY", pd.DataFrame())
    spy_df = _SPY
    print("  ∑  Building index & sector data…")
    indexes = fetch_indexes(data)
    sectors = fetch_sectors(data, spy_df)
    regime_spy = data.get("SPY", pd.DataFrame())
    if len(regime_spy) >= 25:
        s10 = float(sma(regime_spy["Close"],10).iloc[-1])
        s20 = float(sma(regime_spy["Close"],20).iloc[-1])
        regime = "BULLISH" if s10>s20*1.002 else "BEARISH" if s10<s20*.998 else "NEUTRAL"
    else:
        regime = "NEUTRAL"
    print("  ∑  Filtering universe…")
    valid = [t for t in all_tickers if t in data and basic_ok(data[t])]
    print(f"  ✓  {len(valid)} stocks passed price/volume filter")
    print("  ∑  Computing RS ranks…")
    rs_pct = compute_rs_ranks(data, valid)
    print(f"  📊  Market regime: {regime}")
    print("  ∑  Computing market breadth…")
    breadth = compute_breadth(data, valid)
    print(f"     MCO: {breadth.get('mcclellan','—')}  "
          f"%>50SMA: {breadth.get('pct_above_50sma','—')}%  VIX: {breadth.get('vix','—')}")
    cache = load_sector_cache()
    BULLISH_SCANS = [
        ("VCP",scan_vcp),("HTF",scan_high_tight_flag),("EP",scan_episodic_pivot),
        ("BASE",scan_base_breakout),("STAGE2",scan_stage2),("PP",scan_pocket_pivot),
        ("EMA_PULL",scan_ema_pullback),("NR7",scan_nr7),("RS_HIGH",scan_rs_new_high),
    ]
    BEARISH_SCANS = [
        ("BREAKDOWN",scan_breakdown),("STAGE4",scan_stage4),("FAILED_BO",scan_failed_breakout),
        ("SHORT_EP",scan_short_ep),("DIST_TOP",scan_distribution_top),
    ]
    CHOPPY_SCANS = [
        ("DARVAS",scan_darvas_box),("200_BOUNCE",scan_200ma_bounce),
        ("OB_BOUNCE",scan_oversold_bounce),("SUPPORT",scan_support_bounce),
        ("VOL_SQUEEZE",scan_vol_squeeze),
    ]
    def run_scans(scan_list):
        hits = []
        for name, fn in scan_list:
            count = 0
            for t in valid:
                try:
                    r = fn(t, data[t], rs_pct.get(t,0), cache)
                    if r: hits.append(r); count += 1
                except: pass
            print(f"     {name:<16} {count} hits")
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:CFG["top_n"]]
    print(f"\n  🔍  Bullish scans ({len(valid)} stocks)…")
    swing_bull = run_scans(BULLISH_SCANS)
    print(f"\n  🔍  Bearish scans…")
    swing_bear = run_scans(BEARISH_SCANS)
    print(f"\n  🔍  Choppy market scans…")
    swing_chop = run_scans(CHOPPY_SCANS)
    print(f"\n  🔍  CANSLIM long term scans…")
    lt_hits = []
    for t in valid:
        try:
            lt_hits.extend(scan_canslim(t, data[t], rs_pct.get(t,0), cache))
        except: pass
    lt_hits.sort(key=lambda x: x["score"], reverse=True)
    lt_hits = lt_hits[:CFG["top_n"]]
    print(f"     CANSLIM          {len(lt_hits)} hits")
    all_hits = swing_bull + swing_bear + swing_chop + lt_hits
    hit_tickers = list({r["ticker"] for r in all_hits})
    if hit_tickers:
        print(f"\n  ↓  Fetching fundamentals for {len(hit_tickers)} hits…")
        cache = enrich_fundamentals(hit_tickers, cache)
        for r in all_hits:
            si = cache.get(r["ticker"], {})
            r["name"]        = si.get("name", r.get("name", r["ticker"]))
            r["sector"]      = si.get("sector", r.get("sector",""))
            r["industry"]    = si.get("industry","")
            r["eps_growth"]  = si.get("eps_growth",0)
            r["rev_growth"]  = si.get("rev_growth",0)
            r["eps"]         = si.get("eps_growth",0)
            r["rev"]         = si.get("rev_growth",0)
            r["inst"]        = si.get("inst_own",0)
            r["short_float"] = si.get("short_float",None)
            r["sf"]          = si.get("short_float",None)
            r["canslim"]     = canslim_ok(r["ticker"], r["rs_pctile"], cache)
    print("  ∑  Fetching earnings calendar…")
    earnings = fetch_earnings_week(hit_tickers[:100])
    print(f"     {len(earnings)} earnings this week")
    output = {
        "scanned_at":    datetime.now(timezone.utc).isoformat(),
        "market_regime": regime,
        "universe":      len(valid),
        "breadth":       breadth,
        "indexes":       indexes,
        "sectors":       sectors,
        "earnings":      earnings,
        "swing":         swing_bull,
        "bearish":       swing_bear,
        "choppy":        swing_chop,
        "longterm":      lt_hits,
    }
    total = len(swing_bull)+len(swing_bear)+len(swing_chop)+len(lt_hits)
    with open(CFG["output_file"], "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓  {total} setups → {CFG['output_file']}")
    print(f"     Bullish:{len(swing_bull)}  Bearish:{len(swing_bear)}  "
          f"Choppy:{len(swing_chop)}  LongTerm:{len(lt_hits)}")
    upload_to_github(output, cfg)

def setup_cron():
    """Install launchd plist on macOS — runs every weekday at 9:30 PM UTC (4:30 PM ET)."""
    script_path = Path(__file__).resolve()
    python_path = sys.executable
    log_path    = Path.home() / "Library/Logs/marketedge_scan.log"
    plist_path  = Path.home() / "Library/LaunchAgents/com.marketedge.scanner.plist"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.marketedge.scanner</string>
  <key>ProgramArguments</key><array>
    <string>{python_path}</string>
    <string>{script_path}</string>
  </array>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>21</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>StandardOutPath</key><string>{log_path}</string>
  <key>StandardErrorPath</key><string>{log_path}</string>
  <key>RunAtLoad</key><false/>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict></plist>"""
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist)
    print(f"\n  ✓  Cron plist written → {plist_path}")
    print(f"     Runs every weekday at 9:30 PM UTC (4:30 PM ET after market close)")
    print(f"     Logs → {log_path}")
    print(f"\n  To activate:  launchctl load {plist_path}")
    print(f"  To check:     launchctl list | grep marketedge")
    print(f"  To disable:   launchctl unload {plist_path}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--setup-cron" in args:
        setup_cron()
    else:
        run(test_mode="--test" in args)
