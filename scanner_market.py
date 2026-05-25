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
    "chart_bars":      180,         # OHLCV bars sent to frontend (180 daily = ~6M)
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
    # FMP key is optional but enables quarterly EPS/Rev data
    fmp_key = os.environ.get("FMP_API_KEY", cfg.get("FMP_API_KEY",""))
    if fmp_key:
        print("  ✓  FMP API key found — quarterly fundamentals enabled")
    else:
        print("  ⚠  No FMP_API_KEY — using TTM fundamentals only")
        print("     Get a free key at financialmodelingprep.com and add:")
        print("     FMP_API_KEY=your_key to ~/.marketedge_config")
    cfg["FMP_API_KEY"] = fmp_key
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

def weekly_ohlc_chart(d: pd.DataFrame, bars: int = 104) -> list:
    """Return last `bars` weekly candles — 104 weeks = 2 years"""
    try:
        weekly = d.resample("W").agg({
            "Open": "first", "High": "max",
            "Low": "min",  "Close": "last", "Volume": "sum"
        }).dropna()
        return [
            {"t": int(ts.timestamp()),
             "o": round(float(r["Open"]),  2),
             "h": round(float(r["High"]),  2),
             "l": round(float(r["Low"]),   2),
             "c": round(float(r["Close"]), 2),
             "v": int(r["Volume"])}
            for ts, r in weekly.tail(bars).iterrows()
        ]
    except:
        return []

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

# ─── FMP COMPREHENSIVE FUNDAMENTALS ──────────────────────────────────────────
def fetch_fmp_full(ticker: str, fmp_key: str) -> dict:
    """
    Fetch quarterly + annual fundamentals + shares float from FMP.

    Covers:
      C — MRQ EPS vs same Q prior year, 3-quarter acceleration chain
      A — Annual EPS growing 3 consecutive years, annual EPS stability
      S — Float (shares outstanding), share buybacks trend
      Profitability — MRQ EPS > 0
    """
    result = {
        # Quarterly (C criteria)
        "mrq_eps_growth":    None,
        "mrq_rev_growth":    None,
        "eps_accelerating":  False,
        "rev_accelerating":  False,
        "eps_accel_3q":      False,   # 3 consecutive quarters of acceleration
        "rev_accel_3q":      False,
        "eps_positive":      False,
        "mrq_eps":           None,
        "mrq_rev":           None,
        # Annual (A criteria)
        "annual_eps_growth_3y": None,   # 3-year compound EPS growth %
        "annual_eps_stable":    False,  # EPS grew all 3 prior years
        "annual_rev_growth_3y": None,
        # Supply (S criteria)
        "shares_outstanding": None,     # millions
        "float_category":     None,     # "small" <50M, "mid" <200M, "large" >200M
        "buyback_trend":      False,    # shares declining YoY (company buying back)
        # Meta
        "q_data_available":  False,
        "a_data_available":  False,
    }
    if not fmp_key:
        return result

    def safe_eps(q): return q.get("epsdiluted") or q.get("eps") or 0.0
    def safe_rev(q): return q.get("revenue") or 0.0
    def safe_shares(q): return q.get("weightedAverageShsOutDil") or q.get("weightedAverageShsOut") or 0

    # ── QUARTERLY DATA (C criteria) ───────────────────────────────────────────
    try:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
        r = requests.get(url, params={"period":"quarter","limit":13,"apikey":fmp_key}, timeout=15)
        if r.ok:
            qdata = r.json()
            if isinstance(qdata, list) and len(qdata) >= 5:
                # Q0=MRQ, Q1=prior, Q2=2 quarters ago, Q3=3 quarters ago
                # Q4=same Q last year, Q5=same Q as Q1 last year, etc.
                eps = [safe_eps(q) for q in qdata]
                rev = [safe_rev(q) for q in qdata]
                shr = [safe_shares(q) for q in qdata]

                result["mrq_eps"]      = round(eps[0], 4)
                result["eps_positive"] = eps[0] > 0
                result["mrq_rev"]      = rev[0]

                # MRQ growth vs same quarter 1 year ago
                if len(eps) > 4 and eps[4] != 0:
                    result["mrq_eps_growth"] = round((eps[0]-eps[4])/abs(eps[4])*100, 1)
                if len(rev) > 4 and rev[4] > 0:
                    result["mrq_rev_growth"] = round((rev[0]-rev[4])/rev[4]*100, 1)

                # Acceleration: compute YoY growth for last 3 quarters
                eps_growths, rev_growths = [], []
                for i in range(3):         # Q0, Q1, Q2
                    ya = i + 4             # same quarter 1 year ago
                    if len(eps) > ya and eps[ya] != 0:
                        eps_growths.append((eps[i]-eps[ya])/abs(eps[ya])*100)
                    else:
                        eps_growths.append(None)
                    if len(rev) > ya and rev[ya] > 0:
                        rev_growths.append((rev[i]-rev[ya])/rev[ya]*100)
                    else:
                        rev_growths.append(None)

                # One-quarter acceleration (Q0 > Q1)
                if eps_growths[0] is not None and eps_growths[1] is not None:
                    result["eps_accelerating"] = eps_growths[0] > eps_growths[1]
                if rev_growths[0] is not None and rev_growths[1] is not None:
                    result["rev_accelerating"] = rev_growths[0] > rev_growths[1]

                # Three-quarter acceleration chain (Q0 > Q1 > Q2)
                if all(g is not None for g in eps_growths):
                    result["eps_accel_3q"] = eps_growths[0] > eps_growths[1] > eps_growths[2]
                if all(g is not None for g in rev_growths):
                    result["rev_accel_3q"] = rev_growths[0] > rev_growths[1] > rev_growths[2]

                # Float / buyback trend (S criteria)
                if shr[0] > 0:
                    result["shares_outstanding"] = round(shr[0] / 1e6, 1)  # millions
                    sh_m = shr[0] / 1e6
                    result["float_category"] = ("small" if sh_m < 50
                                                else "mid" if sh_m < 200
                                                else "large")
                    # Buyback: shares declining vs 4 quarters ago
                    if len(shr) > 4 and shr[4] > 0:
                        result["buyback_trend"] = shr[0] < shr[4] * 0.98

                result["q_data_available"] = True
    except: pass

    # ── ANNUAL DATA (A criteria) ──────────────────────────────────────────────
    try:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
        r = requests.get(url, params={"period":"annual","limit":5,"apikey":fmp_key}, timeout=15)
        if r.ok:
            adata = r.json()
            if isinstance(adata, list) and len(adata) >= 4:
                ann_eps = [safe_eps(q) for q in adata]   # newest first
                ann_rev = [safe_rev(q) for q in adata]
                ann_shr = [safe_shares(q) for q in adata]

                # 3-year compound EPS growth (Y0 vs Y3)
                if ann_eps[3] != 0 and ann_eps[0] != 0:
                    cagr_eps = ((ann_eps[0]/abs(ann_eps[3]))**(1/3) - 1)*100
                    result["annual_eps_growth_3y"] = round(cagr_eps, 1)

                # 3-year compound Rev growth
                if ann_rev[3] > 0 and ann_rev[0] > 0:
                    cagr_rev = ((ann_rev[0]/ann_rev[3])**(1/3) - 1)*100
                    result["annual_rev_growth_3y"] = round(cagr_rev, 1)

                # Annual EPS stable: each of last 3 years grew YoY
                # Y0>Y1, Y1>Y2, Y2>Y3 (all growing, no down years)
                grew = all(
                    ann_eps[i] > 0 and ann_eps[i+1] > 0 and ann_eps[i] > ann_eps[i+1]
                    for i in range(3)
                )
                result["annual_eps_stable"] = grew

                # Shares declining trend (buyback confirmation from annual)
                if not result["buyback_trend"] and len(ann_shr) >= 3 and ann_shr[0] > 0:
                    result["buyback_trend"] = ann_shr[0] < ann_shr[2] * 0.97

                result["a_data_available"] = True
    except: pass

    return result


def compute_distribution_days(spy_df: pd.DataFrame, qqq_df: pd.DataFrame) -> dict:
    """
    O'Neil distribution day count: down day on above-average volume on SPY/QQQ.
    Count over last 25 sessions. 4+ = caution, 6+ = likely market top.
    Also computes follow-through day (FTD): big up day on volume after a low.
    """
    result = {
        "spy_dist_days": 0,
        "qqq_dist_days": 0,
        "combined_dist": 0,
        "market_under_pressure": False,
        "follow_through_day": False,
        "ftd_days_ago": None,
    }
    def count_dist(df, window=25):
        if df is None or len(df) < window + 20: return 0
        tail = df.tail(window + 20)
        vol_avg = float(tail["Volume"].rolling(50, min_periods=20).mean().iloc[-window-1])
        count = 0
        for i in range(-window, 0):
            c  = float(df["Close"].iloc[i])
            o  = float(df["Open"].iloc[i])
            v  = float(df["Volume"].iloc[i])
            pc = float(df["Close"].iloc[i-1])
            # Distribution: close lower than prior close, on above-average volume
            # O'Neil also counts stalling days (closes near high but in upper range on big vol)
            if c < pc and v > vol_avg * 1.05:
                count += 1
        return count

    try:
        sd = count_dist(spy_df)
        qd = count_dist(qqq_df)
        result["spy_dist_days"] = sd
        result["qqq_dist_days"] = qd
        result["combined_dist"] = max(sd, qd)
        result["market_under_pressure"] = max(sd, qd) >= 4
    except: pass

    # Follow-Through Day: big up day (≥1.7%) on higher volume, day 4+ after a low
    try:
        if spy_df is not None and len(spy_df) >= 30:
            # Find most recent low
            lows = spy_df["Close"].tail(40)
            low_idx = lows.idxmin()
            low_pos = list(lows.index).index(low_idx)
            bars_since_low = len(lows) - 1 - low_pos
            if bars_since_low >= 4:
                # Check last 5 bars for a FTD
                for i in range(-5, 0):
                    c  = float(spy_df["Close"].iloc[i])
                    pc = float(spy_df["Close"].iloc[i-1])
                    v  = float(spy_df["Volume"].iloc[i])
                    va = float(spy_df["Volume"].rolling(50).mean().iloc[i])
                    gain = (c/pc - 1)*100
                    if gain >= 1.7 and v > va:
                        result["follow_through_day"] = True
                        result["ftd_days_ago"] = abs(i)
                        break
    except: pass

    return result


def enrich_with_fmp(hit_tickers: list[str], fmp_key: str, cache: dict) -> dict:
    """
    Fetch comprehensive FMP data for all scan hits.
    Two API calls per ticker (quarterly + annual) = ~2 req per stock.
    FMP free: 250 req/day → handles ~120 tickers/day comfortably.
    Data cached for 2 days to avoid burning quota on repeated runs.
    """
    if not fmp_key:
        return cache

    fmp_needed = []
    now = datetime.now()
    for t in hit_tickers:
        si = cache.get(t, {})
        last_fmp = si.get("_fmp_fetched")
        if last_fmp:
            try:
                age = (now - datetime.fromisoformat(last_fmp)).days
                if age < 2: continue
            except: pass
        fmp_needed.append(t)

    if not fmp_needed:
        print(f"  ✓  FMP data: all cached ({len(hit_tickers)} tickers)")
        return cache

    print(f"  ↓  FMP comprehensive fundamentals for {len(fmp_needed)} tickers…")
    out = dict(cache)
    for i, t in enumerate(fmp_needed):
        fmp = fetch_fmp_full(t, fmp_key)
        si = out.get(t, {})
        si.update({**fmp, "_fmp_fetched": now.isoformat()})
        out[t] = si
        time.sleep(0.6)   # ~100 tickers/min, well within free tier
        if i > 0 and i % 20 == 0:
            print(f"     … {i}/{len(fmp_needed)}")
            save_sector_cache(out)

    save_sector_cache(out)
    return out
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
    # Prefer quarterly MRQ data over TTM when available
    eps_val = si.get("mrq_eps_growth") if si.get("q_data_available") else si.get("eps_growth", 0)
    rev_val = si.get("mrq_rev_growth") if si.get("q_data_available") else si.get("rev_growth", 0)
    return {
        "ticker":          ticker,
        "name":            si.get("name", ticker),
        "sector":          si.get("sector",""),
        "industry":        si.get("industry",""),
        "price":           round(float(daily["Close"].iloc[-1]), 2),
        "rs_pctile":       rs_pctile,
        "rs":              rs_pctile,
        "eps_growth":      eps_val or 0,
        "rev_growth":      rev_val or 0,
        "eps":             eps_val or 0,
        "rev":             rev_val or 0,
        "inst":            si.get("inst_own", 0),
        "short_float":     si.get("short_float", None),
        "sf":              si.get("short_float", None),
        "eps_accel":       si.get("eps_accelerating", False),
        "rev_accel":       si.get("rev_accelerating", False),
        "eps_positive":    si.get("eps_positive", None),
        "q_data":          si.get("q_data_available", False),
        "score":           score,
        "status":          status,
        "tags":            tags,
        "chart":           ohlc_chart(daily, CFG["chart_bars"]),
        "weekly_chart":    weekly_ohlc_chart(daily, 104),
        "rs_line":         rs_l,
        "weeks_tight":     wt,
        "wt":              wt,
        "scan":            scan,
        "scanned_at":      datetime.now(timezone.utc).isoformat(),
        **extra,
    }

def canslim_ok(ticker, rs_pctile, cache) -> bool:
    """
    O'Neil CANSLIM gate — as complete as free data allows.
    C: MRQ EPS ≥20% YoY, profitable
    A: Annual EPS stable (grew each of last 3 years) OR 3yr CAGR ≥15%
    RS: ≥80th percentile
    Both EPS and revenue growing required.
    """
    if rs_pctile < 80: return False
    si = cache.get(ticker, {})
    q_avail = si.get("q_data_available", False)
    a_avail = si.get("a_data_available", False)

    # C criteria — quarterly precision
    if q_avail:
        if not si.get("eps_positive", True): return False
        eps_g = si.get("mrq_eps_growth") or 0
        rev_g = si.get("mrq_rev_growth") or 0
        if eps_g < 20: return False
        if rev_g < 15: return False
    else:
        # Fallback TTM
        if (si.get("eps_growth", 0) or 0) < 20: return False
        if (si.get("rev_growth", 0) or 0) < 15: return False

    # A criteria — annual stability check
    if a_avail:
        stable = si.get("annual_eps_stable", False)
        cagr   = si.get("annual_eps_growth_3y") or 0
        # Must have either 3 consecutive growing years OR 3yr CAGR ≥15%
        if not stable and cagr < 15: return False

    return True

# ══════════════════════════════════════════════════════════════════════════════
#  BULLISH SCANS
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  CORE BULLISH SCANS  (6 setups — kept tight, no noise)
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared helpers ─────────────────────────────────────────────────────────────
def _adr(daily, n=14):
    """Average Daily Range % over n days."""
    return float(((daily["High"]/daily["Low"]-1)*100).rolling(n).mean().iloc[-1])

def _extension_pct(price, daily):
    """How far % above the 52w low is the stock — proxy for how extended it is."""
    low52 = float(daily["Low"].rolling(min(252,len(daily)), min_periods=50).min().iloc[-1])
    return (price - low52) / low52 * 100 if low52 > 0 else 0

# ══════════════════════════════════════════════════════════════════════════════
# SCAN 1 — VCP (Volatility Contraction Pattern)
# Minervini: prior uptrend ≥30%, price > 150 > 200 SMA, 2-3 contractions each
# ≥⅓ tighter than last, volume drying each contraction, near 52w high.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  METHODOLOGY SCANS — 4 pure implementations
#  Each trader's exact criteria, no compromises between them
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared helper ──────────────────────────────────────────────────────────────
def _adr(daily, n=14):
    return float(((daily["High"]/daily["Low"]-1)*100).rolling(n).mean().iloc[-1])

def _extension_pct(price, daily):
    low52 = float(daily["Low"].rolling(min(252,len(daily)), min_periods=50).min().iloc[-1])
    return (price - low52) / low52 * 100 if low52 > 0 else 0

# ══════════════════════════════════════════════════════════════════════════════
#  QULLAMAGGIE — EP, Bull Flag, EMA Pullback
#  Core rules: high ADR, momentum, RS leader, tight setups
#  No fundamental requirement — he is purely technical for swing trades
#  Uses 10 EMA and 20 EMA (not 9/21)
# ══════════════════════════════════════════════════════════════════════════════
def scan_qullamaggie(ticker, daily, rs_pctile, cache) -> list[dict]:
    results = []
    if not basic_ok(daily): return results
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 40: return results

    stock_adr = _adr(daily)
    s50 = sma(close, 50)
    s50_val = float(s50.iloc[-1])
    e10 = ema(close, 10)
    e20 = ema(close, 20)
    e10_val = float(e10.iloc[-1])
    e20_val = float(e20.iloc[-1])

    # ── 1. EPISODIC PIVOT ──────────────────────────────────────────────────
    # Qullamaggie: gap 10%+ on 3x+ volume from a basing area, RS top 10%
    if rs_pctile >= 80 and len(daily) >= 40:
        for i in range(len(daily)-2, max(0, len(daily)-62), -1):
            gap_pct = (float(daily["Open"].iloc[i]) / float(daily["Close"].iloc[i-1]) - 1) * 100
            if gap_pct < 10: continue
            vol_pre = float(daily["Volume"].iloc[max(0,i-10):i].mean()) if i >= 5 else float(daily["Volume"].mean())
            vol_ratio = float(daily["Volume"].iloc[i]) / max(vol_pre, 1)
            if vol_ratio < 3.0: continue   # Qullamaggie wants 3x+ not just 2.5x
            gap_open = float(daily["Open"].iloc[i])
            bars_since = len(daily) - 1 - i
            if bars_since > 30: continue
            if price < gap_open * 0.97: continue  # must hold above gap open
            # Pre-gap base: must have been consolidating (not running) before gap
            if i >= 15:
                pre = daily.iloc[max(0,i-20):i]
                pre_range = (float(pre["High"].max()) - float(pre["Low"].min())) / float(pre["Close"].mean()) * 100
                if pre_range > 25: continue   # was extended, not basing
            score = round(rs_pctile*.35 + min(100,gap_pct*4)*.25 +
                         min(100,vol_ratio*12)*.20 + max(0,100-bars_since*3)*.20, 1)
            status = "READY" if score>=72 and bars_since<=10 else "GOOD" if score>=57 else "DEVELOPING"
            tags = [f"EP gap +{gap_pct:.0f}%", f"Vol {vol_ratio:.1f}x", "Holding gap"]
            if bars_since <= 3: tags.append("Fresh EP")
            results.append(base_result(ticker, daily, rs_pctile, score, status, tags,
                {"strategy":"EP","methodology":"Qullamaggie",
                 "gap_pct":round(gap_pct,1),"vol_ratio":round(vol_ratio,2),
                 "bars_since_gap":bars_since,"gap_open":round(gap_open,2),
                 "pivot":round(gap_open,2),"from_high_pct":round(max(0,(gap_open-price)/gap_open*100),1)},
                "Episodic Pivot", cache))
            break

    # ── 2. BULL FLAG ───────────────────────────────────────────────────────
    # Qullamaggie: ADR 3%+, pole 15%+ in ≤15 bars, tight flag, volume drying
    if rs_pctile >= 65 and stock_adr >= 3.0 and price >= s50_val * 0.96:
        best = None
        for flag_len in range(5, 21):
            if len(daily) < flag_len + 5: continue
            flag = daily.tail(flag_len)
            f_cls = flag["Close"].values; f_lows = flag["Low"].values
            f_vols = flag["Volume"].values
            f_hi = float(flag["High"].max()); f_lo = float(flag["Low"].min())
            f_mid = float(flag["Close"].mean())
            if f_mid <= 0: continue
            f_range = (f_hi-f_lo)/f_mid*100
            if f_range > 15: continue
            drift = (float(f_cls[-1])-float(f_cls[0]))/float(f_cls[0])*100
            if drift > 3 or drift < -12: continue
            # Price above 10 EMA throughout (Qullamaggie rule)
            e10_flag = e10.iloc[-flag_len:].values
            ema_breaks = sum(1 for i in range(len(f_cls)) if f_cls[i] < e10_flag[i]*0.985)
            if ema_breaks > flag_len//3: continue
            half = max(1, flag_len//2)
            if float(min(f_lows[half:])) < float(min(f_lows[:half]))*0.97: continue
            vol_contracting = float(np.mean(f_vols[half:])) < float(np.mean(f_vols[:half]))
            for pole_len in range(5, 16):
                if len(daily) < flag_len+pole_len: break
                pole = daily.iloc[-(flag_len+pole_len):-flag_len]
                if len(pole) < 4: continue
                green = sum(1 for i in range(len(pole))
                           if float(pole["Close"].iloc[i]) >= float(pole["Open"].iloc[i]))
                if green/len(pole) < 0.60: continue
                pole_lo = float(pole["Low"].min()); pole_top = float(pole["High"].max())
                if pole_lo <= 0: continue
                pole_gain = (pole_top-pole_lo)/pole_lo*100
                if pole_gain < 15: continue
                pole_close = float(pole["Close"].iloc[-1])
                straightness = (pole_close-pole_lo)/(pole_top-pole_lo) if pole_top>pole_lo else 0
                if straightness < 0.55: continue
                if f_range > pole_gain*0.45: continue
                vol_flag = float(flag["Volume"].mean()); vol_pole = float(pole["Volume"].mean())
                vol_ratio = vol_flag/max(vol_pole,1)
                if vol_ratio > 0.85: continue
                from_top = (pole_top-price)/pole_top*100
                if from_top > 8 or from_top < -3: continue
                score = round(min(100,pole_gain/80*100)*.28 + max(0,100-f_range*5)*.22 +
                             max(0,(1-vol_ratio)*100)*.18 + max(0,100-ema_breaks*15)*.12 +
                             rs_pctile*.20 + (8 if vol_contracting else 0), 1)
                if best is None or score > best["score"]:
                    status = "READY" if score>=70 and from_top<=3 else "GOOD" if score>=55 else "DEVELOPING"
                    tags = [f"Pole +{pole_gain:.0f}%", f"Flag {flag_len}d"]
                    if vol_ratio < 0.5: tags.append("Vol dried up")
                    if vol_contracting: tags.append("Vol contracting")
                    if from_top <= 3: tags.append("Near breakout")
                    best = base_result(ticker, daily, rs_pctile, score, status, tags,
                        {"strategy":"FLAG","methodology":"Qullamaggie",
                         "pole_gain_pct":round(pole_gain,1),"flag_days":flag_len,
                         "flag_range_pct":round(f_range,1),"vol_ratio":round(vol_ratio,2),
                         "pivot":round(pole_top,2),"from_high_pct":round(from_top,1)},
                        "Bull Flag", cache)
        if best: results.append(best)

    # ── 3. EMA PULLBACK ────────────────────────────────────────────────────
    # Qullamaggie: pullback to 10 or 20 EMA, low volume, stock in uptrend
    # He specifically uses 10/20 EMA, not 9/21
    if rs_pctile >= 65 and stock_adr >= 2.5:
        if e10_val > e20_val and price >= s50_val * 0.97:
            dist10 = (price-e10_val)/e10_val*100
            dist20 = (price-e20_val)/e20_val*100
            near10 = -3 <= dist10 <= 6
            near20 = -3 <= dist20 <= 8
            if near10 or near20:
                vol_today = float(daily["Volume"].iloc[-1])
                vol_avg = float(daily["Volume"].rolling(20).mean().iloc[-1])
                vol_ratio = vol_today/max(vol_avg,1)
                if vol_ratio <= 1.3:
                    # Volume declining into pullback
                    if len(daily) >= 3:
                        v3 = [float(daily["Volume"].iloc[i]) for i in range(-3,0)]
                        if v3[-1] > v3[0]*1.3: pass  # volume rising = skip
                        else:
                            # Prior uptrend
                            prior_gain = (price-float(close.iloc[-21]))/float(close.iloc[-21])*100 if len(close)>=21 else 5
                            if prior_gain >= 0:
                                best_dist = dist10 if near10 and abs(dist10)<abs(dist20) else dist20
                                ema_type = "10 EMA" if near10 and abs(dist10)<abs(dist20) else "20 EMA"
                                score = round(rs_pctile*.35 + max(0,100-abs(best_dist)*12)*.25 +
                                             max(0,(1-vol_ratio)*80)*.20 +
                                             max(0,100-(price/s50_val-1)*100)*.20, 1)
                                status = "READY" if score>=65 else "GOOD" if score>=50 else "DEVELOPING"
                                results.append(base_result(ticker, daily, rs_pctile, score, status,
                                    [f"Pullback to {ema_type}", "Low vol pullback", "Uptrend intact"],
                                    {"strategy":"EMA_PULL","methodology":"Qullamaggie",
                                     "ema10":round(e10_val,2),"ema20":round(e20_val,2),
                                     "dist_ema_pct":round(abs(best_dist),1),"vol_ratio":round(vol_ratio,2),
                                     "pivot":round(float(close.rolling(20).max().iloc[-1]),2),
                                     "from_high_pct":round(abs(best_dist),1)},
                                    "EMA Pullback", cache))
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MINERVINI — VCP (Volatility Contraction Pattern)
#  SEPA criteria: exact uptrend template, strict contractions
#  Requires fundamentals: EPS 25%+, Rev 20%+
# ══════════════════════════════════════════════════════════════════════════════
def scan_minervini(ticker, daily, rs_pctile, cache) -> dict | None:
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(close) < 200: return None
    if rs_pctile < 80: return None   # Minervini SEPA requires RS 80+ strictly

    # SEPA Uptrend Template — Minervini's exact requirements:
    # 1. Price above 50 SMA, 150 SMA, and 200 SMA
    # 2. 50 SMA above 150 SMA above 200 SMA (all in proper order)
    # 3. 200 SMA trending up for at least 1 month
    # 4. Price at least 30% above 52-week low
    # 5. Price within 25% of 52-week high
    s50  = float(sma(close, 50).iloc[-1])
    s150 = float(sma(close, 150).iloc[-1])
    s200 = float(sma(close, 200).iloc[-1])
    s200_1m = float(sma(close, 200).iloc[-21])

    # Strict MA stack — all must be in order
    if not (price > s50 and price > s150 and price > s200): return None
    if not (s50 > s150 and s150 > s200): return None
    if s200 <= s200_1m * 0.998: return None   # 200 SMA must be rising

    # Sector filter: Minervini focuses on growth stocks — not banks or utilities
    si = cache.get(ticker, {})
    industry = si.get("industry", "")
    if industry in ("Banks—Diversified", "Banks—Regional", "Insurance—Life",
                    "Insurance—Diversified", "Asset Management", "Capital Markets"): return None

    high52 = float(daily["High"].rolling(252, min_periods=100).max().iloc[-1])
    low52  = float(daily["Low"].rolling(252,  min_periods=100).min().iloc[-1])
    from_high = (high52-price)/high52*100
    from_low  = (price-low52)/low52*100

    if from_high > 25: return None    # within 25% of 52w high
    if from_low  < 30: return None    # at least 30% above 52w low

    # Fundamentals — Minervini requires strong EPS and Revenue
    # Only gate if we actually have data (not empty cache on first run)
    si = cache.get(ticker, {})
    q_avail = si.get("q_data_available", False)
    eps_g = (si.get("mrq_eps_growth") if q_avail else si.get("eps_growth",0)) or 0
    rev_g = (si.get("mrq_rev_growth") if q_avail else si.get("rev_growth",0)) or 0
    has_data = bool(si.get("eps_growth") or si.get("mrq_eps_growth"))
    if has_data:
        if eps_g < 25 and rev_g < 20: return None
        if q_avail and not si.get("eps_positive", True): return None

    # VCP: 3 windows contracting — Minervini's 10/20/40 day framework
    bars = daily.tail(100)
    def range_pct(n):
        sl = bars.tail(n); mid = float(sl["Close"].mean())
        return (float(sl["High"].max())-float(sl["Low"].min()))/mid*100 if mid>0 else 99
    r10, r20, r40 = range_pct(10), range_pct(20), range_pct(40)

    # Each window must be meaningfully tighter (≥10% tighter each step)
    if not (r10 < r20*0.90 and r20 < r40*0.90): return None
    if r10 > 10: return None    # Minervini: final contraction ≤10%
    if r40 < 15: return None    # must have had prior range to contract from

    # Volume must be clearly drying up — Minervini wants 40-50% below average
    vol10 = float(bars["Volume"].tail(10).mean())
    vol40 = float(bars["Volume"].tail(40).mean())
    vol_dry = vol10/max(vol40,1)
    if vol_dry > 0.75: return None   # Minervini: volume must dry up 25-40% below average

    # Prior uptrend before base
    low_pre = float(daily["Low"].iloc[-200:-60].min()) if len(daily)>=200 else low52
    prior_run = (high52-low_pre)/low_pre*100 if low_pre>0 else 0
    if prior_run < 30: return None

    contractions = 1
    if r20 < r40*0.75: contractions = 2
    if r10 < r20*0.70: contractions = 3
    if contractions < 2: return None

    score = round(rs_pctile*.30 + max(0,100-from_high*3)*.20 +
                 max(0,100-r10*8)*.20 + max(0,(1-vol_dry)*100)*.15 +
                 min(100,contractions*33)*.10 + min(20,eps_g/5)*.05, 1)
    status = ("READY" if score>=72 and from_high<=5 and contractions>=3
              else "GOOD" if score>=58 and from_high<=12 else "DEVELOPING")
    tags = [f"{contractions} contractions", f"Range {r10:.0f}% tight"]
    if vol_dry < 0.5:   tags.append("Vol drying up")
    if from_high <= 5:  tags.append("Near pivot")
    if eps_g >= 25:     tags.append(f"EPS +{eps_g:.0f}%")

    return base_result(ticker, daily, rs_pctile, score, status, tags,
        {"strategy":"VCP","methodology":"Minervini",
         "pivot":round(high52,2),"from_high_pct":round(from_high,1),
         "range_10d":round(r10,1),"range_20d":round(r20,1),"range_40d":round(r40,1),
         "contractions":contractions,"vol_ratio":round(vol_dry,2),
         "sma50":round(s50,2),"sma150":round(s150,2),"sma200":round(s200,2)},
        "VCP", cache)


# ══════════════════════════════════════════════════════════════════════════════
#  O'NEIL / IBD — Base Breakout + Pocket Pivot
#  C+A+N+S+L+I+M criteria applied to technical setups
#  RS 80+, EPS 25%+, Rev 20%+, profitable, breakout volume required
# ══════════════════════════════════════════════════════════════════════════════
def scan_oneil(ticker, daily, rs_pctile, cache) -> list[dict]:
    results = []
    if not basic_ok(daily): return results
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 120: return results
    if rs_pctile < 75: return results   # O'Neil: RS 80+ preferred

    si = cache.get(ticker, {})
    q_avail = si.get("q_data_available", False)
    eps_g = (si.get("mrq_eps_growth") if q_avail else si.get("eps_growth",0)) or 0
    rev_g = (si.get("mrq_rev_growth") if q_avail else si.get("rev_growth",0)) or 0
    profitable = si.get("eps_positive", True) if q_avail else True

    # O'Neil fundamental gate — only apply if we have real data
    # If cache is empty (first run), let technical setup through and filter later
    has_data = bool(si.get("eps_growth") or si.get("mrq_eps_growth"))
    if has_data:
        if not profitable: return results
        if eps_g < 20 or rev_g < 15: return results

    s50  = float(sma(close, 50).iloc[-1])
    if price < s50 * 0.96: return results

    # ── 1. BASE BREAKOUT ────────────────────────────────────────────────────
    # O'Neil: 6+ weeks, ≤15% range, volume contraction, breakout on 40%+ vol
    if len(close) >= 150:
        s150 = float(sma(close, 150).iloc[-1])
        # 150 SMA must be flat or rising (Stage 2)
        sma150_ok = True
        if len(close) >= 170:
            s150_3w = float(sma(close, 150).iloc[-21])
            sma150_ok = s150 >= s150_3w * 0.993

        if sma150_ok:
            # Prior uptrend ≥ 25% (O'Neil: stock must have had a prior advance)
            if len(daily) >= 180:
                low_pre    = float(daily["Low"].iloc[-180:-50].min())
                high52_pre = float(daily["High"].iloc[-180:-20].max())
                prior_run  = (high52_pre-low_pre)/low_pre*100 if low_pre>0 else 0
            else:
                prior_run = 25  # assume ok if not enough data

            if prior_run >= 25:
                weekly = daily.resample("W").agg(
                    {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
                if len(weekly) >= 8:
                    weekly["rng"]  = (weekly["High"]-weekly["Low"])/weekly["Close"]*100
                    weekly["s50w"] = sma(weekly["Close"], 10)
                    # Count tight weeks (non-consecutive ok — look at last 20 weeks)
                    recent = weekly.tail(20)
                    tight  = recent[(recent["rng"]<=18) & (recent["Close"]>=recent["s50w"]*0.93)]
                    base_weeks = len(tight)

                    if base_weeks >= 6:
                        base_hi  = float(weekly.tail(base_weeks)["High"].max())
                        base_lo  = float(weekly.tail(base_weeks)["Low"].min())
                        base_mid = float(weekly.tail(base_weeks)["Close"].mean())
                        base_range = (base_hi-base_lo)/base_mid*100 if base_mid>0 else 99

                        if base_range <= 15:  # O'Neil: ≤15% flat base
                            high52 = float(daily["High"].rolling(min(252,len(daily)),min_periods=50).max().iloc[-1])
                            from_high = (high52-price)/high52*100

                            if from_high <= 8:  # within 8% of pivot
                                today_vol = float(daily["Volume"].iloc[-1])
                                avg_vol20 = float(daily["Volume"].rolling(20).mean().iloc[-1])
                                breakout_vol = today_vol/max(avg_vol20,1)

                                # Hard gate: AT the pivot needs volume confirmation
                                at_pivot_no_vol = (from_high <= 3 and breakout_vol < 1.4)
                                if not at_pivot_no_vol:
                                    vol_base  = float(weekly["Volume"].tail(base_weeks).mean())
                                    vol_prior = float(weekly["Volume"].iloc[-(base_weeks+6):-base_weeks].mean()) \
                                               if len(weekly)>base_weeks+6 else vol_base
                                    vol_ratio = vol_base/max(vol_prior,1)

                                    score = round(rs_pctile*.25 + max(0,100-base_range*5)*.20 +
                                                 min(100,base_weeks/15*100)*.15 +
                                                 max(0,(1-vol_ratio)*80)*.15 +
                                                 max(0,100-from_high*10)*.15 +
                                                 min(100,eps_g/2)*.10, 1)
                                    bvol_sc = min(10, breakout_vol*6) if breakout_vol>=1.4 else 0
                                    score = round(score + bvol_sc, 1)

                                    status = ("READY" if score>=70 and from_high<=3 and breakout_vol>=1.4
                                             else "GOOD" if score>=55 and from_high<=6 else "DEVELOPING")
                                    tags = [f"{base_weeks}w base", f"Range {base_range:.0f}%"]
                                    if breakout_vol >= 1.4: tags.append(f"Vol {breakout_vol:.1f}x ✓")
                                    elif from_high <= 5:    tags.append("Near pivot — await vol")
                                    if vol_ratio < 0.65:    tags.append("Vol dried up")
                                    if eps_g >= 25:         tags.append(f"EPS +{eps_g:.0f}%")

                                    results.append(base_result(ticker, daily, rs_pctile, score, status, tags,
                                        {"strategy":"BASE","methodology":"O'Neil",
                                         "pivot":round(high52,2),"from_high_pct":round(from_high,1),
                                         "base_weeks":base_weeks,"base_range_pct":round(base_range,1),
                                         "vol_ratio":round(vol_ratio,2),"breakout_vol":round(breakout_vol,2),
                                         "sma50":round(s50,2)},
                                        "Base Breakout", cache))

    # ── 2. POCKET PIVOT ─────────────────────────────────────────────────────
    # Gil/Morales (O'Neil methodology): up-day vol > max of last 10 down-day vols
    # Must be within base or near 50 SMA — not extended
    s150_val = float(sma(close, 150).iloc[-1]) if len(close)>=150 else s50
    pct_above_50 = (price-s50)/s50*100
    if price >= s50*0.96 and price >= s150_val*0.95 and pct_above_50 <= 20:
        for lookback in range(1, 4):
            idx = -lookback
            if abs(idx)+11 > len(daily): continue
            bar = daily.iloc[idx]
            if float(bar["Close"]) < float(bar["Open"]): continue  # must be up day
            vol_today = float(bar["Volume"])
            prior = daily.iloc[idx-10:idx]
            down_vols = [float(prior["Volume"].iloc[i]) for i in range(len(prior))
                        if float(prior["Close"].iloc[i]) < float(prior["Open"].iloc[i])]
            if len(down_vols) < 3: continue
            max_down_vol = max(down_vols)
            if vol_today <= max_down_vol: continue
            vol_ratio = vol_today/max_down_vol
            score = round(rs_pctile*.30 + min(100,vol_ratio*30)*.25 +
                         max(0,100-pct_above_50*4)*.20 + max(0,100-lookback*20)*.15 +
                         min(20,eps_g/5)*.10, 1)
            status = "READY" if score>=65 and lookback==1 else "GOOD" if score>=52 else "DEVELOPING"
            tags = [f"PP vol {vol_ratio:.1f}x", "Within base", f"EPS +{eps_g:.0f}%"]
            if pct_above_50 < 5: tags.append("Near 50 SMA")
            results.append(base_result(ticker, daily, rs_pctile, score, status, tags,
                {"strategy":"PP","methodology":"O'Neil",
                 "vol_ratio":round(vol_ratio,2),"bars_ago":lookback,
                 "sma50":round(s50,2),"pct_above_50":round(pct_above_50,1),
                 "pivot":round(price*1.05,2),"from_high_pct":round(pct_above_50,1)},
                "Pocket Pivot", cache))
            break

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  WEINSTEIN — Stage 2 Breakout
#  Purely technical — no fundamental requirement
#  30-week SMA rising, price breaks out of Stage 1 base, volume expanding
#  Works for any stock regardless of earnings
# ══════════════════════════════════════════════════════════════════════════════
def scan_weinstein(ticker, daily, rs_pctile, cache) -> dict | None:
    if not basic_ok(daily): return None
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 200: return None
    if rs_pctile < 50: return None   # Weinstein is purely technical, RS 50+ sufficient

    weekly = daily.resample("W").agg(
        {"High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if len(weekly) < 35: return None

    w30 = sma(weekly["Close"], 30)
    if pd.isna(w30.iloc[-1]): return None
    sma30_now = float(w30.iloc[-1])
    price_w   = float(weekly["Close"].iloc[-1])

    # Stage 2: price above 30-week SMA
    if price_w <= sma30_now * 1.00: return None  # must be clearly above

    # 30-week SMA must be rising — this is Weinstein's core signal
    sma30_8w  = float(w30.iloc[-9]) if len(w30)>=9 else sma30_now
    sma30_4w  = float(w30.iloc[-5]) if len(w30)>=5 else sma30_now
    if sma30_now <= sma30_8w * 1.001: return None  # must be clearly rising

    # How many weeks in Stage 2 (above 30w SMA)
    weeks_above = 0
    for i in range(len(weekly)-1, max(0,len(weekly)-52), -1):
        if float(weekly["Close"].iloc[i]) > float(w30.iloc[i]):
            weeks_above += 1
        else:
            break

    # Early Stage 2 preferred (Weinstein: buy the breakout, not after 6 months)
    if weeks_above > 20: return None   # Weinstein: buy early-to-mid Stage 2
    if weeks_above < 1:  return None

    # Stage 1 base before the breakout — not strictly required by Weinstein
    # His core criteria is just: price above rising 30wk SMA, volume expanding

    # Volume expanding on breakout (Weinstein: needs institutional buying)
    vol_recent = float(weekly["Volume"].tail(4).mean())
    vol_prior  = float(weekly["Volume"].iloc[-12:-4].mean()) if len(weekly)>=12 else vol_recent
    vol_ratio  = vol_recent/max(vol_prior,1)
    # No hard volume gate — score it instead (low vol = lower score)

    # RS line should be strong (Weinstein uses Mansfield RS)
    # We approximate: stock must be outperforming SPY over 26 weeks
    rs_line_vals = rs_line(daily, _SPY, min(len(daily), 130))
    rs_leading = False
    if len(rs_line_vals) >= 26:
        rs_now  = rs_line_vals[-1]
        rs_26w  = rs_line_vals[-26]
        rs_leading = rs_now >= rs_26w  # RS line rising = leading the market

    sma30_slope_pct = (sma30_now - sma30_8w) / sma30_8w * 100
    score = round(
        rs_pctile * .40 +                              # RS (max 40)
        min(20, max(0, sma30_slope_pct * 4)) +         # SMA30 slope (max 20)
        max(0, 20 - weeks_above * 1.5) +               # earlier = better (max 20)
        min(15, vol_ratio * 9) +                       # volume expansion (max 15)
        (5 if rs_leading else 0),                      # RS line bonus (max 5)
    1)
    score = min(100, max(0, score))   # hard cap 0-100

    status = ("READY" if score>=65 and weeks_above<=6 and vol_ratio>=1.2
              else "GOOD" if score>=52 and weeks_above<=15 else "DEVELOPING")
    tags = [f"Stage 2 wk {weeks_above}", "30wk SMA rising"]
    if vol_ratio >= 1.3:  tags.append(f"Vol {vol_ratio:.1f}x expanding")
    if rs_leading:        tags.append("RS line leading")
    if weeks_above <= 4:  tags.append("Early Stage 2")

    return base_result(ticker, daily, rs_pctile, score, status, tags,
        {"strategy":"STAGE2","methodology":"Weinstein",
         "sma30w":round(sma30_now,2),"weeks_in_stage2":weeks_above,
         "vol_ratio":round(vol_ratio,2),"sma30_slope":round(sma30_slope_pct,2),
         "pivot":round(price_w*1.02,2),"from_high_pct":round((float(daily["High"].rolling(52,min_periods=10).max().iloc[-1])-price)/price*100,1)},
        "Stage 2", cache)



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
    """Run all 5 CANSLIM sub-scans using comprehensive FMP data when available."""
    if not basic_ok(daily): return []
    close = daily["Close"]; price = float(close.iloc[-1])
    if len(daily) < 100: return []
    si = cache.get(ticker, {})

    q_avail    = si.get("q_data_available", False)
    a_avail    = si.get("a_data_available", False)
    eps_g      = (si.get("mrq_eps_growth") if q_avail else si.get("eps_growth", 0)) or 0
    rev_g      = (si.get("mrq_rev_growth") if q_avail else si.get("rev_growth", 0)) or 0
    eps_accel  = si.get("eps_accelerating", False)
    rev_accel  = si.get("rev_accelerating", False)
    eps_3q     = si.get("eps_accel_3q", False)     # 3 consecutive quarters accelerating
    rev_3q     = si.get("rev_accel_3q", False)
    eps_pos    = si.get("eps_positive", True)
    ann_stable = si.get("annual_eps_stable", False)
    ann_cagr   = si.get("annual_eps_growth_3y") or 0
    ann_rev_cagr = si.get("annual_rev_growth_3y") or 0
    buyback    = si.get("buyback_trend", False)
    float_cat  = si.get("float_category", "large")  # small/mid/large
    inst       = si.get("inst_own", 0) or 0
    mktcap     = si.get("mktcap", 0) or 0

    if q_avail and not eps_pos: return []

    results = []

    def build_tags(base, extra_conditions):
        t = list(base)
        if eps_3q:      t.append("EPS accel 3Q ↑↑")
        elif eps_accel: t.append("EPS accel ↑")
        if rev_accel:   t.append("Rev accel ↑")
        if ann_stable:  t.append("3yr EPS growth")
        if buyback:     t.append("Buybacks ↑")
        if float_cat == "small": t.append("Small float")
        for cond in extra_conditions: t.append(cond)
        return t

    # S/L bonus points
    float_bonus   = 8 if float_cat=="small" else 4 if float_cat=="mid" else 0
    buyback_bonus = 5 if buyback else 0
    annual_bonus  = 8 if ann_stable else (4 if ann_cagr >= 15 else 0)
    accel_bonus   = 8 if eps_3q else (4 if eps_accel else 0)

    # 1. New Highs + Earnings (C+A+N)
    if eps_g >= 25 and rev_g >= 20 and rs_pctile >= 80:
        high52 = float(daily["High"].rolling(min(252,len(daily))).max().iloc[-1])
        from_high = (high52-price)/high52*100
        if from_high <= 8:
            s50_val = float(sma(close, 50).iloc[-1])
            if price >= s50_val * 0.97:
                score = round(
                    rs_pctile*.28 + min(100,eps_g/1.5)*.22 + min(100,rev_g/1.5)*.18 +
                    max(0,100-from_high*8)*.22 +
                    accel_bonus*.04 + annual_bonus*.03 + float_bonus*.02 + buyback_bonus*.01,
                1)
                results.append(base_result(ticker,daily,rs_pctile,score,
                    "READY" if score>=75 else "GOOD",
                    build_tags([f"EPS +{eps_g:.0f}%", f"Rev +{rev_g:.0f}%", f"RS {rs_pctile:.0f}th"], []),
                    {"strategy":"NH_EARNINGS","canslim":score,"pattern":"New High",
                     "from_high_pct":round(from_high,1),"inst":inst},"CANSLIM",cache))

    # 2. Market Leader Pullback (first pullback to 50d)
    if rs_pctile >= 80 and eps_g >= 20 and rev_g >= 15:
        s50 = sma(close,50); s50_val = float(s50.iloc[-1])
        dist_50 = (price-s50_val)/s50_val*100
        s50_4w = float(s50.iloc[-21]) if len(s50)>=21 else s50_val
        was_above = float(close.iloc[-21]) > s50_4w*1.06 if len(close)>=21 else False
        if was_above and -2 <= dist_50 <= 5:
            score = round(
                rs_pctile*.32 + max(0,(5-abs(dist_50))*12)*.25 +
                min(100,eps_g/2)*.20 + min(100,rev_g/2)*.18 +
                accel_bonus*.03 + annual_bonus*.02,
            1)
            results.append(base_result(ticker,daily,rs_pctile,score,
                "READY" if score>=72 else "GOOD",
                build_tags([f"First 50d pullback", f"RS {rs_pctile:.0f}th", f"EPS +{eps_g:.0f}%"], []),
                {"strategy":"LEADER_PULLBACK","canslim":score,"pattern":"50d SMA Test",
                 "from_high_pct":round(dist_50,1),"inst":inst},"CANSLIM",cache))

    # 3. Emerging Leader (small/mid cap + 3-quarter acceleration + early Stage 2)
    if eps_g >= 30 and rev_g >= 25 and rs_pctile >= 75 and mktcap < 10e9:
        s200_val = float(sma(close,200).iloc[-1]) if len(close)>=200 else price*.9
        s50_val  = float(sma(close,50).iloc[-1])
        if price > s200_val and price > s50_val * 0.97:
            size_bonus = 20 if mktcap < 2e9 else 10
            score = round(
                rs_pctile*.28 + min(100,eps_g/1.5)*.28 +
                min(100,rev_g/1.5)*.20 + size_bonus*.12 +
                accel_bonus*.07 + float_bonus*.05,
            1)
            results.append(base_result(ticker,daily,rs_pctile,score,
                "READY" if score>=70 else "GOOD",
                build_tags([f"EPS +{eps_g:.0f}%", f"Rev +{rev_g:.0f}%", "Small cap leader"], []),
                {"strategy":"EMERGING","canslim":score,"pattern":"Emerging Stage 2",
                 "from_high_pct":5.0,"inst":inst},"CANSLIM",cache))

    # 4. Institutional Accumulation — requires buyback confirmation when available
    if inst >= 40 and rs_pctile >= 70 and eps_g >= 20 and rev_g >= 15:
        s50_val = float(sma(close,50).iloc[-1])
        if price >= s50_val * 0.97:
            score = round(
                rs_pctile*.28 + min(100,inst)*.22 +
                min(100,eps_g/2)*.22 + min(100,rev_g/2)*.18 +
                buyback_bonus*.05 + accel_bonus*.05,
            1)
            if score >= 65:
                results.append(base_result(ticker,daily,rs_pctile,score,
                    "GOOD" if score>=70 else "DEVELOPING",
                    build_tags([f"Inst own {inst:.0f}%", f"EPS +{eps_g:.0f}%", f"Rev +{rev_g:.0f}%"], []),
                    {"strategy":"INST_ACCUM","canslim":score,"pattern":"Accumulation Base",
                     "from_high_pct":8.0,"inst":inst},"CANSLIM",cache))

    # 5. Industry Group Leader — top-tier RS + 3yr annual EPS growth
    if rs_pctile >= 88 and eps_g >= 20 and rev_g >= 15:
        high52 = float(daily["High"].rolling(min(252,len(daily))).max().iloc[-1])
        from_high = (high52-price)/high52*100
        score = round(
            rs_pctile*.38 + min(100,eps_g/1.5)*.28 + min(100,rev_g/2)*.22 +
            annual_bonus*.07 + accel_bonus*.05,
        1)
        results.append(base_result(ticker,daily,rs_pctile,score,
            "READY" if score>=80 else "GOOD",
            build_tags([f"RS {rs_pctile:.0f}th percentile", f"EPS +{eps_g:.0f}%", "Sector leader"],
                       [f"Annual CAGR +{ann_cagr:.0f}%"] if ann_cagr > 0 else []),
            {"strategy":"INDUSTRY_LEADER","canslim":score,"pattern":"Market Leader",
             "from_high_pct":round(from_high,1),"inst":inst},"CANSLIM",cache))

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
    ca_tickers  = []   # US only
    all_tickers = list(dict.fromkeys(us_tickers +
                   list(SECTOR_TICKERS.keys()) + ["SPY","QQQ","VIX"]))
    data = download_ohlcv(all_tickers)
    global _SPY
    _SPY = data.get("SPY", pd.DataFrame())
    spy_df = _SPY
    print("  ∑  Building index & sector data…")
    indexes = fetch_indexes(data)
    sectors = fetch_sectors(data, spy_df)
    regime_spy = data.get("SPY", pd.DataFrame())
    if len(regime_spy) >= 200:
        s50  = float(sma(regime_spy["Close"], 50).iloc[-1])
        s200 = float(sma(regime_spy["Close"], 200).iloc[-1])
        s50_3w = float(sma(regime_spy["Close"], 50).iloc[-15])
        # BULLISH: SPY > 50 SMA > 200 SMA and 50 SMA rising
        # BEARISH: SPY < 50 SMA < 200 SMA or 50 SMA declining sharply
        spy_price = float(regime_spy["Close"].iloc[-1])
        if spy_price > s50 and s50 > s200 and s50 >= s50_3w * 0.998:
            regime = "BULLISH"
        elif spy_price < s50 and s50 < s200 * 1.01:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"
    elif len(regime_spy) >= 50:
        s50  = float(sma(regime_spy["Close"], 50).iloc[-1])
        spy_price = float(regime_spy["Close"].iloc[-1])
        regime = "BULLISH" if spy_price > s50 * 1.01 else "BEARISH" if spy_price < s50 * 0.98 else "NEUTRAL"
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

    # Distribution days (O'Neil M criteria)
    print("  ∑  Computing distribution days…")
    qqq_df = data.get("QQQ", pd.DataFrame())
    dist = compute_distribution_days(spy_df, qqq_df)
    breadth["spy_dist_days"]   = dist["spy_dist_days"]
    breadth["qqq_dist_days"]   = dist["qqq_dist_days"]
    breadth["combined_dist"]   = dist["combined_dist"]
    breadth["market_under_pressure"] = dist["market_under_pressure"]
    breadth["follow_through_day"]    = dist["follow_through_day"]
    print(f"     SPY dist days: {dist['spy_dist_days']}  QQQ: {dist['qqq_dist_days']}"
          f"  {'⚠ Market under pressure' if dist['market_under_pressure'] else '✓ OK'}"
          f"{'  FTD detected' if dist['follow_through_day'] else ''}")
    cache = load_sector_cache()
    print(f"\n  🔍  Qullamaggie scans (EP, Flag, EMA Pullback)…")
    q_hits = []
    for t in valid:
        try:
            q_hits.extend(scan_qullamaggie(t, data[t], rs_pct.get(t,0), cache))
        except: pass
    q_hits.sort(key=lambda x: x["score"], reverse=True)
    seen=set(); q_deduped=[]
    for r in q_hits:
        if r["ticker"] not in seen: q_deduped.append(r); seen.add(r["ticker"])
    q_hits = q_deduped[:CFG["top_n"]]
    print(f"     → {len(q_hits)} setups ({dict(__import__('collections').Counter(r['strategy'] for r in q_hits))})")

    print(f"\n  🔍  Minervini scans (VCP)…")
    m_hits = []
    for t in valid:
        try:
            r = scan_minervini(t, data[t], rs_pct.get(t,0), cache)
            if r: m_hits.append(r)
        except: pass
    m_hits.sort(key=lambda x: x["score"], reverse=True)
    m_hits = m_hits[:CFG["top_n"]]
    print(f"     → {len(m_hits)} setups")

    print(f"\n  🔍  O'Neil/IBD scans (Base Breakout, Pocket Pivot)…")
    o_hits = []
    for t in valid:
        try:
            o_hits.extend(scan_oneil(t, data[t], rs_pct.get(t,0), cache))
        except: pass
    o_hits.sort(key=lambda x: x["score"], reverse=True)
    seen=set(); o_deduped=[]
    for r in o_hits:
        if r["ticker"] not in seen: o_deduped.append(r); seen.add(r["ticker"])
    o_hits = o_deduped[:CFG["top_n"]]
    print(f"     → {len(o_hits)} setups ({dict(__import__('collections').Counter(r['strategy'] for r in o_hits))})")

    print(f"\n  🔍  Weinstein scans (Stage 2)…")
    w_hits = []
    for t in valid:
        try:
            r = scan_weinstein(t, data[t], rs_pct.get(t,0), cache)
            if r: w_hits.append(r)
        except: pass
    w_hits.sort(key=lambda x: x["score"], reverse=True)
    w_hits = w_hits[:CFG["top_n"]]
    print(f"     → {len(w_hits)} setups")

    # Combine all swing hits for fundamentals enrichment
    all_hits_swing = q_hits + m_hits + o_hits + w_hits
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
        seen = set(); deduped = []
        for r in hits:
            if r["ticker"] not in seen:
                deduped.append(r); seen.add(r["ticker"])
        return deduped[:CFG["top_n"]]

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
    seen_lt = set(); lt_deduped = []
    for r in lt_hits:
        if r["ticker"] not in seen_lt:
            lt_deduped.append(r); seen_lt.add(r["ticker"])
    lt_hits = lt_deduped[:CFG["top_n"]]
    print(f"     CANSLIM          {len(lt_hits)} unique tickers")

    all_hits = all_hits_swing + swing_bear + swing_chop + lt_hits
    hit_tickers = list({r["ticker"] for r in all_hits})
    if hit_tickers:
        print(f"\n  ↓  Fetching fundamentals for {len(hit_tickers)} hits…")
        cache = enrich_fundamentals(hit_tickers, cache)
        fmp_key = cfg.get("FMP_API_KEY", "")
        if fmp_key:
            cache = enrich_with_fmp(hit_tickers, fmp_key, cache)
        else:
            print("  ⚠  Skipping FMP quarterly data (no FMP_API_KEY in config)")
        for r in all_hits:
            si = cache.get(r["ticker"], {})
            q_avail = si.get("q_data_available", False)
            r["name"]          = si.get("name", r.get("name", r["ticker"]))
            r["sector"]        = si.get("sector", r.get("sector",""))
            r["industry"]      = si.get("industry","")
            r["eps_growth"]    = (si.get("mrq_eps_growth") if q_avail else si.get("eps_growth",0)) or 0
            r["rev_growth"]    = (si.get("mrq_rev_growth") if q_avail else si.get("rev_growth",0)) or 0
            r["eps"]           = r["eps_growth"]
            r["rev"]           = r["rev_growth"]
            r["inst"]          = si.get("inst_own", 0)
            r["short_float"]   = si.get("short_float", None)
            r["sf"]            = si.get("short_float", None)
            r["eps_accel"]     = si.get("eps_accelerating", False)
            r["rev_accel"]     = si.get("rev_accelerating", False)
            r["eps_accel_3q"]  = si.get("eps_accel_3q", False)
            r["eps_positive"]  = si.get("eps_positive", None)
            r["annual_stable"] = si.get("annual_eps_stable", False)
            r["annual_cagr"]   = si.get("annual_eps_growth_3y", None)
            r["float_cat"]     = si.get("float_category", None)
            r["buyback"]       = si.get("buyback_trend", False)
            r["q_data"]        = q_avail
            r["canslim"]       = canslim_ok(r["ticker"], r["rs_pctile"], cache)

    # Post-enrichment: apply fundamental filters to O'Neil and Minervini
    def passes_oneil_fundamentals(r):
        si = cache.get(r["ticker"], {})
        q_avail = si.get("q_data_available", False)
        eps_g = (si.get("mrq_eps_growth") if q_avail else si.get("eps_growth",0)) or 0
        rev_g = (si.get("mrq_rev_growth") if q_avail else si.get("rev_growth",0)) or 0
        if q_avail and not si.get("eps_positive", True): return False
        return eps_g >= 20 and rev_g >= 15

    def passes_minervini_fundamentals(r):
        si = cache.get(r["ticker"], {})
        q_avail = si.get("q_data_available", False)
        eps_g = (si.get("mrq_eps_growth") if q_avail else si.get("eps_growth",0)) or 0
        rev_g = (si.get("mrq_rev_growth") if q_avail else si.get("rev_growth",0)) or 0
        if q_avail and not si.get("eps_positive", True): return False
        return eps_g >= 25 or rev_g >= 20

    o_hits  = [r for r in o_hits if passes_oneil_fundamentals(r)]
    m_hits  = [r for r in m_hits if passes_minervini_fundamentals(r)]
    print(f"     Post-filter: Q:{len(q_hits)} MIN:{len(m_hits)} ONEIL:{len(o_hits)} WEIN:{len(w_hits)}")
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
        "qullamaggie":   q_hits,
        "minervini":     m_hits,
        "oneil":         o_hits,
        "weinstein":     w_hits,
        "bearish":       swing_bear,
        "choppy":        swing_chop,
        "longterm":      lt_hits,
    }
    total = len(q_hits)+len(m_hits)+len(o_hits)+len(w_hits)+len(swing_bear)+len(swing_chop)+len(lt_hits)
    with open(CFG["output_file"], "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓  {total} setups → {CFG['output_file']}")
    print(f"     Q:{len(q_hits)} MIN:{len(m_hits)} ONEIL:{len(o_hits)} WEIN:{len(w_hits)} "
          f"Bear:{len(swing_bear)} Choppy:{len(swing_chop)} LT:{len(lt_hits)}")
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
