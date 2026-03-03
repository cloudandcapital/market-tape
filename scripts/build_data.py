#!/usr/bin/env python3
"""Build static JSON + chart assets for CLOUD & CAPITAL - Market Tape."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BENCHMARK = "SPY"

# Keep display symbols stable in UI while mapping to yfinance sources.
DATA_SOURCE_TICKER: dict[str, str] = {
    "DXY": "DX-Y.NYB",
}

# Override names for known market tape instruments.
NAME_OVERRIDES: dict[str, str] = {
    "SPY": "S&P 500 ETF",
    "TLT": "20+ Year Treasury Bond ETF",
    "IEF": "7-10 Year Treasury Bond ETF",
    "HYG": "iShares iBoxx High Yield Corp Bond ETF",
    "DXY": "US Dollar Index",
    "^VIX": "CBOE Volatility Index",
    "XLK": "Technology Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLE": "Energy Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR",
    "XLY": "Consumer Discretionary Select Sector SPDR",
    "XLP": "Consumer Staples Select Sector SPDR",
    "XLU": "Utilities Select Sector SPDR",
    "XLB": "Materials Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
    "XLC": "Communication Services Select Sector SPDR",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "IWF": "iShares Russell 1000 Growth ETF",
    "IWD": "iShares Russell 1000 Value ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "SMH": "VanEck Semiconductor ETF",
    "SOXX": "iShares Semiconductor ETF",
    "IGV": "iShares Expanded Tech-Software Sector ETF",
    "SKYY": "First Trust Cloud Computing ETF",
    "AIQ": "Global X Artificial Intelligence & Technology ETF",
    "PAVE": "Global X US Infrastructure Development ETF",
    "SPLV": "Invesco S&P 500 Low Volatility ETF",
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "USO": "United States Oil Fund",
    "DBC": "Invesco DB Commodity Index Tracking Fund",
    "VNQ": "Vanguard Real Estate ETF",
    "REET": "iShares Global REIT ETF",
    "TIP": "TIPS",
    "EFA": "iShares MSCI EAFE ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "EWZ": "iShares MSCI Brazil ETF",
    "INDA": "iShares MSCI India ETF",
    "FXI": "iShares China Large-Cap ETF",
    "VGK": "Vanguard FTSE Europe ETF",
    "EWC": "iShares MSCI Canada ETF",
}

LEVERAGED_MAP: dict[str, tuple[str, str]] = {
    "SPY": ("SPXL", "SPXS"),
    "QQQ": ("TQQQ", "SQQQ"),
    "SOXX": ("SOXL", "SOXS"),
    "SMH": ("SOXL", "SOXS"),
    "TLT": ("TMF", "TMV"),
    "XLE": ("ERX", "ERY"),
    "XLF": ("FAS", "FAZ"),
    "IWM": ("TNA", "TZA"),
}

GROUPS: dict[str, list[str]] = {
    "Macro Regime": ["SPY", "TLT", "IEF", "HYG", "DXY", "^VIX"],
    "US Sectors": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    "Style & Size": ["QQQ", "IWM", "IWF", "IWD", "MTUM", "USMV"],
    "AI & Infrastructure": ["SMH", "SOXX", "IGV", "SKYY", "AIQ", "PAVE"],
    "Momentum & Leadership": ["NVDA", "AVGO", "SMCI", "AMD", "META", "TSLA", "LLY", "CELH", "ARM", "COST"],
    "Defensives & Staples": ["XLP", "XLU", "XLV", "USMV", "SPLV", "BIL"],
    "Commodities & Real Assets": ["GLD", "SLV", "USO", "DBC", "VNQ", "REET", "TIP"],
    "Global / Countries": ["EFA", "EEM", "EWJ", "EWZ", "INDA", "FXI", "VGK", "EWC"],
}

SECTOR_LEADER_TICKERS = GROUPS["US Sectors"]
COUNTRY_LEADER_TICKERS = ["EWJ", "EWZ", "INDA", "FXI", "VGK", "EWC"]

COLUMN_GUIDE = [
    {"key": "ticker", "label": "Ticker", "description": "Trading symbol."},
    {"key": "trend_grade", "label": "Grade", "description": "Trend structure grade: A (strong), B (mixed), C (weak)."},
    {"key": "name", "label": "Name", "description": "Instrument display name with yfinance/override fallback."},
    {"key": "last", "label": "Last", "description": "Latest daily close."},
    {"key": "intra_pct", "label": "Intra%", "description": "Latest intraday move vs prior close."},
    {"key": "d1_pct", "label": "1D%", "description": "Close-to-close 1-day return."},
    {"key": "d5_pct", "label": "5D%", "description": "5-session return."},
    {"key": "d20_pct", "label": "20D%", "description": "20-session return."},
    {"key": "atr_pct", "label": "ATR%", "description": "14-day ATR as percent of price."},
    {"key": "dist50_atr", "label": "Dist50/ATR", "description": "Distance from SMA50 in ATR units."},
    {"key": "rs1m", "label": "RS1M", "description": "1-month volatility-adjusted relative strength vs SPY."},
    {"key": "letf", "label": "LETF", "description": "Mapped leveraged long/short ETF pair."},
    {"key": "mini_rs_chart", "label": "Mini RS", "description": "90-session mini relative-strength sparkline vs SPY."},
]

EVENT_TEMPLATES = [
    {"event": "Global PMI pulse check", "impact": "Medium", "region": "Global"},
    {"event": "US labor market watch", "impact": "High", "region": "US"},
    {"event": "US inflation watchlist", "impact": "High", "region": "US"},
    {"event": "Rates and liquidity check", "impact": "Medium", "region": "US"},
    {"event": "Consumer demand read-through", "impact": "Medium", "region": "US"},
    {"event": "Central bank communication window", "impact": "High", "region": "Global"},
    {"event": "Commodities inventory update", "impact": "Medium", "region": "Global"},
    {"event": "Global risk sentiment reset", "impact": "Medium", "region": "Global"},
]

GENERIC_NAME_PREFIXES = [
    "iShares",
    "Invesco",
    "Vanguard",
    "SPDR",
    "SPDR®",
    "Global X",
    "First Trust",
    "VanEck",
]

GENERIC_NAME_WORDS = [
    "ETF",
    "Trust",
    "Fund",
    "Index",
    "Portfolio",
]


def source_ticker(display_ticker: str) -> str:
    return DATA_SOURCE_TICKER.get(display_ticker, display_ticker)


def normalize_filename(ticker: str) -> str:
    base = ticker.replace("^", "INDEX_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def to_utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any, digits: int = 3, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return default
    if isinstance(value, (int, np.integer)):
        return float(value)
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(fval):
        return default
    return round(fval, digits)


def sanitize_name(name: Any, fallback: str) -> str:
    if isinstance(name, str):
        cleaned = " ".join(name.strip().split())
        if cleaned:
            return cleaned
    return fallback


def build_short_name(name: str, ticker: str, max_chars: int = 24) -> str:
    """
    Build compact table labels while preserving the full name for tooltips.

    Rules:
    - strip known issuer prefixes
    - strip generic fund words
    - collapse whitespace
    - truncate to max length with ellipsis
    - fallback to ticker if empty
    """

    short = sanitize_name(name, ticker).replace("®", "")

    for prefix in GENERIC_NAME_PREFIXES:
        pattern = r"^\s*" + re.escape(prefix.replace("®", "")) + r"\s+"
        short = re.sub(pattern, "", short, flags=re.IGNORECASE)

    word_pattern = r"\b(?:%s)\b" % "|".join(re.escape(word) for word in GENERIC_NAME_WORDS)
    short = re.sub(word_pattern, "", short, flags=re.IGNORECASE)
    short = re.sub(r"\s{2,}", " ", short).strip(" ,.-")

    if not short:
        short = ticker

    if len(short) > max_chars:
        trunc_len = max(1, max_chars - 1)
        short = short[:trunc_len].rstrip() + "…"

    return short


def build_universe() -> tuple[list[str], dict[str, str]]:
    seen: set[str] = set()
    source_tickers: list[str] = []
    source_to_display: dict[str, str] = {}
    for tickers in GROUPS.values():
        for display in tickers:
            source = source_ticker(display)
            source_to_display[source] = display
            if source not in seen:
                source_tickers.append(source)
                seen.add(source)

    benchmark_source = source_ticker(BENCHMARK)
    source_to_display[benchmark_source] = BENCHMARK
    if benchmark_source not in seen:
        source_tickers.append(benchmark_source)

    return source_tickers, source_to_display


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    copy = frame.copy()
    copy.columns = [str(col).title() for col in copy.columns]
    return copy.sort_index()


def download_history(tickers: list[str], *, period: str, interval: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    history = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    frames: dict[str, pd.DataFrame] = {}
    if not history.empty:
        if isinstance(history.columns, pd.MultiIndex):
            available = set(history.columns.get_level_values(0))
            for ticker in tickers:
                if ticker not in available:
                    continue
                ticker_frame = history[ticker].dropna(how="all")
                if not ticker_frame.empty:
                    frames[ticker] = _standardize_frame(ticker_frame)
        else:
            frames[tickers[0]] = _standardize_frame(history.dropna(how="all"))

    missing = [ticker for ticker in tickers if ticker not in frames or frames[ticker].empty]
    for ticker in missing:
        fallback = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if fallback is None or fallback.empty:
            continue
        frames[ticker] = _standardize_frame(fallback.dropna(how="all"))

    return frames


def fetch_name_map(source_tickers: list[str], source_to_display: dict[str, str]) -> dict[str, str]:
    """Resolve display names with yfinance name fields, then manual overrides, then ticker."""

    name_map: dict[str, str] = {}
    for source in source_tickers:
        display = source_to_display.get(source, source)
        yf_name = None
        try:
            info = yf.Ticker(source).get_info()
        except Exception:
            info = {}

        if isinstance(info, dict):
            yf_name = info.get("longName") or info.get("shortName")

        override_name = NAME_OVERRIDES.get(display)
        fallback = override_name or display
        resolved = sanitize_name(yf_name, fallback)

        # Final hard guarantee against blank names.
        if not resolved:
            resolved = display
        name_map[display] = resolved

    return name_map


def pct_return(close: pd.Series, periods: int) -> float | None:
    if close.empty or len(close) <= periods:
        return None
    start = close.iloc[-(periods + 1)]
    end = close.iloc[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start == 0:
        return None
    return (float(end) / float(start) - 1.0) * 100.0


def compute_atr(frame: pd.DataFrame, window: int = 14) -> float | None:
    if not {"High", "Low", "Close"}.issubset(set(frame.columns)):
        return None

    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)
    if close.empty:
        return None

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(window=window, min_periods=window).mean()
    if atr_series.empty or not np.isfinite(atr_series.iloc[-1]):
        return None
    return float(atr_series.iloc[-1])


def compute_vol_adjusted_rs(close: pd.Series, spy_close: pd.Series, lookback: int = 20) -> float | None:
    aligned = pd.concat([close.rename("asset"), spy_close.rename("spy")], axis=1).dropna()
    if len(aligned) <= lookback:
        return None

    asset = aligned["asset"]
    spy = aligned["spy"]

    asset_ret = asset.iloc[-1] / asset.iloc[-(lookback + 1)] - 1.0
    spy_ret = spy.iloc[-1] / spy.iloc[-(lookback + 1)] - 1.0

    daily = asset.pct_change().dropna().tail(lookback)
    if daily.empty:
        return None

    realized_vol = float(daily.std(ddof=0) * math.sqrt(lookback))
    if realized_vol <= 0 or not math.isfinite(realized_vol):
        return None

    return float((asset_ret - spy_ret) / realized_vol)


def compute_trend_grade(close: pd.Series) -> tuple[str, float | None, float | None, float | None]:
    """Return (grade, ema10, ema20, sma50) using requested rules."""

    if close.empty:
        return "B", None, None, None

    ema10 = close.ewm(span=10, adjust=False).mean().iloc[-1] if len(close) >= 10 else np.nan
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1] if len(close) >= 20 else np.nan
    sma50 = close.rolling(window=50, min_periods=50).mean().iloc[-1] if len(close) >= 50 else np.nan
    last = float(close.iloc[-1])

    if np.isfinite(ema10) and np.isfinite(ema20) and np.isfinite(sma50):
        if ema10 > ema20 > sma50 and last > sma50:
            return "A", float(ema10), float(ema20), float(sma50)
        if ema10 < ema20 and last < sma50:
            return "C", float(ema10), float(ema20), float(sma50)

    return "B", float(ema10) if np.isfinite(ema10) else None, float(ema20) if np.isfinite(ema20) else None, float(sma50) if np.isfinite(sma50) else None


def make_rs_chart(ticker: str, close: pd.Series, spy_close: pd.Series, output_dir: Path) -> str:
    aligned = pd.concat([close.rename("asset"), spy_close.rename("spy")], axis=1).dropna().tail(90)
    if len(aligned) < 20:
        return ""

    rel = (aligned["asset"] / aligned["asset"].iloc[0]) / (aligned["spy"] / aligned["spy"].iloc[0])
    if rel.empty:
        return ""

    color = "#22c55e" if rel.iloc[-1] >= 1.0 else "#ef4444"
    y_min = float(rel.min())
    y_max = float(rel.max())
    spread = y_max - y_min
    pad = spread * 0.25 if spread > 0 else 0.03

    fig, ax = plt.subplots(figsize=(2.6, 0.92), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.plot(rel.values, linewidth=1.8, color=color)
    ax.axhline(1.0, color="#94a3b8", linewidth=0.8, alpha=0.55)
    ax.set_xlim(0, max(1, len(rel) - 1))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.axis("off")

    filename = f"{normalize_filename(ticker)}.png"
    path = output_dir / filename
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return f"data/mini_rs/{filename}"


def build_intraday_last(intraday_frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    latest: dict[str, float] = {}
    for ticker, frame in intraday_frames.items():
        if "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if close.empty:
            continue
        latest[ticker] = float(close.iloc[-1])
    return latest


def empty_row(ticker: str, name: str) -> dict[str, Any]:
    long_short = LEVERAGED_MAP.get(ticker)
    full_name = sanitize_name(name, ticker)
    return {
        "ticker": ticker,
        "name": full_name,
        "short_name": build_short_name(full_name, ticker),
        "last": 0.0,
        "intra_pct": 0.0,
        "d1_pct": 0.0,
        "d5_pct": 0.0,
        "d20_pct": 0.0,
        "atr_pct": 0.0,
        "dist50_atr": 0.0,
        "rs1m": 0.0,
        "trend_grade": "B",
        "leveraged": {
            "long": long_short[0] if long_short else None,
            "short": long_short[1] if long_short else None,
        },
        "mini_rs_chart": "",
    }


def build_row(
    ticker: str,
    name: str,
    daily_frames: dict[str, pd.DataFrame],
    intraday_last: dict[str, float],
    spy_close: pd.Series,
    chart_dir: Path,
) -> dict[str, Any]:
    source = source_ticker(ticker)
    frame = daily_frames.get(source)

    row = empty_row(ticker, name)
    if frame is None or frame.empty or "Close" not in frame:
        return row

    close = frame["Close"].dropna()
    if close.empty:
        return row

    last_close = float(close.iloc[-1])
    d1 = pct_return(close, 1)
    d5 = pct_return(close, 5)
    d20 = pct_return(close, 20)

    prev_close = float(close.iloc[-2]) if len(close) > 1 else None
    intra_price = intraday_last.get(source)
    if intra_price is not None and prev_close and prev_close != 0:
        intra_pct = (intra_price / prev_close - 1.0) * 100.0
    else:
        intra_pct = d1

    atr = compute_atr(frame, window=14)
    trend_grade, _, _, sma50 = compute_trend_grade(close)

    dist50_atr = 0.0
    if atr is not None and atr > 0 and sma50 is not None:
        dist50_atr = (last_close - float(sma50)) / atr

    row.update(
        {
            "last": safe_float(last_close, digits=2, default=0.0),
            "intra_pct": safe_float(intra_pct, digits=2, default=0.0),
            "d1_pct": safe_float(d1, digits=2, default=0.0),
            "d5_pct": safe_float(d5, digits=2, default=0.0),
            "d20_pct": safe_float(d20, digits=2, default=0.0),
            "atr_pct": safe_float((atr / last_close) * 100.0 if atr and last_close else 0.0, digits=2, default=0.0),
            "dist50_atr": safe_float(dist50_atr, digits=2, default=0.0),
            "rs1m": safe_float(compute_vol_adjusted_rs(close, spy_close, lookback=20), digits=3, default=0.0),
            "trend_grade": trend_grade,
            "mini_rs_chart": make_rs_chart(ticker, close, spy_close, chart_dir),
        }
    )

    return row


def top_n_leaders(rows_by_ticker: dict[str, dict[str, Any]], tickers: list[str], limit: int = 3) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for ticker in tickers:
        row = rows_by_ticker.get(ticker)
        if not row:
            continue
        ranked.append({"ticker": ticker, "name": row["name"], "rs1m": row["rs1m"]})
    ranked.sort(key=lambda item: item["rs1m"], reverse=True)
    return ranked[:limit]


def next_business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    probe = start
    while len(days) < count:
        if probe.weekday() < 5:
            days.append(probe)
        probe += timedelta(days=1)
    return days


def build_events(now_utc: datetime) -> dict[str, Any]:
    business_dates = next_business_days(now_utc.date(), len(EVENT_TEMPLATES))
    events: list[dict[str, Any]] = []
    for event_date, template in zip(business_dates, EVENT_TEMPLATES):
        events.append(
            {
                "date": event_date.isoformat(),
                "event": template["event"],
                "impact": template["impact"],
                "region": template["region"],
                "note": "Template macro watch item. Replace with your preferred live calendar source if needed.",
            }
        )
    return {"generated_at_utc": to_utc_iso(now_utc), "events": events}


def parse_event_date(date_str: str) -> date | None:
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return None


def load_manual_events(output_dir: Path) -> list[dict[str, Any]]:
    manual_path = output_dir / "manual_events.json"
    if not manual_path.exists():
        return []

    try:
        payload = json.loads(manual_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    source_events = payload.get("events")
    if not isinstance(source_events, list):
        return []

    manual_events: list[dict[str, Any]] = []
    for item in source_events:
        if not isinstance(item, dict):
            continue
        event_date = str(item.get("date", "")).strip()
        if not parse_event_date(event_date):
            continue
        event_name = str(item.get("event", "")).strip()
        if not event_name:
            continue
        manual_events.append(
            {
                "date": event_date,
                "event": event_name,
                "impact": str(item.get("impact", "Medium")).strip() or "Medium",
                "region": str(item.get("region", "Global")).strip() or "Global",
            }
        )
    return manual_events


def build_merged_events(now_utc: datetime, output_dir: Path) -> dict[str, Any]:
    template_payload = build_events(now_utc)
    merged_events = list(template_payload["events"]) + load_manual_events(output_dir)
    merged_events.sort(key=lambda event: (parse_event_date(str(event.get("date", ""))) or date.max, str(event.get("event", ""))))
    return {"generated_at_utc": to_utc_iso(now_utc), "events": merged_events}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_snapshot_rows(groups_payload: list[dict[str, Any]]) -> None:
    required_keys = {
        "ticker",
        "name",
        "short_name",
        "last",
        "intra_pct",
        "d1_pct",
        "d5_pct",
        "d20_pct",
        "atr_pct",
        "dist50_atr",
        "rs1m",
        "trend_grade",
        "leveraged",
    }

    for group in groups_payload:
        for row in group["rows"]:
            missing = required_keys - set(row.keys())
            if missing:
                raise ValueError(f"Row for {row.get('ticker', 'UNKNOWN')} missing keys: {sorted(missing)}")
            if not str(row.get("name", "")).strip():
                raise ValueError(f"Row for {row.get('ticker', 'UNKNOWN')} has blank name.")
            if row["trend_grade"] not in {"A", "B", "C"}:
                raise ValueError(f"Row for {row['ticker']} has invalid trend grade: {row['trend_grade']}")
            leveraged = row.get("leveraged")
            if not isinstance(leveraged, dict) or not {"long", "short"}.issubset(leveraged.keys()):
                raise ValueError(f"Row for {row['ticker']} has invalid leveraged payload.")


def finite_metric(value: Any) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric):
        return None
    return metric


def median_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [finite_metric(row.get(key)) for row in rows]
    valid = [val for val in values if val is not None]
    if not valid:
        return None
    return float(np.median(valid))


def classify_trend(median_value: float | None, *, upper: float, lower: float) -> str:
    if median_value is None:
        return "Sideways"
    if median_value > upper:
        return "Up"
    if median_value < lower:
        return "Down"
    return "Sideways"


def find_vix_last(groups_payload: list[dict[str, Any]]) -> float | None:
    for group in groups_payload:
        if group.get("name") != "Macro Regime":
            continue
        for row in group.get("rows", []):
            ticker = str(row.get("ticker", "")).upper()
            if ticker in {"^VIX", "VIX"}:
                return finite_metric(row.get("last"))
    return None


def build_market_status(groups_payload: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build deterministic high-level market status from existing row metrics.

    Rules:
    - exposure: % of instruments with RS1M > 0
    - trend: median 20D/5D/1D return buckets
    - risk:
      - volatility from VIX if available
      - sentiment from % instruments with positive intraday move
      - momentum from trend-grade balance (#A vs #C)
    """

    rows = [row for group in groups_payload for row in group.get("rows", [])]
    row_count = len(rows)

    # Exposure level and guidance.
    positive_rs = 0
    for row in rows:
        rs1m = finite_metric(row.get("rs1m"))
        if rs1m is not None and rs1m > 0:
            positive_rs += 1
    exposure_ratio = (positive_rs / row_count * 100.0) if row_count else 0.0
    exposure_level = int(max(0, min(100, round(exposure_ratio))))
    if exposure_level >= 70:
        exposure_guidance = "Risk-On"
    elif exposure_level >= 40:
        exposure_guidance = "Hold"
    else:
        exposure_guidance = "Defensive"

    # Trend by median return horizon.
    median_d20 = median_metric(rows, "d20_pct")
    median_d5 = median_metric(rows, "d5_pct")
    median_d1 = median_metric(rows, "d1_pct")
    trend = {
        "long_term": classify_trend(median_d20, upper=2.0, lower=-1.0),
        "intermediate_term": classify_trend(median_d5, upper=1.0, lower=-0.5),
        "short_term": classify_trend(median_d1, upper=0.5, lower=-0.5),
    }

    # Volatility risk from VIX level.
    vix_last = find_vix_last(groups_payload)
    if vix_last is None:
        volatility = "Neutral"
    elif vix_last <= 15:
        volatility = "Low"
    elif vix_last <= 22:
        volatility = "Neutral"
    else:
        volatility = "High"

    # Sentiment from intraday breadth.
    intra_values = [finite_metric(row.get("intra_pct")) for row in rows]
    valid_intra = [value for value in intra_values if value is not None]
    if not valid_intra:
        sentiment = "Neutral"
    else:
        positive_intra_pct = sum(1 for value in valid_intra if value > 0) / len(valid_intra) * 100.0
        if positive_intra_pct > 60:
            sentiment = "Bullish"
        elif positive_intra_pct < 40:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

    # Momentum from trend-grade distribution.
    grade_a = sum(1 for row in rows if str(row.get("trend_grade", "")).upper() == "A")
    grade_c = sum(1 for row in rows if str(row.get("trend_grade", "")).upper() == "C")
    if grade_a > grade_c:
        momentum = "Positive"
    elif grade_c > grade_a:
        momentum = "Negative"
    else:
        momentum = "Neutral"

    return {
        "exposure": {
            "level": exposure_level,
            "guidance": exposure_guidance,
        },
        "trend": trend,
        "risk": {
            "volatility": volatility,
            "sentiment": sentiment,
            "momentum": momentum,
        },
    }


def validate_market_status(status: dict[str, Any]) -> None:
    if not isinstance(status, dict):
        raise ValueError("status must be a dictionary")

    exposure = status.get("exposure", {})
    trend = status.get("trend", {})
    risk = status.get("risk", {})
    if not isinstance(exposure, dict) or not isinstance(trend, dict) or not isinstance(risk, dict):
        raise ValueError("status payload sections are malformed")

    level = exposure.get("level")
    guidance = exposure.get("guidance")
    if not isinstance(level, int) or level < 0 or level > 100:
        raise ValueError("status.exposure.level must be int in [0,100]")
    if guidance not in {"Risk-On", "Hold", "Defensive"}:
        raise ValueError("status.exposure.guidance must be Risk-On/Hold/Defensive")

    if trend.get("long_term") not in {"Up", "Sideways", "Down"}:
        raise ValueError("status.trend.long_term invalid")
    if trend.get("intermediate_term") not in {"Up", "Sideways", "Down"}:
        raise ValueError("status.trend.intermediate_term invalid")
    if trend.get("short_term") not in {"Up", "Sideways", "Down"}:
        raise ValueError("status.trend.short_term invalid")

    if risk.get("volatility") not in {"Low", "Neutral", "High"}:
        raise ValueError("status.risk.volatility invalid")
    if risk.get("sentiment") not in {"Bullish", "Neutral", "Bearish"}:
        raise ValueError("status.risk.sentiment invalid")
    if risk.get("momentum") not in {"Positive", "Neutral", "Negative"}:
        raise ValueError("status.risk.momentum invalid")


def build_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mini_chart_dir = output_dir / "mini_rs"
    mini_chart_dir.mkdir(parents=True, exist_ok=True)
    for old_chart in mini_chart_dir.glob("*.png"):
        old_chart.unlink(missing_ok=True)

    now_utc = datetime.now(timezone.utc)
    source_tickers, source_to_display = build_universe()

    daily_frames = download_history(source_tickers, period="12mo", interval="1d")
    intraday_frames = download_history(source_tickers, period="2d", interval="5m")
    intraday_last = build_intraday_last(intraday_frames)

    benchmark_source = source_ticker(BENCHMARK)
    benchmark_frame = daily_frames.get(benchmark_source)
    if benchmark_frame is None or benchmark_frame.empty or "Close" not in benchmark_frame:
        raise RuntimeError("Missing SPY history. Cannot compute relative strength metrics.")

    spy_close = benchmark_frame["Close"].dropna()
    if spy_close.empty:
        raise RuntimeError("SPY close series is empty. Cannot compute relative strength metrics.")

    name_map = fetch_name_map(source_tickers, source_to_display)

    groups_payload: list[dict[str, Any]] = []
    rows_by_ticker: dict[str, dict[str, Any]] = {}

    for group_name, tickers in GROUPS.items():
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            row = build_row(
                ticker=ticker,
                name=name_map.get(ticker, NAME_OVERRIDES.get(ticker, ticker)),
                daily_frames=daily_frames,
                intraday_last=intraday_last,
                spy_close=spy_close,
                chart_dir=mini_chart_dir,
            )
            rows.append(row)
            rows_by_ticker[ticker] = row
        groups_payload.append({"name": group_name, "rows": rows})

    validate_snapshot_rows(groups_payload)

    leaders = {
        "sectors": top_n_leaders(rows_by_ticker, SECTOR_LEADER_TICKERS, limit=3),
        "countries": top_n_leaders(rows_by_ticker, COUNTRY_LEADER_TICKERS, limit=3),
    }
    status = build_market_status(groups_payload)
    validate_market_status(status)

    snapshot = {
        "generated_at_utc": to_utc_iso(now_utc),
        "benchmark": BENCHMARK,
        "columns": COLUMN_GUIDE,
        "groups": groups_payload,
    }

    events = build_merged_events(now_utc, output_dir)
    meta = {
        "app": "CLOUD & CAPITAL - Market Tape",
        "generated_at_utc": to_utc_iso(now_utc),
        "benchmark": BENCHMARK,
        "group_count": len(GROUPS),
        "instrument_count": len(source_tickers),
        "leaders": leaders,
        "status": status,
        "data_files": {
            "snapshot": "data/snapshot.json",
            "events": "data/events.json",
            "meta": "data/meta.json",
        },
    }

    write_json(output_dir / "snapshot.json", snapshot)
    write_json(output_dir / "events.json", events)
    write_json(output_dir / "meta.json", meta)

    print(f"Built data at {output_dir.resolve()}")
    print(f"- snapshot rows: {sum(len(group['rows']) for group in groups_payload)}")
    print(f"- RS mini charts: {len(list(mini_chart_dir.glob('*.png')))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build market dashboard data payloads.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where snapshot.json/events.json/meta.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_data(Path(args.output_dir))


if __name__ == "__main__":
    main()
