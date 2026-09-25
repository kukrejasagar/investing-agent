#!/usr/bin/env python3
"""Swing-trading technical report from daily OHLCV + Volume Delta CSVs.

Usage: python3 swing_report.py [data_dir] [output.md]

Stdlib only. Every number in the report is computed from the CSVs.
"""
import csv
import datetime as dt
import glob
import math
import os
import re
import statistics
import sys

TOL = 0.005          # ±0.5% zone tolerance
BOUNCE_WINDOW = 3    # trading days after the touch to resolve bounce vs. break
MIN_N = 2            # minimum touches to list a zone
RELIABLE_N = 5       # below this, flag as low sample size
MAX_ZONES = 4
SWING_K = 3          # swing low = lowest low within ±3 bars
DIR_LOOKBACK = 10    # price vs. delta direction window


# ---------- data ----------

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    bars = []
    for r in rows:
        bars.append({
            "date": dt.datetime.fromtimestamp(int(r["time"]), dt.timezone.utc).date(),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["Volume"]),
            "vd_high": float(r["Volume Delta (High"]),
            "vd_low": float(r["Volume Delta (Low)"]),
            "vd": float(r["Volume Delta (Close)"]),
        })
    bars.sort(key=lambda b: b["date"])
    return bars


def ticker_from(path):
    m = re.match(r"(?:[A-Z]+_)?([A-Z.]+)_1D", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


# ---------- indicators ----------

def sma(xs, n):
    out = [None] * len(xs)
    s = 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= n:
            s -= xs[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(xs, n):
    """EMA seeded with the SMA of the first n valid values."""
    out = [None] * len(xs)
    k = 2 / (n + 1)
    valid = [i for i, x in enumerate(xs) if x is not None]
    if len(valid) < n:
        return out
    start = valid[n - 1]
    out[start] = sum(xs[i] for i in valid[:n]) / n
    for i in range(start + 1, len(xs)):
        out[i] = xs[i] * k + out[i - 1] * (1 - k)
    return out


def rsi_wilder(closes, n=14):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def bollinger(closes, n=20, k=2):
    mid = sma(closes, n)
    up, lo = [None] * len(closes), [None] * len(closes)
    for i in range(n - 1, len(closes)):
        sd = statistics.pstdev(closes[i - n + 1:i + 1])  # population std dev
        up[i] = mid[i] + k * sd
        lo[i] = mid[i] - k * sd
    return mid, up, lo


def rolling_sum(xs, n):
    out = [None] * len(xs)
    for i in range(n - 1, len(xs)):
        out[i] = sum(xs[i - n + 1:i + 1])
    return out


def compute(bars):
    c = [b["close"] for b in bars]
    ind = {
        "sma20": sma(c, 20), "sma50": sma(c, 50), "sma100": sma(c, 100),
        "ema12": ema(c, 12), "ema26": ema(c, 26),
        "rsi": rsi_wilder(c, 14),
    }
    macd = [a - b if a is not None and b is not None else None
            for a, b in zip(ind["ema12"], ind["ema26"])]
    ind["macd"] = macd
    ind["signal"] = ema(macd, 9)
    ind["hist"] = [m - s if m is not None and s is not None else None
                   for m, s in zip(macd, ind["signal"])]
    ind["bb_mid"], ind["bb_up"], ind["bb_lo"] = bollinger(c)
    vd = [b["vd"] for b in bars]
    cum, s = [], 0.0
    for x in vd:
        s += x
        cum.append(s)
    ind["cum_vd"] = cum
    ind["vd10"] = rolling_sum(vd, DIR_LOOKBACK)
    ind["vd20"] = rolling_sum(vd, 20)
    return ind


# ---------- classification ----------

def trend_state(bars, ind):
    i = len(bars) - 1
    c = bars[i]["close"]
    s20, s50, s100 = ind["sma20"][i], ind["sma50"][i], ind["sma100"][i]
    e12, e26 = ind["ema12"][i], ind["ema26"][i]
    s50_prev = ind["sma50"][i - 10]
    # each check votes +1 / -1, or 0 when the two values are within ±TOL (flat)
    pairs = [(c, s20), (s20, s50), (s50, s100), (e12, e26), (s50, s50_prev)]
    score = sum(0 if abs(a / b - 1) <= TOL else (1 if a > b else -1) for a, b in pairs)
    if score >= 3:
        label = "up"
    elif score <= -3:
        label = "down"
    else:
        label = "sideways"
    return label, score


def macd_signal(ind):
    i = len(ind["macd"]) - 1
    m, s, h = ind["macd"][i], ind["signal"][i], ind["hist"][i]
    # most recent crossover
    cross_ago, cross_type = None, None
    for j in range(i, 0, -1):
        if ind["hist"][j] is None or ind["hist"][j - 1] is None:
            break
        if (ind["hist"][j] > 0) != (ind["hist"][j - 1] > 0):
            cross_ago = i - j
            cross_type = "bullish" if ind["hist"][j] > 0 else "bearish"
            break
    momentum = "rising" if h > ind["hist"][i - 1] else "falling"
    return m, s, h, cross_type, cross_ago, momentum


def rsi_label(r):
    if r >= 70:
        return "overbought"
    if r <= 30:
        return "oversold"
    return "neutral"


def bb_position(close, up, lo):
    pb = (close - lo) / (up - lo)
    if pb > 1:
        lab = "above upper band (overbought)"
    elif pb >= 0.8:
        lab = "near upper band"
    elif pb < 0:
        lab = "below lower band (oversold)"
    elif pb <= 0.2:
        lab = "near lower band"
    else:
        lab = "mid-band (neutral)"
    return pb, lab


def volume_confirmation(bars, ind):
    i = len(bars) - 1
    dp = bars[i]["close"] - bars[i - DIR_LOOKBACK]["close"]
    dv = ind["vd10"][i]
    confirmed = (dp > 0) == (dv > 0)
    # history of the flag over the last 20 sessions
    hist = []
    for j in range(i - 19, i + 1):
        p = bars[j]["close"] - bars[j - DIR_LOOKBACK]["close"]
        hist.append((p > 0) == (ind["vd10"][j] > 0))
    return dp, dv, confirmed, sum(hist), len(hist)


# ---------- support zones ----------

def candidates(bars, ind):
    """Return (price, source) candidate support levels from the whole series."""
    out = []
    n = len(bars)
    lows = [b["low"] for b in bars]
    # 1. swing lows (fractal: lowest low within ±SWING_K bars)
    for i in range(SWING_K, n - SWING_K):
        win = lows[i - SWING_K:i + SWING_K + 1]
        if lows[i] == min(win):
            out.append((lows[i], "swing low"))
    # 2. lower Bollinger Band touches
    for i in range(n):
        lo = ind["bb_lo"][i]
        if lo is not None and bars[i]["low"] <= lo * (1 + TOL):
            out.append((bars[i]["low"], "lower-BB touch"))
    # 3. SMA20/50/100 confluence (at least two within ±0.5% of each other)
    for i in range(n):
        vals = [ind[k][i] for k in ("sma20", "sma50", "sma100") if ind[k][i] is not None]
        best = None
        for a in range(len(vals)):
            grp = [v for v in vals if abs(v / vals[a] - 1) <= TOL]
            if len(grp) >= 2 and (best is None or len(grp) > len(best)):
                best = grp
        if best:
            out.append((sum(best) / len(best), "SMA confluence"))
    # 4. volume-delta absorption: heavy selling at the low that was bought back
    vdl = sorted(b["vd_low"] for b in bars)
    thresh = vdl[int(len(vdl) * 0.2)]  # bottom 20% of intrabar low-delta
    for b in bars:
        rng = b["high"] - b["low"]
        if rng <= 0 or b["vd_low"] >= 0 or b["vd_low"] > thresh:
            continue
        recovered = (b["vd"] - b["vd_low"]) >= 0.5 * abs(b["vd_low"])
        upper_half = (b["close"] - b["low"]) / rng >= 0.5
        if recovered and upper_half:
            out.append((b["low"], "VD absorption"))
    return out


def cluster(cands):
    """Greedy clustering: a candidate joins a zone if within ±TOL of its centre."""
    zones = []
    for price, src in sorted(cands):
        if zones and abs(price / zones[-1]["center"] - 1) <= TOL:
            z = zones[-1]
            z["prices"].append(price)
            z["sources"][src] = z["sources"].get(src, 0) + 1
            z["center"] = sum(z["prices"]) / len(z["prices"])
        else:
            zones.append({"center": price, "prices": [price], "sources": {src: 1}})
    return zones


def backtest_zone(bars, center):
    """Count touches from above and whether each bounced within BOUNCE_WINDOW days.

    Touch: prior close above the zone top and today's low reaches the zone top
    (includes gaps straight through the zone). Resolution: the first close
    outside the zone on the touch day or the next BOUNCE_WINDOW days —
    above the top = reversal, below the bottom = break. Still inside after
    the window = unresolved (counted in N, not as a reversal).
    """
    top, bot = center * (1 + TOL), center * (1 - TOL)
    res = {"n": 0, "bounce": 0, "break": 0, "unresolved": 0, "pending": 0, "dates": []}
    i = 1
    while i < len(bars):
        if bars[i - 1]["close"] > top and bars[i]["low"] <= top:
            end = i + BOUNCE_WINDOW
            if end >= len(bars):
                res["pending"] += 1
                break
            outcome = "unresolved"
            for j in range(i, end + 1):
                if bars[j]["close"] < bot:
                    outcome = "break"
                    break
                if bars[j]["close"] > top:
                    outcome = "bounce"
                    break
            res["n"] += 1
            res[outcome] += 1
            res["dates"].append((bars[i]["date"], outcome))
            # skip until price has closed back above the zone to avoid double counting
            # (bar k's close above the zone then serves as the next touch's prior close)
            k = i
            while k < len(bars) and bars[k]["close"] <= top:
                k += 1
            i = k + 1
            continue
        i += 1
    res["rate"] = res["bounce"] / res["n"] if res["n"] else None
    return res


def support_zones(bars, ind):
    last = bars[-1]["close"]
    zones = [z for z in cluster(candidates(bars, ind)) if z["center"] * (1 + TOL) < last]
    for z in zones:
        z["bt"] = backtest_zone(bars, z["center"])
    qualified = [z for z in zones if z["bt"]["n"] >= MIN_N]
    qualified.sort(key=lambda z: last - z["center"])  # nearest first
    # drop zones whose ±TOL band overlaps an already-listed (nearer) zone, so
    # listed zones never share touches; keep the one with more touches
    picked = []
    for z in qualified:
        clash = next((p for p in picked if abs(z["center"] / p["center"] - 1) <= 2 * TOL), None)
        if clash is None:
            picked.append(z)
        elif z["bt"]["n"] > clash["bt"]["n"]:
            picked[picked.index(clash)] = z
    picked.sort(key=lambda z: last - z["center"])
    return picked[:MAX_ZONES]


# ---------- report ----------

def fmt(x, d=2):
    return "n/a" if x is None else f"{x:,.{d}f}"


def pct(x):
    return f"{x * 100:.0f}%"


def rate_str(bt):
    s = f"{pct(bt['rate'])} ({bt['n']})"
    return s + " ⚠️" if bt["n"] < RELIABLE_N else s


def analyse(path):
    bars = load(path)
    ind = compute(bars)
    i = len(bars) - 1
    t, tscore = trend_state(bars, ind)
    pb, pb_lab = bb_position(bars[i]["close"], ind["bb_up"][i], ind["bb_lo"][i])
    return {
        "ticker": ticker_from(path), "bars": bars, "ind": ind,
        "trend": t, "tscore": tscore, "macd": macd_signal(ind),
        "rsi": ind["rsi"][i], "pb": pb, "pb_lab": pb_lab,
        "vol": volume_confirmation(bars, ind), "zones": support_zones(bars, ind),
    }


def report(results):
    L = []
    end = max(r["bars"][-1]["date"] for r in results)
    L.append("# Swing-Trading Technical Report")
    L.append("")
    L.append(f"Data through **{end}** · generated by `swing_report.py` from the CSVs in `data/` "
             "(all values computed, not estimated).")
    L.append("")
    L.append("## Cross-ticker summary")
    L.append("")
    L.append("| Ticker | Trend | RSI | Bollinger Position | Volume Confirmation | Nearest Support | Its Reversal Rate (N) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        last = r["bars"][-1]["close"]
        if r["zones"]:
            z = r["zones"][0]
            sup = f"{fmt(z['center'])} ({(z['center'] / last - 1) * 100:+.1f}%)"
            rr = rate_str(z["bt"])
        else:
            sup, rr = "none qualified", "—"
        L.append(f"| **{r['ticker']}** | {r['trend']} | {r['rsi']:.1f} ({rsi_label(r['rsi'])}) | "
                 f"%B {r['pb']:.2f} — {r['pb_lab']} | {'confirmed' if r['vol'][2] else 'diverging'} | "
                 f"{sup} | {rr} |")
    L.append("")
    L.append("⚠️ = N < 5: low sample size — directional signal only, not a reliable rate.")
    L.append("")

    for r in results:
        b, ind = r["bars"], r["ind"]
        i = len(b) - 1
        c = b[i]["close"]
        L.append(f"## {r['ticker']} — close {fmt(c)} ({b[i]['date']})")
        L.append("")
        L.append(f"Series: {b[0]['date']} → {b[-1]['date']} ({len(b)} sessions)")
        L.append("")

        # 1. trend
        L.append("### 1. Trend state")
        L.append("")
        L.append("| SMA20 | SMA50 | SMA100 | EMA12 | EMA26 |")
        L.append("|---|---|---|---|---|")
        L.append("| " + " | ".join(fmt(ind[k][i]) for k in ("sma20", "sma50", "sma100", "ema12", "ema26")) + " |")
        L.append("")
        order = sorted([("Price", c), ("SMA20", ind["sma20"][i]), ("SMA50", ind["sma50"][i]),
                        ("SMA100", ind["sma100"][i])], key=lambda x: -x[1])
        stack = " > ".join(n for n, _ in order)
        if [n for n, _ in order] == ["Price", "SMA20", "SMA50", "SMA100"]:
            align = "fully bullish stack"
        elif [n for n, _ in order] == ["SMA100", "SMA50", "SMA20", "Price"]:
            align = "fully bearish stack"
        else:
            align = "mixed alignment"
        s50_slope = (ind["sma50"][i] / ind["sma50"][i - 10] - 1) * 100
        L.append(f"- **Alignment:** {stack} → {align}. EMA12 {'>' if ind['ema12'][i] > ind['ema26'][i] else '<'} EMA26. "
                 f"SMA50 10-day slope {s50_slope:+.2f}%.")
        m, s, h, ct, ca, mom = r["macd"]
        cross = f"last crossover {ct} {ca} session(s) ago" if ct else "no crossover in range"
        L.append(f"- **MACD (12,26,9):** MACD {m:+.2f}, signal {s:+.2f}, histogram {h:+.2f} ({mom}); "
                 f"{'above' if m > s else 'below'} signal line, {'above' if m > 0 else 'below'} zero; {cross}.")
        L.append(f"- **Trend verdict:** **{r['trend']}** (score {r['tscore']:+d} on 5 checks: price vs SMA20, "
                 "SMA20 vs SMA50, SMA50 vs SMA100, EMA12 vs EMA26, SMA50 slope; within 0.5% = flat/0; ≥+3 up, ≤−3 down).")
        L.append("")

        # 2. RSI + BB
        L.append("### 2. RSI and Bollinger position")
        L.append("")
        L.append(f"- **RSI(14, Wilder):** {r['rsi']:.1f} → {rsi_label(r['rsi'])}. "
                 f"5 sessions ago: {ind['rsi'][i - 5]:.1f}.")
        L.append(f"- **Bollinger (20, 2σ population):** upper {fmt(ind['bb_up'][i])}, mid {fmt(ind['bb_mid'][i])}, "
                 f"lower {fmt(ind['bb_lo'][i])}; bandwidth {(ind['bb_up'][i] - ind['bb_lo'][i]) / ind['bb_mid'][i] * 100:.1f}%. "
                 f"%B {r['pb']:.2f} → {r['pb_lab']}.")
        L.append("")

        # 3. volume delta
        dp, dv, conf, nconf, nh = r["vol"]
        L.append("### 3. Volume Delta confirmation")
        L.append("")
        L.append(f"- **10-day price change:** {dp:+.2f} ({dp / b[i - DIR_LOOKBACK]['close'] * 100:+.1f}%) · "
                 f"**10-day rolling delta:** {dv:+,.0f} → **{'confirmed' if conf else 'diverging'}** "
                 f"({'price and delta agree' if conf else 'price and delta point opposite ways'}).")
        L.append(f"- 20-day rolling delta {ind['vd20'][i]:+,.0f}; cumulative delta over the series "
                 f"{ind['cum_vd'][i]:+,.0f} (change over last 20 sessions "
                 f"{ind['cum_vd'][i] - ind['cum_vd'][i - 20]:+,.0f}).")
        L.append(f"- Flag was 'confirmed' on {nconf} of the last {nh} sessions.")
        L.append("")

        # 4. support
        L.append("### 4. Support levels (backtested)")
        L.append("")
        if not r["zones"]:
            L.append("No zone below price met N ≥ 2.")
        else:
            L.append("| Zone | Distance | Sources (candidate count) | N | Reversed | Broke | Unresolved | Reversal rate | Last touches |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for z in r["zones"]:
                bt = z["bt"]
                src = ", ".join(f"{k} ×{v}" for k, v in sorted(z["sources"].items()))
                lo, hi = z["center"] * (1 - TOL), z["center"] * (1 + TOL)
                recent = "; ".join(f"{d:%Y-%m-%d} {o}" for d, o in bt["dates"][-3:])
                flag = " ⚠️ low sample size — directional signal only, not a reliable rate" if bt["n"] < RELIABLE_N else ""
                L.append(f"| {fmt(z['center'])} ({fmt(lo)}–{fmt(hi)}) | {(z['center'] / c - 1) * 100:+.1f}% | {src} | "
                         f"{bt['n']} | {bt['bounce']} | {bt['break']} | {bt['unresolved']} | {pct(bt['rate'])}{flag} | {recent} |")
        L.append("")

    L.append("## Method")
    L.append("")
    L.append(f"""- **SMA/EMA:** simple and exponential averages of close; EMAs seeded with the SMA of their first N values. MACD = EMA12 − EMA26, signal = 9-period EMA of MACD.
- **RSI:** Wilder's smoothing (first average = simple mean of 14 gains/losses, then (prev×13 + current)/14).
- **Bollinger:** 20-period SMA ± 2 × population standard deviation (ddof = 0). %B = (close − lower)/(upper − lower).
- **Trend:** five checks (close vs SMA20, SMA20 vs SMA50, SMA50 vs SMA100, EMA12 vs EMA26, SMA50 vs its value 10 sessions ago), each +1 bullish / −1 bearish / 0 if the two values are within 0.5% (flat); net score ≥ +3 = up, ≤ −3 = down, otherwise sideways.
- **Volume Delta:** "Volume Delta (Close)" is each day's net delta. Confirmed when the sign of the 10-day close-to-close change matches the sign of the 10-day rolling sum of delta; otherwise diverging.
- **Support candidates** (full series): swing lows (lowest low within ±{SWING_K} bars); bars whose low came within 0.5% of the lower Bollinger Band; days where at least two of SMA20/50/100 were within 0.5% of each other (level = their mean); volume-delta absorption bars (intrabar low-delta in the most negative 20% of the series, at least half of it bought back by the close, and close in the upper half of the bar's range; level = bar low). Candidates are merged greedily into ±0.5% zones; only zones entirely below the last close are kept. Where two qualifying zones' bands overlap (centres within 1%), only the one with more touches is listed.
- **Backtest:** a touch = prior close above the zone top and the bar's low reaches the zone top (gaps through the zone count). The first close outside the zone on the touch day or the next {BOUNCE_WINDOW} sessions decides it: above the zone = reversed, below = broke; still inside = unresolved (counted in N, not as a reversal). After a touch, the next one needs a fresh close back above the zone. Touches with fewer than {BOUNCE_WINDOW} sessions of follow-through at the end of the data are excluded. Zones need N ≥ {MIN_N}; N < {RELIABLE_N} is flagged. Up to {MAX_ZONES} zones, nearest to price first.
- **Caveats:** touches are counted across the full series, including touches before the level formed, so rates include look-ahead. Swing-low candidates are reversals by construction, which biases rates upward for zones built mostly from swing lows. Rates on small N are not statistically meaningful.""")
    L.append("")
    return "\n".join(L)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "reports", "swing_report.md")
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    results = [analyse(f) for f in files]
    text = report(results)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
