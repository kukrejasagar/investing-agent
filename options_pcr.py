"""
Options Put/Call Ratio Fetcher
Fetches P/C ratio for the top 5 most actively traded options using
Yahoo Finance (free, delayed data). Filters for 2026 expirations.

Usage:
    python options_pcr.py           # fetch live data from Yahoo Finance
    python options_pcr.py --demo    # run with sample data (no network needed)

Yahoo Finance requires a session crumb since 2024; this script handles
that handshake automatically.
"""

import sys
import time
import datetime
import requests
from collections import defaultdict

# Top 5 most actively traded options underlyings (by consistent volume/OI)
SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
CRUMB_URL   = "https://query2.finance.yahoo.com/v1/test/getcrumb"
CONSENT_URL = "https://fc.yahoo.com"


# ---------------------------------------------------------------------------
# Sample data — realistic values representative of April 2026 market conditions
# ---------------------------------------------------------------------------
DEMO_DATA = {
    "SPY": {
        "expirations": 47,
        "put_volume":   18_432_100,
        "call_volume":  16_780_500,
        "put_oi":      312_450_000,
        "call_oi":     290_120_000,
    },
    "QQQ": {
        "expirations": 42,
        "put_volume":   9_812_400,
        "call_volume":  9_102_200,
        "put_oi":      156_230_000,
        "call_oi":     148_780_000,
    },
    "AAPL": {
        "expirations": 38,
        "put_volume":   5_234_700,
        "call_volume":  6_890_300,
        "put_oi":       88_340_000,
        "call_oi":     104_560_000,
    },
    "NVDA": {
        "expirations": 40,
        "put_volume":   7_650_200,
        "call_volume":  9_430_100,
        "put_oi":      122_100_000,
        "call_oi":     168_450_000,
    },
    "TSLA": {
        "expirations": 39,
        "put_volume":   6_120_800,
        "call_volume":  7_980_600,
        "put_oi":       94_200_000,
        "call_oi":     118_670_000,
    },
}


# ---------------------------------------------------------------------------
# Yahoo Finance session + API helpers
# ---------------------------------------------------------------------------

def create_session() -> tuple[requests.Session, str]:
    """
    Establish a Yahoo Finance session and obtain a crumb token.
    Yahoo Finance (post-2024) requires cookies + crumb for API access.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    # Step 1: Hit the consent endpoint to receive session cookies
    try:
        session.get(CONSENT_URL, timeout=10)
    except Exception:
        pass  # Endpoint may redirect; cookie may still be set

    # Step 2: Exchange cookies for an API crumb
    crumb_resp = session.get(CRUMB_URL, timeout=10)
    crumb_resp.raise_for_status()
    crumb = crumb_resp.text.strip()

    if not crumb or len(crumb) > 50:
        raise RuntimeError(f"Unexpected crumb value: {crumb!r}")

    print(f"  Session established (crumb: {crumb})")
    return session, crumb


def fetch_expirations(session: requests.Session, crumb: str, symbol: str) -> list[int]:
    """Return all available expiration timestamps for a symbol."""
    url = OPTIONS_URL.format(symbol=symbol)
    resp = session.get(url, params={"crumb": crumb}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("optionChain", {}).get("result", [])
    if not result:
        return []
    return result[0].get("expirationDates", [])


def fetch_options_chain(
    session: requests.Session, crumb: str, symbol: str, expiration_ts: int
) -> dict:
    """Return calls and puts for a given symbol and expiration timestamp."""
    url = OPTIONS_URL.format(symbol=symbol)
    resp = session.get(url, params={"crumb": crumb, "date": expiration_ts}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("optionChain", {}).get("result", [])
    if not result or not result[0].get("options"):
        return {"calls": [], "puts": []}
    chain = result[0]["options"][0]
    return {
        "calls": chain.get("calls", []),
        "puts":  chain.get("puts",  []),
    }


def sum_field(contracts: list[dict], field: str) -> int:
    return sum(c.get(field, 0) or 0 for c in contracts)


def filter_2026(timestamps: list[int]) -> list[int]:
    return [ts for ts in timestamps if datetime.datetime.utcfromtimestamp(ts).year == 2026]


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_symbol_live(session: requests.Session, crumb: str, symbol: str) -> dict | None:
    print(f"\n  Fetching expirations for {symbol}...", end="", flush=True)
    expirations = fetch_expirations(session, crumb, symbol)
    expirations_2026 = filter_2026(expirations)

    if not expirations_2026:
        print(" no 2026 expirations found")
        return None

    print(f" {len(expirations_2026)} dates in 2026")

    totals: dict[str, int] = defaultdict(int)

    for i, exp_ts in enumerate(expirations_2026):
        exp_date = datetime.datetime.utcfromtimestamp(exp_ts).strftime("%Y-%m-%d")
        print(f"    [{i+1:02d}/{len(expirations_2026)}] {exp_date}", end="\r", flush=True)

        chain = fetch_options_chain(session, crumb, symbol, exp_ts)

        for field in ("volume", "openInterest"):
            totals[f"put_{field}"]  += sum_field(chain["puts"],  field)
            totals[f"call_{field}"] += sum_field(chain["calls"], field)

        if i < len(expirations_2026) - 1:
            time.sleep(0.25)

    print(f"    Done — {len(expirations_2026)} expirations processed          ")

    return {
        "symbol":      symbol,
        "expirations": len(expirations_2026),
        "put_volume":  totals["put_volume"],
        "call_volume": totals["call_volume"],
        "pcr_volume":  safe_ratio(totals["put_volume"],       totals["call_volume"]),
        "put_oi":      totals["put_openInterest"],
        "call_oi":     totals["call_openInterest"],
        "pcr_oi":      safe_ratio(totals["put_openInterest"], totals["call_openInterest"]),
    }


def analyze_symbol_demo(symbol: str) -> dict:
    d = DEMO_DATA[symbol]
    print(f"\n  {symbol}: {d['expirations']} expirations (sample data)")
    return {
        "symbol":      symbol,
        "expirations": d["expirations"],
        "put_volume":  d["put_volume"],
        "call_volume": d["call_volume"],
        "pcr_volume":  safe_ratio(d["put_volume"],  d["call_volume"]),
        "put_oi":      d["put_oi"],
        "call_oi":     d["call_oi"],
        "pcr_oi":      safe_ratio(d["put_oi"],      d["call_oi"]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], demo: bool) -> None:
    mode_label = "SAMPLE DATA — for illustration only" if demo else "Yahoo Finance, ~15-min delayed"
    sep  = "=" * 90
    thin = "-" * 90
    print(f"\n{sep}")
    print(f" PUT/CALL RATIO SUMMARY — 2026 Expirations ({mode_label})")
    print(sep)
    print(
        f"  {'Symbol':<7} {'Exps':>5} {'PCR Vol':>10} {'PCR OI':>10} "
        f"{'Put Vol':>13} {'Call Vol':>13} {'Put OI':>13} {'Call OI':>13}"
    )
    print(thin)
    for r in results:
        pcr_v = f"{r['pcr_volume']:.4f}" if r["pcr_volume"] == r["pcr_volume"] else "    N/A"
        pcr_o = f"{r['pcr_oi']:.4f}"     if r["pcr_oi"]    == r["pcr_oi"]    else "    N/A"
        print(
            f"  {r['symbol']:<7} {r['expirations']:>5} {pcr_v:>10} {pcr_o:>10} "
            f"{r['put_volume']:>13,} {r['call_volume']:>13,} "
            f"{r['put_oi']:>13,} {r['call_oi']:>13,}"
        )
    print(sep)
    print()
    print("  PCR Interpretation:")
    print("    < 0.7  — Bullish (calls dominate; market is positioned for upside)")
    print("    ~ 1.0  — Neutral")
    print("    > 1.0  — Bearish (puts dominate; hedging or downside bets are elevated)")
    print()
    print("  Columns:")
    print("    Exps    = number of 2026 expiration dates with active contracts")
    print("    PCR Vol = sum of all 2026 put volume  / sum of all 2026 call volume")
    print("    PCR OI  = sum of all 2026 put OI      / sum of all 2026 call OI")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_mode = "--demo" in sys.argv

    sep = "=" * 55
    print(sep)
    print(" Options P/C Ratio Fetcher")
    print(f" Date    : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f" Symbols : {', '.join(SYMBOLS)}")
    if demo_mode:
        print(" Mode    : DEMO (sample data, no network)")
    else:
        print(" Source  : Yahoo Finance (free, ~15-min delayed)")
    print(sep)

    results = []

    if demo_mode:
        print("\nLoading sample data...")
        for symbol in SYMBOLS:
            results.append(analyze_symbol_demo(symbol))

    else:
        print("\nEstablishing Yahoo Finance session...")
        try:
            session, crumb = create_session()
        except Exception as exc:
            print(f"[!] Failed to establish session: {exc}")
            print("\nTip: run with --demo to see sample output without network access.")
            return

        for symbol in SYMBOLS:
            try:
                result = analyze_symbol_live(session, crumb, symbol)
                if result:
                    results.append(result)
            except requests.HTTPError as exc:
                print(f"\n  [!] HTTP {exc.response.status_code} for {symbol}: {exc}")
            except Exception as exc:
                print(f"\n  [!] Error for {symbol}: {exc}")
            time.sleep(0.5)

    if results:
        print_summary(results, demo=demo_mode)
    else:
        print("\n[!] No data collected.")


if __name__ == "__main__":
    main()
