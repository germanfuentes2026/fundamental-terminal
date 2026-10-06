"""Fundamental Terminal — Streamlit desk for SEC XBRL fundamental scores."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.altman import compute_altman
from src.piotroski import compute_piotroski
from src.sec_loader import (
    CompanyMeta,
    extract_statement_bundle,
    fetch_market_snapshot,
    load_company_facts,
    load_submissions,
    enrich_meta,
    resolve_ticker,
    SecLoaderError,
)

st.set_page_config(
    page_title="Fundamental Terminal",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLOOMBERG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #1a2420 0%, #0b0e11 45%, #07090b 100%);
    color: #d7e0d8;
}

header[data-testid="stHeader"] {
    background: #0b0e11;
    border-bottom: 1px solid #1f2a22;
}

section[data-testid="stSidebar"] {
    background: #0a0d10;
    border-right: 1px solid #1f2a22;
}

section[data-testid="stSidebar"] * {
    color: #c5d0c6;
}

.block-container {
    padding-top: 1.1rem;
    max-width: 1480px;
}

h1, h2, h3 {
    font-family: "IBM Plex Sans", sans-serif;
    letter-spacing: 0.04em;
}

.ft-masthead {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border: 1px solid #243328;
    background: linear-gradient(90deg, #101612 0%, #0d1210 60%, #151208 100%);
    padding: 14px 18px 12px 18px;
    margin-bottom: 14px;
}
.ft-brand {
    font-family: "IBM Plex Mono", monospace;
    color: #f5a623;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
}
.ft-title {
    font-size: 28px;
    font-weight: 700;
    color: #eef6ef;
    line-height: 1.1;
    margin-top: 4px;
}
.ft-sub {
    color: #7f8f82;
    font-size: 13px;
    margin-top: 4px;
}
.ft-clock {
    text-align: right;
    font-family: "IBM Plex Mono", monospace;
    color: #9aa89b;
    font-size: 12px;
}
.ft-ticker {
    color: #f5a623;
    font-size: 22px;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background: #111714;
    border: 1px solid #243328;
    padding: 10px 12px;
}
div[data-testid="stMetric"] label {
    color: #8b9a8d !important;
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.12em;
    font-size: 11px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: "IBM Plex Mono", monospace;
    color: #e8f3e9;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0d1110;
    border-bottom: 1px solid #243328;
}
.stTabs [data-baseweb="tab"] {
    background: #0d1110;
    color: #8b9a8d;
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.08em;
}
.stTabs [aria-selected="true"] {
    color: #f5a623 !important;
    border-bottom: 2px solid #f5a623;
}

.pass { color: #3ddc84; font-weight: 600; }
.fail { color: #ff5c5c; font-weight: 600; }
.miss { color: #f5a623; font-weight: 600; }

.zone-SAFE { color: #3ddc84; }
.zone-GREY { color: #f5a623; }
.zone-DISTRESS { color: #ff5c5c; }

.stButton>button {
    background: #f5a623;
    color: #111;
    border: 0;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: "IBM Plex Mono", monospace;
}
.stButton>button:hover {
    background: #ffc056;
    color: #111;
}

hr { border-color: #243328; }

.stDataFrame { border: 1px solid #243328; }

.audit-note {
    font-family: "IBM Plex Mono", monospace;
    font-size: 12px;
    color: #8b9a8d;
}
</style>
"""

st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)


def _fmt_num(v, kind: str = "int") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "ratio":
        return f"{v:.4f}"
    if kind == "score":
        return f"{v:.2f}"
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:,.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.1f}M"
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _pass_label(passed: bool | None) -> str:
    if passed is True:
        return "PASS · 1"
    if passed is False:
        return "FAIL · 0"
    return "N/A · —"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_resolve(ticker: str) -> CompanyMeta:
    return resolve_ticker(ticker)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_facts(cik: str) -> dict:
    return load_company_facts(cik)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_subs(cik: str) -> dict:
    return load_submissions(cik)


@st.cache_data(show_spinner=False, ttl=60 * 15)
def cached_mkt(ticker: str) -> dict:
    return fetch_market_snapshot(ticker)


def render_masthead(meta: CompanyMeta | None, mkt: dict | None) -> None:
    ticker_html = ""
    if meta:
        ticker_html = f'<div class="ft-ticker">{meta.ticker} &nbsp;<span style="color:#9aa89b;font-size:14px">{meta.name}</span></div>'
    st.markdown(
        f"""
        <div class="ft-masthead">
          <div>
            <div class="ft-brand">Fundamental Terminal</div>
            <div class="ft-title">SEC Fundamental Analytics</div>
            <div class="ft-sub">Piotroski F-Score · Altman Z-Score</div>
            {ticker_html}
          </div>
          <div class="ft-clock">Data: data.sec.gov &nbsp;|&nbsp; Market: Yahoo (optional)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("**COMMAND**")
    ticker = st.text_input("Ticker", value="AAPL", placeholder="e.g. AAPL, MSFT, JPM").strip().upper()
    run = st.button("Load SEC XBRL", width="stretch")
    st.markdown("---")
    st.markdown("**MARKET (X4)**")
    px_override = st.number_input("Price override (optional)", min_value=0.0, value=0.0, step=0.01)
    mcap_override = st.number_input("Market cap override (optional)", min_value=0.0, value=0.0, step=1_000_000.0, format="%.0f")
    st.caption("If Yahoo/Stooq is unavailable, enter last price. Market cap = price × XBRL shares unless override is set.")
    st.markdown("---")
    st.markdown("**ABOUT THE MODELS**")
    st.caption(
        "Piotroski (2000): nine binary fundamental signals on profitability, "
        "leverage/liquidity/dilution and operating efficiency. Score 0–9."
    )
    st.caption(
        "Altman (1968) original manufacturing Z: "
        "1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5. "
        "Cutoffs 1.81 / 2.99. Not a credit rating."
    )
    st.markdown("---")
    st.caption("Educational desk. Filings can use heterogeneous US-GAAP concepts.")

if "loaded_ticker" not in st.session_state:
    st.session_state.loaded_ticker = ticker or "AAPL"

if run:
    st.session_state.loaded_ticker = ticker

active = st.session_state.loaded_ticker

if not active:
    render_masthead(None, None)
    st.info("Enter a ticker in the sidebar and press **Load SEC XBRL**.")
    st.stop()

try:
    with st.spinner(f"Resolving {active} on SEC EDGAR…"):
        meta = replace(cached_resolve(active))
        subs = cached_subs(meta.cik)
        meta = enrich_meta(meta, subs)
        facts = cached_facts(meta.cik)
        bundle = extract_statement_bundle(facts)
        mkt = cached_mkt(active)
        if px_override and px_override > 0:
            mkt = {**mkt, "price": float(px_override), "source": "manual-price"}
        if mcap_override and mcap_override > 0:
            mkt = {**mkt, "market_cap": float(mcap_override), "source": "manual-mcap"}
except SecLoaderError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load SEC data: {exc}")
    st.stop()

render_masthead(meta, mkt)

f_res = compute_piotroski(bundle)
z_res = compute_altman(
    bundle,
    market_cap=mkt.get("market_cap") if mkt else None,
    market_price=mkt.get("price") if mkt else None,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("TICKER", meta.ticker)
c2.metric("F-SCORE", "—" if f_res.score is None else f"{f_res.score}/9", f_res.label)
c3.metric("Z-SCORE", "—" if z_res.z is None else f"{z_res.z:.2f}", z_res.zone)
c4.metric("PRICE", _fmt_num(mkt.get("price") if mkt else None, "ratio"))
c5.metric("MKT CAP", _fmt_num(mkt.get("market_cap") if mkt else None))
c6.metric("FY", str(f_res.year or z_res.year or "—"))

tab_ov, tab_f, tab_z, tab_form = st.tabs(
    ["OVERVIEW", "PIOTROSKI", "ALTMAN", "FORMULAS"]
)

with tab_ov:
    left, right = st.columns((1.15, 1))
    with left:
        st.subheader("Piotroski F-Score")
        rows = []
        for cr in f_res.criteria:
            rows.append(
                {
                    "signal": cr.code,
                    "name": cr.name,
                    "group": cr.group,
                    "result": _pass_label(cr.passed),
                    "points": cr.points if cr.passed is not None else None,
                    "detail": cr.narrative,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if f_res.missing:
            st.warning("Incomplete signals (counted as 0): " + ", ".join(f_res.missing))
    with right:
        st.subheader("Altman Z-Score")
        zrows = []
        for cm in z_res.components:
            zrows.append(
                {
                    "factor": cm.code,
                    "name": cm.name,
                    "weight": cm.weight,
                    "ratio": None if cm.value is None else round(cm.value, 4),
                    "contribution": None if cm.contribution is None else round(cm.contribution, 4),
                }
            )
        st.dataframe(pd.DataFrame(zrows), width="stretch", hide_index=True)
        st.markdown(
            f"**Zone:** `{z_res.zone}` — {z_res.zone_detail}  \n"
            f"Market equity source: {z_res.market_cap_source or 'unavailable'}"
        )
        if z_res.missing:
            st.warning("Missing Altman factors: " + ", ".join(z_res.missing))

with tab_f:
    st.markdown(
        f"Fiscal years **{f_res.prior_year} → {f_res.year}**. "
        f"Score **{f_res.score if f_res.score is not None else 'n/a'} / 9** · {f_res.label}."
    )
    for cr in f_res.criteria:
        color = "pass" if cr.passed else ("fail" if cr.passed is False else "miss")
        with st.expander(f"{cr.code}  ·  {cr.name}  ·  {_pass_label(cr.passed)}", expanded=False):
            st.markdown(f"<span class='{color}'>{_pass_label(cr.passed)}</span> — {cr.explanation}", unsafe_allow_html=True)
            st.latex(cr.latex)
            st.code(cr.formula, language="text")
            st.markdown(cr.narrative)
            pretty = {k: v for k, v in cr.inputs.items() if not str(k).endswith((".accn",))}
            st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pretty.items()})

with tab_z:
    st.markdown(
        r"$Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5$"
        + (f"  =  **{z_res.z:.3f}**" if z_res.z is not None else "  =  *incomplete*")
    )
    st.caption(z_res.zone_detail)
    for cm in z_res.components:
        with st.expander(
            f"{cm.code}  ·  {cm.name}  ·  "
            f"{'—' if cm.contribution is None else f'{cm.weight} × {cm.value:.4f} = {cm.contribution:.4f}'}",
            expanded=False,
        ):
            st.latex(cm.latex)
            st.code(cm.formula, language="text")
            st.write(cm.explanation)
            pretty = {k: v for k, v in cm.inputs.items()}
            st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pretty.items()})

with tab_form:
    st.subheader("Piotroski F-Score (Piotroski, 2000)")
    st.markdown(
        r"""
Nine binary signals. $F=\sum_{i=1}^{9} F_i \in \{0,\ldots,9\}$.

**Profitability**
- $F_{ROA}=1$ iff $ROA_t = NI_t / \overline{A}_t > 0$
- $F_{CFO}=1$ iff $CFO_t > 0$
- $F_{\Delta ROA}=1$ iff $ROA_t - ROA_{t-1} > 0$
- $F_{ACCRUAL}=1$ iff $CFO_t > NI_t$

**Leverage, liquidity, dilution**
- $F_{\Delta LEVER}=1$ iff $\frac{LTD_t}{A_t} < \frac{LTD_{t-1}}{A_{t-1}}$
- $F_{\Delta LIQUID}=1$ iff $\frac{CA_t}{CL_t} > \frac{CA_{t-1}}{CL_{t-1}}$
- $F_{EQ}=1$ iff $Shares_t \le Shares_{t-1}$

**Efficiency**
- $F_{\Delta MARGIN}=1$ iff $\frac{GP_t}{S_t} > \frac{GP_{t-1}}{S_{t-1}}$
- $F_{\Delta TURN}=1$ iff $\frac{S_t}{\overline{A}_t} > \frac{S_{t-1}}{\overline{A}_{t-1}}$

Interpretation used here: 8–9 high quality, 5–7 average, 0–4 low quality.
        """
    )
    st.subheader("Altman Z-Score (Altman, 1968)")
    st.markdown(
        r"""
$$
Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 1.0 X_5
$$

| Factor | Definition | Weight |
|---|---|---|
| $X_1$ | Working capital / Total assets | 1.2 |
| $X_2$ | Retained earnings / Total assets | 1.4 |
| $X_3$ | EBIT / Total assets | 3.3 |
| $X_4$ | Market value of equity / Total liabilities | 0.6 |
| $X_5$ | Sales / Total assets | 1.0 |

Original manufacturing cutoffs: $Z>2.99$ safe, $1.81\le Z\le 2.99$ grey, $Z<1.81$ distress.

This desk maps EBIT to US-GAAP `OperatingIncomeLoss` when a dedicated EBIT tag is absent.
$X_4$ uses Yahoo Finance market cap when available, else price × XBRL shares.
        """
    )


