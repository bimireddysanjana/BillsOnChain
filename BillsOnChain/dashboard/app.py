"""
BOC (Bill on Chain) — Analytics Dashboard  ·  Classic Edition
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from data_loader import load_all

# ── Auto-style every chart ────────────────────────────────────────────────────
if not getattr(st, "_is_plotly_patched", False):
    _orig_plotly_chart = st.plotly_chart

    def _patched_plotly_chart(fig, *args, **kwargs):
        _AX = dict(
            color="#24292f", tickcolor="#24292f",
            tickfont=dict(color="#24292f", size=11, family="Georgia, serif"),
            title_font=dict(color="#57606a", size=12, family="Georgia, serif"),
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="rgba(0,0,0,0.1)",
            zerolinecolor="rgba(0,0,0,0.08)",
        )
        fig.update_xaxes(**_AX)
        fig.update_yaxes(**_AX)
        fig.update_layout(font_color="#24292f")
        fig.update_layout(legend_font_color="#24292f", legend_font_size=11)
        fig.update_traces(selector=dict(type="pie"),
                          textfont_color="#ffffff",
                          insidetextfont_color="#ffffff")
        return _orig_plotly_chart(fig, *args, **kwargs)

    st.plotly_chart = _patched_plotly_chart
    st._is_plotly_patched = True

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BOC Analytics",
    page_icon="⛓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ─── Base ─────────────────────────────────────────────── */
* { box-sizing: border-box; }
html, body, .stApp { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: #ffffff;
    color: #24292f;
}

/* Force all text visible */
.stApp *, p, span, div, label, h1, h2, h3, h4, h5, h6, li {
    color: #24292f;
}

/* ─── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #f6f8fa !important;
    border-right: 1px solid #d0d7de;
    width: 220px !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #57606a !important;
}

/* Radio items in sidebar styled as separate blocks */
[data-testid="stSidebar"] [data-baseweb="radio"] {
    background: #ffffff !important;
    border: 1px solid #d0d7de !important;
    border-radius: 6px !important;
    padding: 0.55rem 0.8rem !important;
    margin-bottom: 0.45rem !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}

/* Hover style */
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: #f3f4f6 !important;
    border-color: #8c959f !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover * {
    color: #24292f !important;
}

/* Hide native radio circle */
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Adjust text inside styled blocks */
[data-testid="stSidebar"] [data-baseweb="radio"] div {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #57606a !important;
}

/* Selected active state styling using CSS :has() */
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] [data-baseweb="radio"]:has([aria-checked="true"]),
[data-testid="stSidebar"] [data-baseweb="radio"]:has([data-checked="true"]) {
    background: rgba(96, 165, 112, 0.08) !important;
    border-color: #60a570 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) *,
[data-testid="stSidebar"] [data-baseweb="radio"]:has([aria-checked="true"]) *,
[data-testid="stSidebar"] [data-baseweb="radio"]:has([data-checked="true"]) * {
    color: #60a570 !important;
    font-weight: 600 !important;
}

/* ─── Metric cards ──────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #60a570;
    border: 1px solid #60a570;
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(96, 165, 112, 0.3);
}
div[data-testid="stMetric"] label {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
    color: #ffffff !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ─── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #d0d7de;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    color: #57606a !important;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.5rem 1rem;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    color: #60a570 !important;
    border-bottom: 2px solid #60a570 !important;
    background: transparent !important;
}

/* ─── Selectbox ─────────────────────────────────────────── */
.stSelectbox label { color: #57606a !important; font-size: 0.75rem !important; }
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d0d7de !important;
    border-radius: 6px;
    color: #24292f !important;
}
[data-baseweb="select"] span { color: #24292f !important; }

/* ─── DataFrames ────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    overflow: hidden;
}
div[data-testid="stDataFrame"] thead th {
    background: #f6f8fa !important;
    color: #57606a !important;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 1px solid #d0d7de !important;
}
div[data-testid="stDataFrame"] td {
    color: #24292f !important;
    font-size: 0.82rem;
    border-bottom: 1px solid #f6f8fa !important;
}

/* ─── Section header ────────────────────────────────────── */
.sec-head {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #57606a !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #d0d7de;
}

/* ─── Page header ───────────────────────────────────────── */
.page-head {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #24292f;
    margin: 0 0 0.15rem;
    line-height: 1.2;
}
.page-sub {
    font-size: 0.78rem;
    color: #57606a !important;
    margin: 0 0 1.2rem;
}

/* ─── Divider ───────────────────────────────────────────── */
hr { border: none; border-top: 1px solid #d0d7de; margin: 0.8rem 0 1.2rem; }

/* ─── Info / warning boxes ──────────────────────────────── */
.stAlert { border-radius: 6px; }

/* ─── Sidebar brand ─────────────────────────────────────── */
.brand-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #24292f !important;
}
.brand-sub { font-size: 0.68rem; color: #8c959f !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
data          = load_all()
users         = data["users"]
bills         = data["bills"]
extractions   = data["extractions"]
fraud         = data["fraud"]
nfts          = data["nfts"]
reward_credits= data["reward_credits"]
reward_claims = data["reward_claims"]
reward_txns   = data["reward_txns"]
mint_attempts = data["mint_attempts"]
leaderboard   = data["leaderboard"]
wallets       = data["wallets"]
audit         = data["audit"]

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#60a570",
    "green":  "#1a7f37",
    "amber":  "#9a6700",
    "red":    "#cf222e",
    "purple": "#8250df",
    "cyan":   "#0594fa",
    "muted":  "#57606a",
}

_BG   = "rgba(0,0,0,0)"
_FONT = dict(family="Inter, sans-serif", color="#24292f", size=12)

def _layout(**extra):
    """Standard Plotly layout kwargs for a clean, classic look."""
    return dict(
        template="plotly",
        plot_bgcolor=_BG,
        paper_bgcolor=_BG,
        font=_FONT,
        margin=dict(l=0, r=10, t=36, b=0),
        **extra,
    )

def _title(text):
    return dict(text=text, font=dict(family="Inter, sans-serif",
                                     color="#57606a", size=11), x=0)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.4rem 0.5rem 1.2rem; border-bottom:1px solid #21262d; margin-bottom:1rem;'>
        <div style='font-size:1.6rem; margin-bottom:0.4rem;'>⛓️</div>
        <div class='brand-title'>BOC Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("Overview",            "○"),
        ("Users",               "○"),
        ("User Investigator",   "○"),
        ("Bills & Pipeline",    "○"),
        ("Fraud Detection",     "○"),
        ("NFTs & Minting",      "○"),
        ("Rewards & Leaderboard","○"),
        ("Wallets",             "○"),
        ("Audit Log",           "○"),
        ("ML Insights",         "○"),
    ]
    page = st.radio("", [p[0] for p in pages], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.85rem; color:#24292f !important; font-weight:600; padding:0 0.2rem;
                border-top:1px solid #d0d7de; padding-top:0.8rem;'>
        {len(users):,} users &nbsp;·&nbsp; {len(bills):,} bills
    </div>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def hdr(title: str, sub: str = ""):
    st.markdown(f"<p class='page-head'>{title}</p>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<p class='page-sub'>{sub}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

def sec(label: str):
    st.markdown(f"<p class='sec-head'>{label}</p>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    hdr("Platform Overview", "Key metrics across the BOC ecosystem")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Users",    f"{len(users):,}")
    c2.metric("Total Bills",    f"{len(bills):,}")
    c3.metric("NFTs Minted",    f"{len(nfts):,}")
    c4.metric("Fraud Checks",   f"{len(fraud):,}")
    c5.metric("Reward Credits", f"{reward_credits['amount'].sum():,.0f}" if not reward_credits.empty else "0")
    c6.metric("Wallets",        f"{len(wallets):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Pipeline + Category ──
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        sec("Bill Status Pipeline")
        if not bills.empty and "status" in bills.columns:
            STATUS_ORDER = [
                "completed","minted","dedup_passed","dedup_checking",
                "extracted","extracting","fraud_passed","fraud_checking",
                "hash_complete","hashing","uploaded","pending_upload",
                "mint_failed","extraction_failed","fraud_rejected","fraud_flagged","duplicate",
            ]
            PASS = {"completed","minted","dedup_passed","extracted","fraud_passed","hash_complete"}
            FAIL = {"mint_failed","extraction_failed","fraud_rejected","fraud_flagged","duplicate"}

            sc = bills["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
            sc = sc[sc > 0]
            bar_colors = [
                PALETTE["green"] if s in PASS
                else PALETTE["red"] if s in FAIL
                else PALETTE["blue"]
                for s in sc.index
            ]
            fig = go.Figure(go.Bar(
                x=sc.values, y=sc.index, orientation="h",
                marker_color=bar_colors, opacity=0.85,
                text=[f"{v:,}" for v in sc.values],
                textposition="outside",
                textfont=dict(color="#8b949e", size=10),
            ))
            fig.update_layout(
                **_layout(height=360),
                title=_title("Bills by processing status"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(tickfont=dict(size=10, color="#8b949e")),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

    with col_r:
        sec("Category Breakdown")
        if not bills.empty and "category" in bills.columns:
            cat = bills["category"].dropna().value_counts()
            fig = px.pie(
                names=cat.index, values=cat.values,
                hole=0.62,
                color_discrete_sequence=[
                    "#58a6ff","#3fb950","#d29922","#f85149","#bc8cff",
                    "#79c0ff","#56d364","#ffa657","#ff7b72","#d2a8ff",
                    "#7ee787","#f0883e",
                ],
            )
            fig.update_traces(
                textposition="none",   # no overlapping labels
                hovertemplate="<b>%{label}</b><br>%{value} bills (%{percent})<extra></extra>",
            )
            fig.update_layout(
                **_layout(height=360),
                title=_title("Spend category distribution"),
                showlegend=True,
                legend=dict(
                    orientation="v", x=1.0, y=0.5,
                    font=dict(size=10, color="#57606a"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                annotations=[dict(
                    text=f"<b>{len(bills):,}</b><br><span style='font-size:9px'>Bills</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=13, color="#24292f"),
                )],
            )
            st.plotly_chart(fig, width="stretch")

    # ── Row 2: User growth + Bill volume ──
    col_a, col_b = st.columns(2)

    with col_a:
        sec("User Registrations")
        if not users.empty and "createdAt" in users.columns:
            ug = (users.dropna(subset=["createdAt"])
                       .set_index("createdAt")
                       .resample("W")["id"].count()
                       .reset_index()
                       .rename(columns={"id":"Count","createdAt":"Week"}))
            ug["Cumulative"] = ug["Count"].cumsum()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=ug["Week"], y=ug["Count"],
                                 name="Weekly", marker_color=PALETTE["blue"],
                                 opacity=0.6), secondary_y=False)
            fig.add_trace(go.Scatter(x=ug["Week"], y=ug["Cumulative"],
                                     name="Cumulative",
                                     line=dict(color=PALETTE["cyan"], width=1.5),
                                     mode="lines"), secondary_y=True)
            fig.update_layout(**_layout(height=260),
                              title=_title("Weekly new users & running total"),
                              showlegend=True,
                              legend=dict(orientation="h", y=-0.15,
                                          font=dict(size=9, color="#8b949e")))
            st.plotly_chart(fig, width="stretch")

    with col_b:
        sec("Bill Upload Volume")
        if not bills.empty and "createdAt" in bills.columns:
            bg = (bills.dropna(subset=["createdAt"])
                       .set_index("createdAt")
                       .resample("W")["id"].count()
                       .reset_index()
                       .rename(columns={"id":"Bills","createdAt":"Week"}))
            fig = px.area(bg, x="Week", y="Bills",
                          color_discrete_sequence=[PALETTE["purple"]])
            fig.update_traces(fillcolor="rgba(188,140,255,0.1)",
                              line=dict(width=1.5, color=PALETTE["purple"]))
            fig.update_layout(**_layout(height=260),
                              title=_title("Weekly bill uploads"),
                              showlegend=False)
            st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
#  USERS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Users":
    hdr("Users", "Registration, identity, roles and referrals")

    total     = len(users)
    verified  = int((users["emailVerified"] == "true").sum()) if not users.empty else 0
    did_ready = int((users["didStatus"] == "ready").sum()) if not users.empty else 0
    admins    = int((users["role"] == "admin").sum()) if not users.empty else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Users",    f"{total:,}")
    c2.metric("Email Verified", f"{verified:,}", f"{verified/total*100:.1f}%" if total else "")
    c3.metric("DID Ready",      f"{did_ready:,}")
    c4.metric("Admins",         f"{admins:,}")

    t1, t2, t3 = st.tabs(["Registration Trend", "Identity & Role", "Referrals"])

    with t1:
        if not users.empty and "createdAt" in users.columns:
            period = st.selectbox("Group by", ["Day","Week","Month"], index=1)
            freq   = {"Day":"D","Week":"W","Month":"MS"}[period]
            ug = (users.dropna(subset=["createdAt"])
                       .set_index("createdAt")
                       .resample(freq)["id"].count()
                       .reset_index()
                       .rename(columns={"id":"Users","createdAt":"Period"}))
            fig = px.bar(ug, x="Period", y="Users",
                         color_discrete_sequence=[PALETTE["blue"]])
            fig.update_traces(opacity=0.75)
            fig.update_layout(**_layout(height=320),
                              title=_title(f"New users per {period.lower()}"),
                              showlegend=False)
            st.plotly_chart(fig, width="stretch")

    with t2:
        cl, cr = st.columns(2)
        with cl:
            sec("DID Status")
            if "didStatus" in users.columns:
                ds = users["didStatus"].fillna("none").value_counts()
                cmap = {"none":"#d0d7de","pending":PALETTE["amber"],
                        "ready":PALETTE["green"],"failed":PALETTE["red"]}
                fig = px.pie(names=ds.index, values=ds.values, hole=0.58,
                             color=ds.index, color_discrete_map=cmap)
                fig.update_traces(textposition="none",
                                  hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>")
                fig.update_layout(**_layout(height=300), showlegend=True,
                                  legend=dict(font=dict(size=10, color="#57606a"),
                                              bgcolor="rgba(0,0,0,0)"),
                                  title=_title("DID status breakdown"))
                st.plotly_chart(fig, width="stretch")

        with cr:
            sec("User Roles")
            if "role" in users.columns:
                rs = users["role"].fillna("user").value_counts()
                fig = px.bar(x=rs.index, y=rs.values,
                             color=rs.index,
                             color_discrete_map={"user":PALETTE["blue"],"admin":PALETTE["purple"]},
                             labels={"x":"Role","y":"Count"}, text=rs.values)
                fig.update_traces(opacity=0.8, textposition="outside",
                                  textfont=dict(color="#8b949e", size=11))
                fig.update_layout(**_layout(height=300), showlegend=False,
                                  title=_title("Users by role"))
                st.plotly_chart(fig, width="stretch")

    with t3:
        if not users.empty:
            has_code = int(users["referralCode"].notna().sum())
            referred = int(users["referredBy"].notna().sum())
            bonus    = int((users["referralBonusGranted"] == "true").sum())
            r1,r2,r3 = st.columns(3)
            r1.metric("With Referral Code",  f"{has_code:,}")
            r2.metric("Referred by Someone", f"{referred:,}")
            r3.metric("Bonus Granted",        f"{bonus:,}")

    sec("Top 20 — Lifetime Reward Points")
    if not users.empty and "lifetimeRewardPoints" in users.columns:
        top = (users[["name","email","lifetimeRewardPoints","rewardBalance","role","createdAt"]]
               .sort_values("lifetimeRewardPoints", ascending=False).head(20)
               .reset_index(drop=True))
        top.index += 1
        st.dataframe(top, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  BILLS & PIPELINE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Bills & Pipeline":
    hdr("Bills & Pipeline", "Lifecycle, categories, file sizes and OCR extraction")

    b_total   = len(bills)
    completed = int((bills["status"] == "completed").sum()) if not bills.empty else 0
    failed    = int(bills["status"].isin(["fraud_rejected","extraction_failed","mint_failed"]).sum()) if not bills.empty else 0
    dups      = int((bills["status"] == "duplicate").sum()) if not bills.empty else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Bills",  f"{b_total:,}")
    c2.metric("Completed",    f"{completed:,}", f"{completed/b_total*100:.1f}%" if b_total else "")
    c3.metric("Failed",       f"{failed:,}")
    c4.metric("Duplicates",   f"{dups:,}")

    t1, t2, t3 = st.tabs(["Pipeline", "Categories", "Extraction"])

    with t1:
        PASS = {"completed","minted"}
        FAIL = {"fraud_rejected","fraud_flagged","extraction_failed","mint_failed","duplicate"}
        if not bills.empty and "status" in bills.columns:
            sc = bills["status"].value_counts().reset_index()
            sc.columns = ["Status","Count"]
            sc["Color"] = sc["Status"].apply(
                lambda s: PALETTE["green"] if s in PASS
                else PALETTE["red"] if s in FAIL
                else PALETTE["blue"]
            )
            sc = sc.sort_values("Count")
            fig = px.bar(sc, x="Count", y="Status", orientation="h",
                         color="Color", color_discrete_map={c:c for c in sc["Color"].unique()},
                         text=[f"{v:,}" for v in sc["Count"]])
            fig.update_traces(opacity=0.8, textposition="outside",
                              textfont=dict(color="#8b949e",size=10))
            fig.update_layout(**_layout(height=400), showlegend=False,
                              title=_title("Count of bills at each pipeline stage"),
                              xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                              yaxis=dict(tickfont=dict(size=10, color="#8b949e")))
            st.plotly_chart(fig, width="stretch")

        sec("Weekly Upload Trend")
        if not bills.empty and "createdAt" in bills.columns:
            bt = (bills.dropna(subset=["createdAt"])
                       .set_index("createdAt").resample("W")["id"].count()
                       .reset_index().rename(columns={"id":"Bills","createdAt":"Week"}))
            fig = px.bar(bt, x="Week", y="Bills",
                         color_discrete_sequence=[PALETTE["blue"]])
            fig.update_traces(opacity=0.7)
            fig.update_layout(**_layout(height=240), showlegend=False,
                              title=_title("Weekly bill uploads"))
            st.plotly_chart(fig, width="stretch")

    with t2:
        cl, cr = st.columns(2)
        with cl:
            sec("Bills by Category")
            if "category" in bills.columns:
                cat = bills["category"].fillna("unknown").value_counts()
                fig = px.bar(x=cat.values, y=cat.index, orientation="h",
                             color=cat.values,
                             color_continuous_scale=[[0,"#f6f8fa"],[1,PALETTE["blue"]]],
                             text=[f"{v:,}" for v in cat.values])
                fig.update_traces(textposition="outside",
                                  textfont=dict(color="#57606a",size=10))
                fig.update_layout(**_layout(height=360), showlegend=False,
                                  coloraxis_showscale=False,
                                  title=_title(""),
                                  xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                                  yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig, width="stretch")

        with cr:
            sec("Category × Status")
            if "category" in bills.columns and "status" in bills.columns:
                hm = bills.groupby(["category","status"]).size().reset_index(name="n")
                pivot = hm.pivot(index="category",columns="status",values="n").fillna(0)
                fig = px.imshow(pivot, color_continuous_scale=[[0,"#ffffff"],[1,PALETTE["blue"]]],
                                aspect="auto", text_auto=".0f")
                fig.update_traces(textfont=dict(size=9, color="#24292f"))
                fig.update_layout(**_layout(height=360),
                                  title=_title("Bills count by category and status"),
                                  coloraxis_showscale=False,
                                  xaxis=dict(tickfont=dict(size=8)),
                                  yaxis=dict(tickfont=dict(size=9)))
                st.plotly_chart(fig, width="stretch")

    with t3:
        if not extractions.empty:
            e1,e2,e3 = st.columns(3)
            e1.metric("Extractions",    f"{len(extractions):,}")
            e2.metric("Avg Amount",     f"${extractions['totalAmount'].dropna().mean():,.2f}" if "totalAmount" in extractions.columns else "–")
            e3.metric("Avg Tax",        f"${extractions['taxAmount'].dropna().mean():,.2f}"   if "taxAmount"   in extractions.columns else "–")
            cl, cr = st.columns(2)
            with cl:
                sec("OCR Model Usage")
                if "modelUsed" in extractions.columns:
                    mu = extractions["modelUsed"].dropna().value_counts()
                    fig = px.pie(names=mu.index, values=mu.values, hole=0.55,
                                 color_discrete_sequence=[PALETTE["blue"],PALETTE["purple"],
                                                          PALETTE["cyan"],PALETTE["green"],PALETTE["amber"]])
                    fig.update_traces(textposition="none",
                                      hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>")
                    fig.update_layout(**_layout(height=280),
                                      title=_title("Model used for extraction"),
                                      legend=dict(font=dict(size=10, color="#8b949e"),
                                                  bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig, width="stretch")
            with cr:
                sec("Invoice Amount Distribution")
                if "totalAmount" in extractions.columns:
                    fig = px.histogram(extractions["totalAmount"].dropna(), nbins=40,
                                       color_discrete_sequence=[PALETTE["cyan"]])
                    fig.update_traces(opacity=0.7)
                    fig.update_layout(**_layout(height=280),
                                      title=_title("Invoice total amount histogram"),
                                      showlegend=False,
                                      xaxis_title="Amount", yaxis_title="Count")
                    st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
#  FRAUD DETECTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Fraud Detection":
    hdr("Fraud Detection", "Truthscan scores, decisions and flagged outcomes")

    if fraud.empty:
        st.info("No fraud check records found in the data.")
    else:
        total_fc = len(fraud)
        passed   = int((fraud["decision"] == "pass").sum())   if "decision" in fraud.columns else 0
        flagged  = int(fraud["decision"].isin(["flag","reject","fraud"]).sum()) if "decision" in fraud.columns else 0
        avg_sc   = fraud["score"].dropna().mean() if "score" in fraud.columns else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Checks",     f"{total_fc:,}")
        c2.metric("Passed",           f"{passed:,}", f"{passed/total_fc*100:.1f}%" if total_fc else "")
        c3.metric("Flagged/Rejected", f"{flagged:,}")
        c4.metric("Avg Fraud Score",  f"{avg_sc:.4f}" if avg_sc else "–")

        cl, cr = st.columns(2)
        with cl:
            sec("Decision Breakdown")
            if "decision" in fraud.columns:
                dec = fraud["decision"].fillna("unknown").value_counts()
                cmap = {"pass":PALETTE["green"],"flag":PALETTE["amber"],
                        "reject":PALETTE["red"],"fraud":PALETTE["red"],"unknown":"#30363d"}
                fig = px.pie(names=dec.index, values=dec.values, hole=0.58,
                             color=dec.index, color_discrete_map=cmap)
                fig.update_traces(textposition="none",
                                  hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>")
                fig.update_layout(**_layout(height=300),
                                  title=_title("Fraud check decision outcomes"),
                                  legend=dict(font=dict(size=10,color="#8b949e"),
                                              bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig, width="stretch")

        with cr:
            sec("Score Distribution")
            if "score" in fraud.columns:
                scores = fraud["score"].dropna()
                fig = px.histogram(scores, nbins=60, color_discrete_sequence=[PALETTE["red"]])
                fig.update_traces(opacity=0.7)
                fig.add_vline(x=scores.mean(), line_dash="dot", line_color=PALETTE["amber"],
                              annotation_text=f"Mean {scores.mean():.3f}",
                              annotation_font=dict(color=PALETTE["amber"], size=10))
                fig.update_layout(**_layout(height=300), showlegend=False,
                                  title=_title("Distribution of fraud confidence scores"),
                                  xaxis_title="Score", yaxis_title="Count")
                st.plotly_chart(fig, width="stretch")

        sec("Final Result")
        if "finalResult" in fraud.columns:
            fr = fraud["finalResult"].fillna("unknown").value_counts()
            fig = px.bar(x=fr.index, y=fr.values,
                         color_discrete_sequence=[PALETTE["blue"]],
                         text=[f"{v:,}" for v in fr.values])
            fig.update_traces(opacity=0.8, textposition="outside",
                              textfont=dict(color="#8b949e",size=11))
            fig.update_layout(**_layout(height=280), showlegend=False,
                              title=_title("Final result labels from fraud vendor"),
                              xaxis_title="", yaxis_title="Count",
                              xaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
#  NFTS & MINTING
# ════════════════════════════════════════════════════════════════════════════
elif page == "NFTs & Minting":
    hdr("NFTs & Minting", "Token creation, serial numbers and transfer status")

    total_nfts   = len(nfts)
    transferred  = int(nfts["transferredAt"].notna().sum()) if not nfts.empty else 0
    held         = total_nfts - transferred

    c1,c2,c3 = st.columns(3)
    c1.metric("Total NFTs Minted",  f"{total_nfts:,}")
    c2.metric("Transferred",        f"{transferred:,}")
    c3.metric("In Wallet (Held)",   f"{held:,}")

    cl, cr = st.columns(2)
    with cl:
        sec("Minting Over Time")
        if not nfts.empty and "createdAt" in nfts.columns:
            ng = (nfts.dropna(subset=["createdAt"])
                      .set_index("createdAt").resample("W")["id"].count()
                      .reset_index().rename(columns={"id":"NFTs","createdAt":"Week"}))
            ng["Cumulative"] = ng["NFTs"].cumsum()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=ng["Week"], y=ng["NFTs"],
                                 name="Weekly Minted", marker_color=PALETTE["purple"],
                                 opacity=0.65), secondary_y=False)
            fig.add_trace(go.Scatter(x=ng["Week"], y=ng["Cumulative"],
                                     name="Cumulative", mode="lines",
                                     line=dict(color=PALETTE["cyan"], width=1.5)),
                          secondary_y=True)
            fig.update_layout(**_layout(height=300),
                              title=_title("Weekly mints and running total"),
                              legend=dict(orientation="h", y=-0.2,
                                          font=dict(size=9, color="#8b949e")))
            st.plotly_chart(fig, width="stretch")

    with cr:
        sec("Transfer Status")
        if not nfts.empty:
            fig = px.pie(
                names=["Transferred","Held"],
                values=[transferred, held],
                hole=0.62,
                color=["Transferred","Held"],
                color_discrete_map={"Transferred":PALETTE["green"],"Held":"#d0d7de"},
            )
            fig.update_traces(textposition="none",
                              hovertemplate="<b>%{label}</b>: %{value}<extra></extra>")
            fig.update_layout(**_layout(height=300),
                              title=_title("NFT transfer status"),
                              legend=dict(font=dict(size=10, color="#57606a"),
                                          bgcolor="rgba(0,0,0,0)"),
                              annotations=[dict(text=f"<b>{total_nfts:,}</b><br><span style='font-size:9px'>Total</span>",
                                               x=0.5, y=0.5, showarrow=False,
                                               font=dict(size=12,color="#24292f"))])
            st.plotly_chart(fig, width="stretch")

    sec("Mint Attempt Outcomes")
    if not mint_attempts.empty and "status" in mint_attempts.columns:
        ms = mint_attempts["status"].value_counts()
        cmap_m = {k: PALETTE.get({"success":"green","failed":"red","pending":"amber"}.get(k,"blue"),"blue")
                  for k in ms.index}
        fig = px.bar(x=ms.index, y=ms.values,
                     color=ms.index, color_discrete_map=cmap_m,
                     text=[f"{v:,}" for v in ms.values])
        fig.update_traces(opacity=0.8, textposition="outside",
                          textfont=dict(color="#8b949e",size=11))
        fig.update_layout(**_layout(height=260), showlegend=False,
                          title=_title("Mint attempt result distribution"),
                          xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
#  REWARDS & LEADERBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "Rewards & Leaderboard":
    hdr("Rewards & Leaderboard", "Token credits, claims, transactions and top earners")

    tot_cred = int(reward_credits["amount"].sum()) if not reward_credits.empty else 0
    tot_clm  = int(reward_claims["amount"].sum())  if not reward_claims.empty  else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Credits Issued",     f"{tot_cred:,}")
    c2.metric("Amount Claimed",     f"{tot_clm:,}")
    c3.metric("Reward Txns",        f"{len(reward_txns):,}")
    c4.metric("Claim Requests",     f"{len(reward_claims):,}")

    t1, t2, t3 = st.tabs(["Credits & Claims", "Transactions", "Leaderboard"])

    with t1:
        cl, cr = st.columns(2)
        with cl:
            sec("Daily Credits Issued")
            if not reward_credits.empty and "createdAt" in reward_credits.columns:
                rc = (reward_credits.dropna(subset=["createdAt"])
                                    .set_index("createdAt").resample("D")["amount"].sum()
                                    .reset_index().rename(columns={"createdAt":"Date"}))
                fig = px.area(rc, x="Date", y="amount",
                              color_discrete_sequence=[PALETTE["green"]])
                fig.update_traces(fillcolor="rgba(63,185,80,0.08)",
                                  line=dict(width=1.5, color=PALETTE["green"]))
                fig.update_layout(**_layout(height=280), showlegend=False,
                                  title=_title("Daily reward points issued"),
                                  yaxis_title="Points")
                st.plotly_chart(fig, width="stretch")

        with cr:
            sec("Claim Status")
            if not reward_claims.empty and "status" in reward_claims.columns:
                cs = reward_claims["status"].fillna("unknown").value_counts()
                cmap_c = {"pending":PALETTE["amber"],"approved":PALETTE["green"],
                          "rejected":PALETTE["red"],"completed":PALETTE["cyan"],"unknown":"#d0d7de"}
                fig = px.pie(names=cs.index, values=cs.values, hole=0.58,
                             color=cs.index,
                             color_discrete_map={k:cmap_c.get(k,PALETTE["blue"]) for k in cs.index})
                fig.update_traces(textposition="none",
                                  hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>")
                fig.update_layout(**_layout(height=280),
                                  title=_title("Reward claim status breakdown"),
                                  legend=dict(font=dict(size=10,color="#57606a"),
                                              bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig, width="stretch")

    with t2:
        sec("Transaction Volume")
        if not reward_txns.empty and "createdAt" in reward_txns.columns:
            rt = (reward_txns.dropna(subset=["createdAt"])
                             .set_index("createdAt").resample("D")["amount"].sum()
                             .reset_index().rename(columns={"createdAt":"Date"}))
            fig = px.bar(rt, x="Date", y="amount",
                         color_discrete_sequence=[PALETTE["purple"]])
            fig.update_traces(opacity=0.7)
            fig.update_layout(**_layout(height=300), showlegend=False,
                              title=_title("Daily reward token transfer volume"),
                              yaxis_title="Amount")
            st.plotly_chart(fig, width="stretch")

        if not reward_txns.empty and "triggeredBy" in reward_txns.columns:
            sec("Trigger Types")
            tb = reward_txns["triggeredBy"].fillna("unknown").value_counts()
            fig = px.bar(x=tb.index, y=tb.values,
                         color_discrete_sequence=[PALETTE["blue"]],
                         text=[f"{v:,}" for v in tb.values])
            fig.update_traces(opacity=0.8, textposition="outside",
                              textfont=dict(color="#8b949e",size=11))
            fig.update_layout(**_layout(height=240), showlegend=False,
                              title=_title("What triggered each reward transfer"),
                              xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, width="stretch")

    with t3:
        sec("Top 20 Users by Points")
        if not leaderboard.empty:
            top_lb = (leaderboard.sort_values("points", ascending=False)
                                 .drop_duplicates("userId").head(20)
                                 .reset_index(drop=True))
            top_lb.index += 1
            fig = px.bar(top_lb, x="userName", y="points",
                         color="points",
                         color_continuous_scale=[[0,"#f6f8fa"],[0.5,PALETTE["blue"]],[1,PALETTE["cyan"]]],
                         text=[f"{v:,}" for v in top_lb["points"]])
            fig.update_traces(textposition="outside",
                              textfont=dict(color="#57606a",size=9))
            fig.update_layout(**_layout(height=340),
                              coloraxis_showscale=False,
                              title=_title("Leaderboard — top 20 by points"),
                              xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
                              showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(top_lb[["rank","userName","userEmail","points"]],
                         use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  WALLETS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Wallets":
    hdr("Wallets", "Connection status, token associations and history")

    total_w  = len(wallets)
    ready_w  = int((wallets["status"] == "ready").sum()) if not wallets.empty else 0
    nft_a    = int((wallets["nftTokenAssociated"] == "true").sum()) if not wallets.empty else 0
    rew_a    = int((wallets["rewardTokenAssociated"] == "true").sum()) if not wallets.empty else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Wallets",       f"{total_w:,}")
    c2.metric("Ready",               f"{ready_w:,}")
    c3.metric("NFT Token Assoc.",    f"{nft_a:,}")
    c4.metric("Reward Token Assoc.", f"{rew_a:,}")

    cl, cr = st.columns(2)
    with cl:
        sec("Wallet Status")
        if not wallets.empty and "status" in wallets.columns:
            ws = wallets["status"].fillna("unknown").value_counts()
            cmap_w = {"ready":PALETTE["green"],"pending_verification":PALETTE["amber"],
                      "pending_associations":PALETTE["blue"],"disconnected":"#d0d7de"}
            fig = px.pie(names=ws.index, values=ws.values, hole=0.58,
                         color=ws.index,
                         color_discrete_map={k:cmap_w.get(k,PALETTE["blue"]) for k in ws.index})
            fig.update_traces(textposition="none",
                              hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>")
            fig.update_layout(**_layout(height=300),
                              title=_title("Current wallet connection status"),
                              legend=dict(font=dict(size=10,color="#57606a"),
                                          bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, width="stretch")

    with cr:
        sec("Connections Over Time")
        if not wallets.empty and "createdAt" in wallets.columns:
            wg = (wallets.dropna(subset=["createdAt"])
                         .set_index("createdAt").resample("D")["id"].count()
                         .reset_index().rename(columns={"id":"Wallets","createdAt":"Date"}))
            fig = px.area(wg, x="Date", y="Wallets",
                          color_discrete_sequence=[PALETTE["cyan"]])
            fig.update_traces(fillcolor="rgba(121,192,255,0.08)",
                              line=dict(width=1.5,color=PALETTE["cyan"]))
            fig.update_layout(**_layout(height=300), showlegend=False,
                              title=_title("Daily wallet connections"))
            st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
#  AUDIT LOG
# ════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    hdr("Audit Log", "System event tracking and user activity")

    if audit.empty:
        st.info("No audit log data found in the dump.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Events",       f"{len(audit):,}")
        c2.metric("Unique Event Types", f"{audit['eventType'].nunique():,}" if "eventType" in audit.columns else "–")
        c3.metric("Unique Users",       f"{audit['userId'].nunique():,}"    if "userId"    in audit.columns else "–")

        cl, cr = st.columns(2)
        with cl:
            sec("Top Event Types")
            if "eventType" in audit.columns:
                et = audit["eventType"].fillna("unknown").value_counts().head(15)
                fig = px.bar(x=et.values, y=et.index, orientation="h",
                             color=et.values,
                             color_continuous_scale=[[0,"#f6f8fa"],[1,PALETTE["purple"]]],
                             text=[f"{v:,}" for v in et.values])
                fig.update_traces(textposition="outside",
                                  textfont=dict(color="#57606a",size=10))
                fig.update_layout(**_layout(height=440), showlegend=False,
                                  coloraxis_showscale=False,
                                  title=_title("Most frequent audit event types"),
                                  xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
                                  yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig, width="stretch")

        with cr:
            sec("Event Volume Over Time")
            if "createdAt" in audit.columns:
                ag = (audit.dropna(subset=["createdAt"])
                           .set_index("createdAt").resample("D")["id"].count()
                           .reset_index().rename(columns={"id":"Events","createdAt":"Date"}))
                fig = px.area(ag, x="Date", y="Events",
                              color_discrete_sequence=[PALETTE["purple"]])
                fig.update_traces(fillcolor="rgba(188,140,255,0.08)",
                                  line=dict(width=1.5,color=PALETTE["purple"]))
                fig.update_layout(**_layout(height=440), showlegend=False,
                                  title=_title("Daily audit event volume"))
                st.plotly_chart(fig, width="stretch")

        sec("Recent Events (last 50)")
        recent = (audit[["createdAt","eventType","userId","ipAddress"]]
                  .sort_values("createdAt", ascending=False).head(50)
                  .reset_index(drop=True))
        st.dataframe(recent, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  USER INVESTIGATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "User Investigator":
    hdr("User Investigator", "Deep dive search and 360-degree lookup of user profile, uploads, wallets, rewards and referrals")

    if users.empty:
        st.info("No user records found in the database.")
    else:
        # Build search selector options
        users_clean = users.copy()
        users_clean["name"] = users_clean["name"].fillna("Unknown")
        users_clean["email"] = users_clean["email"].fillna("No Email")
        users_clean["did"] = users_clean["did"].fillna("No DID")
        users_clean["mainWalletAddress"] = users_clean["mainWalletAddress"].fillna("No Wallet")
        
        # Build search labels
        search_labels = []
        for idx, row in users_clean.iterrows():
            label = f"{row['name']} ({row['email']}) | DID: {row['did'][:10]}... | Wallet: {row['mainWalletAddress'][:10]}..."
            search_labels.append((row["id"], label))
        
        # Search dropdown
        user_choice = st.selectbox(
            "Search for a user by Name, Email, DID, or Wallet Address",
            options=search_labels,
            format_func=lambda x: x[1],
            index=0
        )
        
        if user_choice:
            selected_user_id = user_choice[0]
            user_row = users[users["id"] == selected_user_id].iloc[0]
            
            # Display user details header
            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns([1.5, 1])
            
            with col_l:
                st.markdown(f"### 👤 {user_row['name'] or 'Unknown User'}")
                st.markdown(f"**Email:** {user_row['email'] or 'N/A'} (Verified: `{user_row['emailVerified']}`)")
                st.markdown(f"**DID:** `{user_row['did'] or 'N/A'}` (Status: `{user_row['didStatus']}`)")
                st.markdown(f"**Main Wallet Address:** `{user_row['mainWalletAddress'] or 'N/A'}`")
                
                # Additional details
                dob = user_row.get("dateOfBirth", None)
                dob_str = pd.to_datetime(dob).strftime("%Y-%m-%d") if pd.notna(dob) else "N/A"
                st.markdown(f"**Date of Birth:** {dob_str} | **Phone:** {user_row.get('phone', 'N/A')}")
                st.markdown(f"**Address:** {user_row.get('address', 'N/A')}")
                
            with col_r:
                sec("Key Metrics")
                m1, m2 = st.columns(2)
                m1.metric("Reward Balance", f"{user_row.get('rewardBalance', 0):,.0f}")
                m2.metric("Lifetime Points", f"{user_row.get('lifetimeRewardPoints', 0):,.0f}")
                
                role_val = user_row.get("role", "user")
                role_color = PALETTE["purple"] if role_val == "admin" else PALETTE["blue"]
                st.markdown(f"**Role:** <span style='color:{role_color}; font-weight:bold;'>{role_val.upper()}</span>", unsafe_allow_html=True)
                
                joined = user_row.get("createdAt", None)
                joined_str = pd.to_datetime(joined).strftime("%Y-%m-%d %H:%M") if pd.notna(joined) else "N/A"
                st.markdown(f"**Date Joined:** {joined_str}")
            
            st.markdown("<br><hr>", unsafe_allow_html=True)
            
            # Setup Tabs
            tab_bills, tab_wallets, tab_rewards, tab_referrals = st.tabs([
                "📄 Bills & Extractions", 
                "🪙 Wallets & NFTs", 
                "🎁 Rewards & Claims", 
                "🔗 Referral Tree"
            ])
            
            # --- TAB 1: Bills & Extractions ---
            with tab_bills:
                user_bills = bills[bills["userId"] == selected_user_id]
                if user_bills.empty:
                    st.info("This user has not uploaded any bills.")
                else:
                    st.markdown(f"**Total Bills Uploaded:** {len(user_bills)}")
                    
                    # Merge with extractions and fraud checks for details
                    bills_details = user_bills.copy()
                    
                    # Merge extractions
                    if not extractions.empty:
                        # Grab relevant extraction fields
                        ext_sub = extractions[["billId", "merchantName", "totalAmount", "taxAmount", "currency", "modelUsed"]].copy()
                        bills_details = bills_details.merge(ext_sub, left_on="id", right_on="billId", how="left").drop(columns=["billId"], errors="ignore")
                    
                    # Merge fraud
                    if not fraud.empty:
                        fraud_sub = fraud[["billId", "score", "finalResult", "decision"]].copy()
                        bills_details = bills_details.merge(fraud_sub, left_on="id", right_on="billId", how="left").drop(columns=["billId"], errors="ignore")
                    
                    # Format columns
                    bills_details["createdAt"] = bills_details["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                    display_cols = ["originalFilename", "createdAt", "status", "category", "merchantName", "totalAmount", "currency", "decision", "score"]
                    # Filter existing columns
                    display_cols = [c for c in display_cols if c in bills_details.columns]
                    
                    st.dataframe(bills_details[display_cols].sort_values("createdAt", ascending=False).reset_index(drop=True), use_container_width=True)
            
            # --- TAB 2: Wallets & NFTs ---
            with tab_wallets:
                # Wallets
                user_wallets = wallets[wallets["userId"] == selected_user_id]
                st.markdown("#### Connected Wallets")
                if user_wallets.empty:
                    st.info("No connected wallets found for this user.")
                else:
                    w_display = user_wallets.copy()
                    if "createdAt" in w_display.columns and pd.notna(w_display["createdAt"]).any():
                        w_display["createdAt"] = w_display["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                    w_cols = ["accountId", "status", "nftTokenAssociated", "rewardTokenAssociated", "createdAt"]
                    w_cols = [c for c in w_cols if c in w_display.columns]
                    st.dataframe(w_display[w_cols].reset_index(drop=True), use_container_width=True)
                
                # NFTs
                st.markdown("<br>#### Minted NFTs", unsafe_allow_html=True)
                user_nfts = nfts[nfts["userId"] == selected_user_id]
                if user_nfts.empty:
                    st.info("No minted NFTs found for this user.")
                else:
                    n_display = user_nfts.copy()
                    if "createdAt" in n_display.columns and pd.notna(n_display["createdAt"]).any():
                        n_display["createdAt"] = n_display["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                    n_display["transferredAt"] = pd.to_datetime(n_display["transferredAt"]).dt.strftime("%Y-%m-%d %H:%M").fillna("Held (Not Transferred)")
                    
                    n_cols = ["tokenId", "serialNumber", "mintTxId", "transferredAt", "createdAt"]
                    n_cols = [c for c in n_cols if c in n_display.columns]
                    st.dataframe(n_display[n_cols].reset_index(drop=True), use_container_width=True)
            
            # --- TAB 3: Rewards & Claims ---
            with tab_rewards:
                c_left, c_right = st.columns(2)
                
                with c_left:
                    st.markdown("#### Reward Credits")
                    user_credits = reward_credits[reward_credits["userId"] == selected_user_id]
                    if user_credits.empty:
                        st.info("No reward credits issued to this user.")
                    else:
                        cr_display = user_credits.copy()
                        cr_display["createdAt"] = cr_display["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                        cr_cols = ["billId", "amount", "createdAt"]
                        cr_cols = [c for c in cr_cols if c in cr_display.columns]
                        st.dataframe(cr_display[cr_cols].sort_values("createdAt", ascending=False).reset_index(drop=True), use_container_width=True)
                
                with c_right:
                    st.markdown("#### Claim Requests")
                    user_claims = reward_claims[reward_claims["userId"] == selected_user_id]
                    if user_claims.empty:
                        st.info("No reward claims requested by this user.")
                    else:
                        cl_display = user_claims.copy()
                        cl_display["createdAt"] = cl_display["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                        cl_cols = ["walletAddress", "amount", "status", "rejectionReason", "createdAt"]
                        cl_cols = [c for c in cl_cols if c in cl_display.columns]
                        st.dataframe(cl_display[cl_cols].sort_values("createdAt", ascending=False).reset_index(drop=True), use_container_width=True)
                
                # Reward transactions log
                st.markdown("<br>#### Reward Transactions Log", unsafe_allow_html=True)
                user_txns = reward_txns[reward_txns["userId"] == selected_user_id]
                if user_txns.empty:
                    st.info("No reward transfer transactions recorded for this user.")
                else:
                    tx_display = user_txns.copy()
                    tx_display["createdAt"] = tx_display["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                    tx_cols = ["walletAccountId", "amount", "triggeredBy", "transferTxId", "createdAt"]
                    tx_cols = [c for c in tx_cols if c in tx_display.columns]
                    st.dataframe(tx_display[tx_cols].sort_values("createdAt", ascending=False).reset_index(drop=True), use_container_width=True)
            
            # --- TAB 4: Referral Tree ---
            with tab_referrals:
                st.markdown("#### Referral Relationship")
                
                ref_code = user_row.get("referralCode", None)
                referred_by = user_row.get("referredBy", None)
                
                col_tree_l, col_tree_r = st.columns(2)
                
                with col_tree_l:
                    st.markdown("**User's Referral Code:**")
                    if ref_code:
                        st.code(ref_code)
                    else:
                        st.write("No referral code generated.")
                        
                    st.markdown("**Referred By:**")
                    if referred_by:
                        # Find referring user details
                        referrer_match = users[users["referralCode"] == referred_by]
                        if not referrer_match.empty:
                            ref_user = referrer_match.iloc[0]
                            st.write(f"👤 {ref_user['name']} ({ref_user['email']})")
                            st.code(f"Code: {referred_by}")
                        else:
                            st.write(f"Unknown Referrer (Code: `{referred_by}`)")
                    else:
                        st.write("Direct signup (not referred by anyone).")
                
                with col_tree_r:
                    # Users referred by this user
                    referred_users = users[users["referredBy"] == ref_code] if ref_code else pd.DataFrame()
                    st.markdown(f"**Users Referred by this User:** {len(referred_users)}")
                    if referred_users.empty:
                        st.info("This user has not referred anyone yet.")
                    else:
                        ru_display = referred_users[["name", "email", "createdAt", "lifetimeRewardPoints"]].copy()
                        ru_display["createdAt"] = ru_display["createdAt"].dt.strftime("%Y-%m-%d %H:%M")
                        st.dataframe(ru_display.reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
#  ML INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "ML Insights":
    hdr("ML Insights", "Machine learning analytics for user behavior and forecasting")

    t1, t2 = st.tabs(["User Segmentation (K-Means)", "Upload Forecasting (Linear Regression)"])

    with t1:
        sec("User Segmentation")
        st.markdown("Groups users into 3 clusters based on total reward balance and lifetime reward points.")
        if not users.empty and "rewardBalance" in users.columns and "lifetimeRewardPoints" in users.columns:
            # Prepare data for K-Means
            cluster_data = users[["id", "name", "rewardBalance", "lifetimeRewardPoints"]].dropna()
            if len(cluster_data) > 3:
                X = cluster_data[["rewardBalance", "lifetimeRewardPoints"]]
                
                # Fit K-Means
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_data["Cluster"] = kmeans.fit_predict(X).astype(str)
                
                # Plot
                fig = px.scatter(
                    cluster_data, x="lifetimeRewardPoints", y="rewardBalance",
                    color="Cluster", hover_data=["name"],
                    color_discrete_sequence=[PALETTE["blue"], PALETTE["green"], PALETTE["purple"]]
                )
                fig.update_layout(**_layout(height=400), title=_title("User Clusters (Reward Balance vs Lifetime Points)"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough user data for clustering.")
        else:
            st.info("Required user data not available.")

    with t2:
        sec("Upload Forecasting")
        st.markdown("Predicts the next 30 days of bill uploads based on historical trends.")
        if not bills.empty and "createdAt" in bills.columns:
            # Prepare daily upload counts
            bg = bills.dropna(subset=["createdAt"]).set_index("createdAt").resample("D")["id"].count().reset_index()
            bg.columns = ["Date", "Uploads"]
            bg = bg.sort_values("Date")
            
            if len(bg) > 2:
                # Convert dates to ordinal for Linear Regression
                bg["DayOrdinal"] = bg["Date"].map(pd.Timestamp.toordinal)
                X = bg[["DayOrdinal"]]
                y = bg["Uploads"]
                
                # Fit Linear Regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Create future dates
                last_date = bg["Date"].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
                future_df = pd.DataFrame({"Date": future_dates})
                future_df["DayOrdinal"] = future_df["Date"].map(pd.Timestamp.toordinal)
                
                # Predict
                future_df["Uploads"] = model.predict(future_df[["DayOrdinal"]])
                future_df["Uploads"] = future_df["Uploads"].clip(lower=0) # No negative uploads
                future_df["Type"] = "Predicted"
                
                bg["Type"] = "Historical"
                
                # Combine
                combined = pd.concat([bg[["Date", "Uploads", "Type"]], future_df[["Date", "Uploads", "Type"]]])
                
                # Plot
                fig = px.line(
                    combined, x="Date", y="Uploads", color="Type",
                    color_discrete_map={"Historical": PALETTE["blue"], "Predicted": PALETTE["amber"]}
                )
                # Make predicted line dashed
                fig.update_traces(patch={"line": {"dash": "dash"}}, selector={"legendgroup": "Predicted"})
                
                fig.update_layout(**_layout(height=400), title=_title("30-Day Upload Forecast"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough historical data for forecasting.")
        else:
            st.info("Required bill data not available.")
