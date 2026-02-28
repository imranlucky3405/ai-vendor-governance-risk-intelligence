import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# Helper: normalize columns for fair comparison in driver charts
# -----------------------------
def normalized_means(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    tmp = frame[cols].copy()
    # min-max normalization per column; safe for constant columns
    for c in cols:
        mn, mx = tmp[c].min(), tmp[c].max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            tmp[c] = 0.0
        else:
            tmp[c] = (tmp[c] - mn) / (mx - mn)
    out = tmp.mean().reset_index()
    out.columns = ["Risk Factor", "Normalized Impact (0–1)"]
    return out

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Vendor Governance & Risk Intelligence",
    layout="wide"
)

# Cache clear button (Streamlit >= 1.18)
st.sidebar.button(
    "🔄 Clear Cache",
    on_click=lambda: (st.cache_data.clear(), st.cache_resource.clear())
)

# -----------------------------
# Session state defaults
# -----------------------------
if "THRESHOLD" not in st.session_state:
    st.session_state["THRESHOLD"] = 0.70

THRESHOLD = st.session_state["THRESHOLD"]

if "annual_cost_of_capital" not in st.session_state:
    st.session_state["annual_cost_of_capital"] = 0.12  # 12%
if "delay_cost_rate_per_day" not in st.session_state:
    st.session_state["delay_cost_rate_per_day"] = 0.0005  # 0.05% per day
if "penalty_rate" not in st.session_state:
    st.session_state["penalty_rate"] = 0.005  # 0.5%
if "mitigation_effectiveness" not in st.session_state:
    st.session_state["mitigation_effectiveness"] = 0.30  # 30%

# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def load_model():
    model_artifact = joblib.load("models/risk_model.pkl")

    model = model_artifact.get("model")
    features = model_artifact.get("features")
    metrics = model_artifact.get("metrics", None)

    return model, features, metrics

# ==========================================================
# Load Data
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("processed/vendor_vpi_scores.csv")

    # Clean object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].fillna("").astype("object")

    return df

model, features, metrics = load_model()
df = load_data()

# ==========================================================
# Load Delivery History
# ==========================================================
@st.cache_data
def load_delivery_history():
    dh = pd.read_csv("data/raw/vendor_delivery_history_scaled.csv")
    dh["delivery_month"] = pd.to_datetime(dh["delivery_month"])
    return dh

delivery_df = load_delivery_history()

# -----------------------------
# Data freshness & activity helpers
# -----------------------------
data_max_month = pd.to_datetime(delivery_df["delivery_month"]).max()
data_min_month = pd.to_datetime(delivery_df["delivery_month"]).min()

vendor_last_month = (
    delivery_df.groupby("vendor_id")["delivery_month"]
    .max()
    .reset_index()
    .rename(columns={"delivery_month": "last_delivery_month"})
)
df = df.merge(vendor_last_month, on="vendor_id", how="left")

# ==========================================================
# Predict Deterioration Risk (do this BEFORE action system)
# ==========================================================
df["low_vpi_risk_prob"] = model.predict_proba(df[features])[:, 1]

# ==========================================================
# ✅ Vectorized Unified Action System (FAST)
# Replaces: df[action_cols] = df.apply(build_action_row, axis=1)
# ==========================================================
# Safe numeric fallbacks (handles missing cols + NaNs)
risk_prob = pd.to_numeric(df["low_vpi_risk_prob"], errors="coerce").fillna(0.0)
avg_delay = pd.to_numeric(df.get("avg_delay_days", 0.0), errors="coerce").fillna(0.0)
pay_delay = pd.to_numeric(df.get("avg_payment_delay", 0.0), errors="coerce").fillna(0.0)
rej_rate = pd.to_numeric(df.get("rejection_rate_pct", 0.0), errors="coerce").fillna(0.0)
penalty_cases = pd.to_numeric(df.get("penalty_cases", 0.0), errors="coerce").fillna(0.0)

# Priority rules (vectorized)
cond_p0 = (risk_prob >= 0.80) | ((avg_delay > 5) & (rej_rate > 7)) | ((pay_delay > 10) & (penalty_cases > 0))
cond_p1 = (risk_prob >= 0.60) | (avg_delay > 5) | (pay_delay > 10) | (rej_rate > 7) | (penalty_cases > 1)
cond_p2 = (risk_prob >= 0.40) | (avg_delay > 2) | (pay_delay > 5) | (rej_rate > 3)

df["Action_Priority"] = np.select(
    [cond_p0, cond_p1, cond_p2],
    ["P0 (Critical)", "P1 (High)", "P2 (Medium)"],
    default="P3 (Low)"
)

# Reasons (top-2, executive-readable) - fully vectorized
# Ordered conditions = "importance / exec readability"
r_pred = (risk_prob >= 0.60)
r_delay_sev = (avg_delay > 5)
r_delay_mod = (avg_delay > 2) & ~r_delay_sev
r_pay_sev = (pay_delay > 10)
r_pay_mod = (pay_delay > 5) & ~r_pay_sev
r_qual_hi = (rej_rate > 7)
r_qual_mod = (rej_rate > 3) & ~r_qual_hi
r_penalty = (penalty_cases > 0)

reason_order = [
    (r_pred, "Predicted deterioration risk"),
    (r_delay_sev, "Severe delivery delay"),
    (r_delay_mod, "Moderate delivery delay"),
    (r_pay_sev, "Payment backlog severity"),
    (r_pay_mod, "Payment delays"),
    (r_qual_hi, "High rejection / quality risk"),
    (r_qual_mod, "Moderate quality risk"),
    (r_penalty, "Penalty exposure"),
]

conds1 = [c for c, _ in reason_order]
texts1 = [t for _, t in reason_order]
first_reason = np.select(conds1, texts1, default="")

conds2 = [c & (first_reason != t) for c, t in reason_order]
second_reason = np.select(conds2, texts1, default="")

# Build "A + B" safely
reason_text = np.where(
    first_reason == "",
    "Early risk signals",
    np.where(second_reason == "", first_reason, first_reason + " + " + second_reason)
)
df["Action_Reason"] = reason_text

# Recommended action (vectorized by priority)
df["Recommended_Action"] = df["Action_Priority"].map({
    "P0 (Critical)": "War-room: Ops+SCM leadership review (weekly)",
    "P1 (High)": "Intervene within 30 days: stabilize plan + clear backlog",
    "P2 (Medium)": "Preventive fixes: improve discipline + monitor weekly",
    "P3 (Low)": "Monitor"
}).fillna("Monitor")

# Owner (vectorized map)
df["Action_Owner"] = df["Action_Priority"].map({
    "P0 (Critical)": "Ops+SCM Head",
    "P1 (High)": "Vendor Manager",
    "P2 (Medium)": "Ops/SCM SPOC",
    "P3 (Low)": "SPOC"
}).fillna("SPOC")

# Status (constant)
df["Action_Status"] = "Open"

# Due date (vectorized: today + due_days)
due_days = df["Action_Priority"].map({
    "P0 (Critical)": 7,
    "P1 (High)": 30,
    "P2 (Medium)": 60,
    "P3 (Low)": 90
}).fillna(60).astype(int)

df["Action_DueDate"] = (
    pd.Timestamp.today().normalize() + pd.to_timedelta(due_days, unit="D")
).dt.date.astype(str)

# ==========================================================
# Active vendor filter (based on last delivery month)
# ==========================================================
active_window = st.sidebar.selectbox(
    "Active vendor window (based on last delivery month)",
    ["All", "Last 3 months", "Last 6 months", "Last 12 months", "Last 24 months"],
    index=0
)

if active_window != "All" and pd.notna(data_max_month):
    months = int(active_window.split()[1])
    cutoff = pd.to_datetime(data_max_month) - pd.DateOffset(months=months)
    df = df[df["last_delivery_month"] >= cutoff].copy()

# ==========================================================
# Create Monthly Trend Dataset
# ==========================================================
@st.cache_data
def create_monthly_trend(delivery_df):
    monthly = (
        delivery_df.groupby("delivery_month")
        .agg({
            "delivery_delay_days": "mean"
        })
        .reset_index()
        .sort_values("delivery_month")
    )

    # Create delay frequency per month
    delay_flag = delivery_df.copy()
    delay_flag["delay_flag"] = (delay_flag["delivery_delay_days"] > 0).astype(int)

    delay_freq = (
        delay_flag.groupby("delivery_month")["delay_flag"]
        .mean()
        .reset_index()
    )

    monthly = monthly.merge(delay_freq, on="delivery_month")

    # Create simplified monthly risk score
    monthly["monthly_risk_index"] = (
        monthly["delivery_delay_days"] * 0.7 +
        monthly["delay_flag"] * 0.3
    )

    return monthly

monthly_trend = create_monthly_trend(delivery_df)

# ==========================================================
# Predict Deterioration Risk (flags & buckets)
# ==========================================================
df["deterioration_risk_flag"] = (df["low_vpi_risk_prob"] >= st.session_state["THRESHOLD"]).astype(int)

def risk_bucket(prob: float) -> str:
    if prob >= 0.80:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.40:
        return "Moderate"
    else:
        return "Low"

df["risk_level"] = df["low_vpi_risk_prob"].apply(risk_bucket)

# ==========================================================
# Sidebar Filters
# ==========================================================
st.sidebar.header("Filters")

risk_filter = st.sidebar.multiselect(
    "VPI Category",
    options=sorted(df["VPI_Category"].unique()),
    default=sorted(df["VPI_Category"].unique())
)

df = df[df["VPI_Category"].isin(risk_filter)].copy()

st.sidebar.subheader("Action Filters")

show_actions = st.sidebar.selectbox(
    "Show actions",
    ["All", "Open only"],
    index=1
)

priority_filter = st.sidebar.multiselect(
    "Priority",
    options=["P0 (Critical)", "P1 (High)", "P2 (Medium)", "P3 (Low)"],
    default=["P0 (Critical)", "P1 (High)", "P2 (Medium)"]
)

if show_actions == "Open only":
    df = df[df["Action_Status"] == "Open"].copy()

df = df[df["Action_Priority"].isin(priority_filter)].copy()

dashboard_view = st.sidebar.radio(
    "Select Dashboard",
    [
        "🏁 Executive Summary",
        "🔧 Operations Dashboard",
        "💰 SCM Dashboard",
        "🚨 Predictive Risk Dashboard",
        "💸 Financial Impact",
        "📈 Delivery Time-Series Intelligence"
    ]
)

# ==========================================================
# 🏁 Executive SUMMARY (1-page executive view)
# ==========================================================
if dashboard_view == "🏁 Executive Summary":

    st.title("Executive Summary – Vendor Governance & Risk Intelligence")

    # -----------------------------
    # Data freshness banner
    # -----------------------------
    st.info(
        f"📅 Data window: {data_min_month:%b %Y} → {data_max_month:%b %Y} | "
        f"Last available month: {data_max_month:%b %Y}"
    )

    if pd.Timestamp.today().normalize() - pd.to_datetime(data_max_month).normalize() > pd.Timedelta(days=180):
        st.warning(
            "⚠️ Data is older than ~6 months. Use this as governance baseline. "
            "For live tracking, refresh with latest 6–12 months data."
        )

    # -----------------------------
    # Headline KPIs (simple, Sr. Management-friendly)
    # -----------------------------
    total_vendors = int(df.shape[0])
    low_vpi = int((df["VPI_Category"] == "Low").sum())

    # Predictive counts
    threshold_val = float(st.session_state["THRESHOLD"])
    above_threshold = int((df["low_vpi_risk_prob"] >= threshold_val).sum())
    critical = int((df["low_vpi_risk_prob"] >= 0.80).sum())
    high = int(((df["low_vpi_risk_prob"] >= 0.60) & (df["low_vpi_risk_prob"] < 0.80)).sum())

    # Ops / SCM snapshots
    avg_ops_delay = float(df["avg_delay_days"].mean())
    avg_rejection = float(df["rejection_rate_pct"].mean())
    avg_pay_delay = float(df["avg_payment_delay"].mean())
    pct_invoices_delayed = float(df["payment_risk_ratio"].mean()) * 100.0

    # Working capital exposure proxy
    annual_cost_of_capital = float(st.session_state.get("annual_cost_of_capital", 0.12))
    df["_wc_exposure_lakhs"] = (
        df["contract_value_lakhs"]
        * (annual_cost_of_capital / 365.0)
        * df["avg_payment_delay"].clip(lower=0)
    )
    total_wc_exposure = float(df["_wc_exposure_lakhs"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vendors", total_vendors)
    c2.metric("Low VPI Vendors", low_vpi)
    c3.metric("Above Risk Threshold", above_threshold)
    c4.metric("Critical / High", f"{critical} / {high}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Delivery Delay (Days)", round(avg_ops_delay, 2))
    c6.metric("Avg Rejection Rate (%)", round(avg_rejection, 2))
    c7.metric("Avg Payment Delay (Days)", round(avg_pay_delay, 2))
    c8.metric("% Invoices Delayed", f"{pct_invoices_delayed:.1f}%")

    c9, _ = st.columns([1, 3])
    c9.metric("Working Capital Exposure (₹ Lakhs, proxy)", round(total_wc_exposure, 2))

    st.divider()

    # -----------------------------
    # 1) Portfolio Trend (Delay)
    # -----------------------------
    st.subheader("Portfolio Trend – Delivery Delay (Monthly)")

    fig_trend = px.line(
        monthly_trend.sort_values("delivery_month"),
        x="delivery_month",
        y="delivery_delay_days",
        markers=True,
        title="Average Delivery Delay Over Time"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # -----------------------------
    # 2) Hotspots (Region / Vendor Type)
    # -----------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("Hotspots – Delay by Region (Days)")
        if "location" in df.columns:
            region_delay = (
                df.groupby("location")["avg_delay_days"]
                .mean()
                .reset_index()
                .sort_values("avg_delay_days", ascending=False)
                .head(8)
            )
            fig_region = px.bar(
                region_delay,
                x="location",
                y="avg_delay_days",
                title="Top Regions by Avg Delivery Delay (Days)"
            )
            st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.info("Region data not available.")

    with right:
        st.subheader("Hotspots – Exposure by Vendor Type (₹ Lakhs)")
        if "vendor_type" in df.columns:
            type_expo = (
                df.groupby("vendor_type")["_wc_exposure_lakhs"]
                .sum()
                .reset_index()
                .sort_values("_wc_exposure_lakhs", ascending=False)
                .head(8)
            )
            type_expo = type_expo.rename(columns={"_wc_exposure_lakhs": "Exposure (₹ Lakhs)"})

            fig_type = px.bar(
                type_expo,
                x="vendor_type",
                y="Exposure (₹ Lakhs)",
                title="Top Vendor Types by Working Capital Exposure (₹ Lakhs)"
            )
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("Vendor type data not available.")

    st.divider()

    # -----------------------------
    # 3) Executive Watchlist – Top 10
    # -----------------------------
    st.subheader("Executive Watchlist – Top 10 Vendors to Intervene")

    def ceo_reason(row):
        reasons = []
        if row["low_vpi_risk_prob"] >= 0.80:
            reasons.append("Critical predicted deterioration")
        if row["avg_delay_days"] > 5:
            reasons.append("Severe delivery delays")
        if row["rejection_rate_pct"] > 7:
            reasons.append("Quality rejection risk")
        if row["avg_payment_delay"] > 10:
            reasons.append("Payment backlog severity")
        if row.get("penalty_cases", 0) > 0:
            reasons.append("Penalty exposure")
        return ", ".join(reasons) if reasons else "Early risk signals"

    def ceo_action(row):
        if row["low_vpi_risk_prob"] >= 0.80:
            return "War-room: Ops+SCM leadership review (weekly)"
        if row["low_vpi_risk_prob"] >= 0.60:
            return "Preventive intervention within 30 days"
        return "Enhanced monitoring"

    watch = df.sort_values("low_vpi_risk_prob", ascending=False).head(10).copy()
    watch["Why"] = watch.apply(ceo_reason, axis=1)
    watch["Action"] = watch.apply(ceo_action, axis=1)

    show = watch[[
        "vendor_name",
        "VPI_Category",
        "low_vpi_risk_prob",
        "avg_delay_days",
        "rejection_rate_pct",
        "avg_payment_delay",
        "_wc_exposure_lakhs",
        "Why",
        "Action"
    ]].copy()

    show = show.rename(columns={
        "vendor_name": "Vendor",
        "low_vpi_risk_prob": "Risk Probability",
        "_wc_exposure_lakhs": "WC Exposure (₹ Lakhs)"
    })

    show["Risk Probability"] = show["Risk Probability"].round(2)
    show["WC Exposure (₹ Lakhs)"] = show["WC Exposure (₹ Lakhs)"].round(2)

    st.dataframe(show, use_container_width=True)

    # -----------------------------
    # Download Section (Executive Export)
    # -----------------------------
    kpi_df = pd.DataFrame([{
        "Data_Window_Start": data_min_month.strftime("%b %Y"),
        "Data_Window_End": data_max_month.strftime("%b %Y"),
        "Total_Vendors": total_vendors,
        "Low_VPI_Vendors": low_vpi,
        "Above_Risk_Threshold": above_threshold,
        "Critical_Vendors": critical,
        "High_Vendors": high,
        "Avg_Delivery_Delay_Days": round(avg_ops_delay, 2),
        "Avg_Rejection_Rate_Pct": round(avg_rejection, 2),
        "Avg_Payment_Delay_Days": round(avg_pay_delay, 2),
        "Pct_Invoices_Delayed": round(pct_invoices_delayed, 1),
        "Working_Capital_Exposure_Lakhs_proxy": round(total_wc_exposure, 2),
        "Risk_Threshold": threshold_val
    }])

    b1, b2 = st.columns(2)

    with b1:
        st.download_button(
            label="⬇️ Download CEO Watchlist (Top 10) – CSV",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name=f"CEO_Watchlist_Top10_{data_max_month:%Y_%m}.csv",
            mime="text/csv"
        )

    with b2:
        st.download_button(
            label="⬇️ Download CEO KPI Snapshot – CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name=f"CEO_KPI_Snapshot_{data_max_month:%Y_%m}.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("So What / Now What")

    st.success(
        "✅ Recommended focus this month:\n"
        "1) Tackle top priority vendors in the Watchlist to protect SLA + revenue.\n"
        "2) Clear SCM payment backlog for high exposure vendors to reduce working capital drag.\n"
        "3) Run corrective quality actions for high rejection vendors to prevent penalties & repeats."
    )

# ==========================================================
# 🔧 OPERATIONS DASHBOARD (Executive – Advanced)
# ==========================================================
if dashboard_view == "🔧 Operations Dashboard":

    st.title("Operations – Executive Delivery & Quality Dashboard")

    st.info(
        f"📅 Data window: {data_min_month:%b %Y} → {data_max_month:%b %Y} | "
        f"Last available month: {data_max_month:%b %Y}"
    )
    if pd.Timestamp.today().normalize() - pd.to_datetime(data_max_month).normalize() > pd.Timedelta(days=180):
        st.warning(
            "⚠️ The underlying delivery history is older than ~6 months. "
            "Use this as a baseline/vendor governance review, not as a live operations tracker."
        )

    high_risk_low_vpi = int((df["VPI_Category"] == "Low").sum())
    avg_delay = float(df["avg_delay_days"].mean())
    avg_rejection = float(df["rejection_rate_pct"].mean())
    chronic_delay_vendors = int((df["delay_frequency"] >= 0.50).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low VPI Vendors (Performance Concern)", high_risk_low_vpi)
    c2.metric("Avg Delivery Delay (Days)", round(avg_delay, 2))
    c3.metric("Avg Rejection Rate (%)", round(avg_rejection, 2))
    c4.metric("Chronic Delay Vendors (≥50% frequency)", chronic_delay_vendors)

    st.caption(
        "Interpretation: Avg Delay = severity (days). Delay Frequency = how often delays occur. "
        "Rejection Rate = quality pain indicator."
    )

    st.subheader("Trend – Average Delivery Delay Over Time (Portfolio)")

    fig_delay = px.line(
        monthly_trend,
        x="delivery_month",
        y="delivery_delay_days",
        markers=True,
        title="Average Delivery Delay Over Time"
    )
    st.plotly_chart(fig_delay, use_container_width=True)

    def rag_delay(x: float) -> str:
        if x > 5:
            return "🔴 High"
        elif x > 2:
            return "🟡 Moderate"
        else:
            return "🟢 Stable"

    def rag_quality(x: float) -> str:
        if x > 7:
            return "🔴 High"
        elif x > 3:
            return "🟡 Moderate"
        else:
            return "🟢 Stable"

    df["Delay_RAG"] = df["avg_delay_days"].apply(rag_delay)
    df["Quality_RAG"] = df["rejection_rate_pct"].apply(rag_quality)

    st.subheader("RAG View – Operations Risk Posture (Vendor Count)")

    rag1 = df["Delay_RAG"].value_counts().reset_index()
    rag1.columns = ["Delay Severity Status", "Vendors"]
    fig_rag_delay = px.bar(
        rag1,
        x="Delay Severity Status",
        y="Vendors",
        title="Vendors by Delivery Delay Severity (RAG)"
    )
    st.plotly_chart(fig_rag_delay, use_container_width=True)

    rag2 = df["Quality_RAG"].value_counts().reset_index()
    rag2.columns = ["Quality Status", "Vendors"]
    fig_rag_q = px.bar(
        rag2,
        x="Quality Status",
        y="Vendors",
        title="Vendors by Quality Rejection Risk (RAG)"
    )
    st.plotly_chart(fig_rag_q, use_container_width=True)

    st.subheader("Priority Map – Delay Severity vs Quality Rejection")

    delay_cut = 5.0
    quality_cut = 7.0

    bubble_choice = st.selectbox(
        "Bubble size",
        ["Contract Value (₹ Lakhs)", "No Bubble (uniform)"],
        index=0,
        key="ops_bubble_choice"
    )
    size_col = "contract_value_lakhs" if bubble_choice == "Contract Value (₹ Lakhs)" else None

    fig_quad = px.scatter(
        df,
        x="avg_delay_days",
        y="rejection_rate_pct",
        size=size_col,
        hover_name="vendor_name",
        hover_data={
            "avg_delay_days": ":.2f",
            "rejection_rate_pct": ":.2f",
            "delay_frequency": ":.2f",
            "contract_value_lakhs": True,
            "VPI_Category": True,
            "Action_Priority": True,
            "Action_Status": True
        },
        title="Delivery Delay (Days) vs Quality Rejection (%)",
        labels={
            "avg_delay_days": "Avg Delay (Days)",
            "rejection_rate_pct": "Rejection Rate (%)"
        }
    )

    fig_quad.add_vline(x=delay_cut, line_width=2, line_dash="dash")
    fig_quad.add_hline(y=quality_cut, line_width=2, line_dash="dash")

    xmax = float(df["avg_delay_days"].max()) if df.shape[0] else delay_cut * 2
    ymax = float(df["rejection_rate_pct"].max()) if df.shape[0] else quality_cut * 2

    xmax = max(xmax, delay_cut * 1.5)
    ymax = max(ymax, quality_cut * 1.5)

    fig_quad.update_xaxes(range=[0, xmax * 1.10])
    fig_quad.update_yaxes(range=[0, ymax * 1.10])

    fig_quad.add_annotation(
        x=delay_cut + (xmax - delay_cut) * 0.50,
        y=quality_cut + (ymax - quality_cut) * 0.50,
        text="🟥 Q1: Critical",
        showarrow=False,
        align="center"
    )
    fig_quad.add_annotation(
        x=delay_cut * 0.50,
        y=quality_cut + (ymax - quality_cut) * 0.50,
        text="🟥 Q2: Quality Risk",
        showarrow=False,
        align="center"
    )
    fig_quad.add_annotation(
        x=delay_cut * 0.50,
        y=quality_cut * 0.50,
        text="🟩 Q3: Stable",
        showarrow=False,
        align="center"
    )
    fig_quad.add_annotation(
        x=delay_cut + (xmax - delay_cut) * 0.50,
        y=quality_cut * 0.50,
        text="🟧 Q4: Delivery Risk",
        showarrow=False,
        align="center"
    )

    fig_quad.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_quad, use_container_width=True)

    st.caption(
        f"Quadrant cutoffs: Delay ≥ {delay_cut} days | Rejection ≥ {quality_cut}%. "
        "Top-right quadrant vendors are highest priority."
    )

    def ops_action_exec(row):
        if row["avg_delay_days"] > 5 and row["rejection_rate_pct"] > 7:
            return "War-room: stabilize delivery + quality audit (weekly review)"
        if row["avg_delay_days"] > 5:
            return "Re-baseline delivery plan + daily tracking until stable"
        if row["rejection_rate_pct"] > 7:
            return "Quality audit + corrective action plan (CAPA)"
        if row["delay_frequency"] >= 0.50:
            return "Fix chronic delays: resource + scheduling discipline"
        return "Monitor"

    df["Ops_Executive_Action"] = df.apply(ops_action_exec, axis=1)

    st.subheader("Top Operations-Risk Vendors (Prioritized List)")

    df["_ops_priority_score"] = (
        df["avg_delay_days"].clip(lower=0) * 0.45
        + (df["rejection_rate_pct"].clip(lower=0) / 10.0) * 0.30
        + df["delay_frequency"].clip(lower=0) * 0.20
        + (
            df["contract_value_lakhs"].fillna(0)
            / (df["contract_value_lakhs"].max() if df["contract_value_lakhs"].max() else 1)
        ) * 0.05
    )

    top_ops = df.sort_values("_ops_priority_score", ascending=False).head(20).copy()

    top_ops_display = top_ops[[
        "vendor_name",
        "VPI_Category",
        "avg_delay_days",
        "delay_frequency",
        "rejection_rate_pct",
        "contract_value_lakhs",
        "last_delivery_month",
        "Ops_Executive_Action"
    ]].copy()

    top_ops_display.rename(columns={
        "avg_delay_days": "Avg Delay (Days)",
        "delay_frequency": "Delay Frequency (0–1)",
        "rejection_rate_pct": "Rejection Rate (%)",
        "contract_value_lakhs": "Contract Value (₹ Lakhs)",
        "last_delivery_month": "Last Delivery Month",
        "Ops_Executive_Action": "Recommended Action"
    }, inplace=True)

    top_ops_display["Last Delivery Month"] = pd.to_datetime(top_ops_display["Last Delivery Month"]).dt.strftime("%b %Y")
    top_ops_display["Delay Frequency (%)"] = (top_ops_display["Delay Frequency (0–1)"] * 100).round(1)
    top_ops_display.drop(columns=["Delay Frequency (0–1)"], inplace=True)

    st.dataframe(top_ops_display, use_container_width=True)

    if "location" in df.columns:
        st.subheader("Delay Hotspots by Region")
        region_delay = (
            df.groupby("location")["avg_delay_days"]
            .mean()
            .reset_index()
            .sort_values("avg_delay_days", ascending=False)
        )
        fig_region = px.bar(
            region_delay,
            x="location",
            y="avg_delay_days",
            title="Average Delivery Delay by Region (Days)"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    if "vendor_type" in df.columns:
        st.subheader("Delay Hotspots by Vendor Type")
        type_delay = (
            df.groupby("vendor_type")["avg_delay_days"]
            .mean()
            .reset_index()
            .sort_values("avg_delay_days", ascending=False)
        )
        fig_type = px.bar(
            type_delay,
            x="vendor_type",
            y="avg_delay_days",
            title="Average Delivery Delay by Vendor Type (Days)"
        )
        st.plotly_chart(fig_type, use_container_width=True)

    st.success(
        "✅ Executive summary: Prioritize vendors in the top-right quadrant (high delay + high rejection). "
        "Next, fix chronic delay vendors (high frequency)."
    )

# ==========================================================
# 💰 SCM DASHBOARD (Executive – Advanced)
# ==========================================================
elif dashboard_view == "💰 SCM Dashboard":

    st.title("SCM – Executive Payment Risk Dashboard")

    st.info(
        f"📅 Data window: {data_min_month:%b %Y} → {data_max_month:%b %Y} | "
        f"Last available month: {data_max_month:%b %Y}"
    )
    if pd.Timestamp.today().normalize() - pd.to_datetime(data_max_month).normalize() > pd.Timedelta(days=180):
        st.warning(
            "⚠️ The underlying history is older than ~6 months. "
            "Use this as a baseline/vendor governance review, not as a live escalation tracker."
        )

    avg_delay_days = float(df["avg_payment_delay"].mean())
    pct_invoices_delayed = float(df["payment_risk_ratio"].mean()) * 100
    vendors_with_penalty = int((df.get("penalty_cases", 0) > 0).sum())
    total_penalty_cases = int(df.get("penalty_cases", pd.Series([0]*len(df))).sum())

    annual_cost_of_capital = float(st.session_state.get("annual_cost_of_capital", 0.12))
    df["_working_capital_cost_lakhs"] = (
        df["contract_value_lakhs"] * (annual_cost_of_capital / 365.0) * df["avg_payment_delay"].clip(lower=0)
    )
    total_wc_exposure = float(df["_working_capital_cost_lakhs"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Payment Delay (Days)", round(avg_delay_days, 2))
    c2.metric("% Invoices Delayed", f"{pct_invoices_delayed:.1f}%")
    c3.metric("Vendors with Penalty", vendors_with_penalty)
    c4.metric("Total Penalty Cases", total_penalty_cases)
    c5.metric("Working Capital Exposure (₹ Lakhs, proxy)", round(total_wc_exposure, 2))

    st.caption(
        "Interpretation: Avg Payment Delay = severity (days). % Invoices Delayed = frequency. "
        "Exposure is a proxy using Contract Value × (Cost of Capital/365) × Delay Days."
    )

    def rag_delay(x: float) -> str:
        if x > 8:
            return "🔴 High"
        elif x > 7:
            return "🟡 Moderate"
        else:
            return "🟢 Stable"

    def rag_frequency(x: float) -> str:
        if x >= 0.60:
            return "🔴 High"
        elif x >= 0.30:
            return "🟡 Moderate"
        else:
            return "🟢 Stable"

    df["Delay_Severity_RAG"] = df["avg_payment_delay"].apply(rag_delay)
    df["Delay_Frequency_RAG"] = df["payment_risk_ratio"].apply(rag_frequency)

    st.subheader("RAG View – Payment Risk Posture (Vendor Count)")

    rag_dist = df["Delay_Severity_RAG"].value_counts().reset_index()
    rag_dist.columns = ["Status", "Vendors"]
    fig_rag = px.bar(
        rag_dist,
        x="Status",
        y="Vendors",
        title="Vendors by Payment Delay Severity (RAG)"
    )
    st.plotly_chart(fig_rag, use_container_width=True)

    st.subheader("Priority Map – Payment Delay Severity vs Delay Frequency")

    delay_cut = 10.0
    freq_cut = 0.60

    bubble_choice = st.selectbox(
        "Bubble size",
        ["Contract Value (₹ Lakhs)", "No Bubble (uniform)"],
        index=0,
        key="scm_bubble_choice"
    )
    size_col = "contract_value_lakhs" if bubble_choice == "Contract Value (₹ Lakhs)" else None

    fig_quad = px.scatter(
        df,
        x="avg_payment_delay",
        y="payment_risk_ratio",
        size=size_col,
        hover_name="vendor_name",
        hover_data={
            "avg_payment_delay": ":.2f",
            "payment_risk_ratio": ":.2f",
            "penalty_cases": True if "penalty_cases" in df.columns else False,
            "contract_value_lakhs": True,
            "_working_capital_cost_lakhs": ":.2f",
            "VPI_Category": True,
            "Action_Priority": True,
            "Action_Status": True
        },
        title="Payment Delay (Days) vs Delay Frequency (Ratio)",
        labels={
            "avg_payment_delay": "Avg Payment Delay (Days)",
            "payment_risk_ratio": "Delay Frequency (Ratio)"
        }
    )

    fig_quad.add_vline(x=delay_cut, line_width=2, line_dash="dash")
    fig_quad.add_hline(y=freq_cut, line_width=2, line_dash="dash")

    xmax = float(df["avg_payment_delay"].max()) if df.shape[0] else delay_cut * 2
    ymax = float(df["payment_risk_ratio"].max()) if df.shape[0] else freq_cut * 2

    xmax = max(xmax, delay_cut * 1.5)
    ymax = max(ymax, freq_cut * 1.5)

    fig_quad.update_xaxes(range=[0, xmax * 1.10])
    fig_quad.update_yaxes(range=[0, min(1.0, ymax * 1.10)])

    fig_quad.add_annotation(
        x=delay_cut + (xmax - delay_cut) * 0.50,
        y=freq_cut + (min(1.0, ymax) - freq_cut) * 0.50,
        text="🟥 Q1: Critical\n(Severe + Frequent)",
        showarrow=False,
        align="center"
    )
    fig_quad.add_annotation(
        x=delay_cut * 0.50,
        y=freq_cut + (min(1.0, ymax) - freq_cut) * 0.50,
        text="🟧 Q2: Process Risk\n(Frequent, Low Severity)",
        showarrow=False,
        align="center"
    )
    fig_quad.add_annotation(
        x=delay_cut * 0.50,
        y=freq_cut * 0.50,
        text="🟩 Q3: Stable",
        showarrow=False,
        align="center"
    )
    fig_quad.add_annotation(
        x=delay_cut + (xmax - delay_cut) * 0.50,
        y=freq_cut * 0.50,
        text="🟨 Q4: Exception Risk\n(Severe, Rare)",
        showarrow=False,
        align="center"
    )

    fig_quad.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_quad, use_container_width=True)

    st.caption(
        f"Quadrant cutoffs: Payment Delay ≥ {delay_cut} days | Delay Frequency ≥ {freq_cut:.2f}. "
        "Top-right quadrant vendors are highest priority."
    )

    def scm_action_exec(row):
        if row["avg_payment_delay"] > 10 and row["payment_risk_ratio"] >= 0.60:
            return "War-room: clear backlog + enforce payment SLA"
        if row["avg_payment_delay"] > 10:
            return "Clear backlog; weekly CFO/SCM review"
        if row["payment_risk_ratio"] >= 0.60:
            return "Fix approval cycle; enforce TAT + controls"
        if row.get("penalty_cases", 0) > 0:
            return "Contract compliance review; prevent recurrence"
        return "Monitor"

    df["SCM_Executive_Action"] = df.apply(scm_action_exec, axis=1)

    st.subheader("Top Payment-Risk Vendors (Prioritized List)")

    df["_priority_score"] = (
        df["avg_payment_delay"].clip(lower=0) * 0.45
        + df["payment_risk_ratio"].clip(lower=0) * 10 * 0.35
        + (df.get("penalty_cases", 0) > 0).astype(int) * 0.10
        + (df["_working_capital_cost_lakhs"].fillna(0)) * 0.10
    )

    top = df.sort_values("_priority_score", ascending=False).head(20).copy()

    top_display = top[[
        "vendor_name",
        "VPI_Category",
        "avg_payment_delay",
        "payment_risk_ratio",
        "penalty_cases" if "penalty_cases" in top.columns else "vendor_name",
        "_working_capital_cost_lakhs",
        "SCM_Executive_Action"
    ]].copy()

    if "penalty_cases" not in top.columns:
        top_display = top_display.rename(columns={"vendor_name": "Penalty Cases"})
        top_display["Penalty Cases"] = 0

    top_display.rename(columns={
        "vendor_name": "Vendor",
        "avg_payment_delay": "Avg Payment Delay (Days)",
        "payment_risk_ratio": "Delay Frequency (Ratio)",
        "_working_capital_cost_lakhs": "Working Capital Exposure (₹ Lakhs)",
        "SCM_Executive_Action": "Recommended Action"
    }, inplace=True)

    top_display["Delay Frequency (%)"] = (top_display["Delay Frequency (Ratio)"] * 100).round(1)
    top_display.drop(columns=["Delay Frequency (Ratio)"], inplace=True)

    st.dataframe(top_display, use_container_width=True)

    if "location" in df.columns:
        st.subheader("Exposure by Region (₹ Lakhs, proxy)")

        region_exposure = (
            df.groupby("location")["_working_capital_cost_lakhs"]
            .sum()
            .reset_index()
            .sort_values("_working_capital_cost_lakhs", ascending=False)
        )
        region_exposure.rename(columns={"_working_capital_cost_lakhs": "Exposure (₹ Lakhs)"}, inplace=True)

        fig_region = px.bar(
            region_exposure,
            x="location",
            y="Exposure (₹ Lakhs)",
            title="Working Capital Exposure by Region"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    if "vendor_type" in df.columns:
        st.subheader("Exposure by Vendor Type (₹ Lakhs, proxy)")

        type_exposure = (
            df.groupby("vendor_type")["_working_capital_cost_lakhs"]
            .sum()
            .reset_index()
            .sort_values("_working_capital_cost_lakhs", ascending=False)
        )
        type_exposure.rename(columns={"_working_capital_cost_lakhs": "Exposure (₹ Lakhs)"}, inplace=True)

        fig_type = px.bar(
            type_exposure,
            x="vendor_type",
            y="Exposure (₹ Lakhs)",
            title="Working Capital Exposure by Vendor Type"
        )
        st.plotly_chart(fig_type, use_container_width=True)

    st.success(
        "✅ Executive summary: Focus first on vendors in the top-right quadrant (high severity + high frequency), "
        "then reduce chronic process delays (high frequency)."
    )

# ==========================================================
# 🚨 PREDICTIVE RISK DASHBOARD – Executive Early Warning
# ==========================================================
elif dashboard_view == "🚨 Predictive Risk Dashboard":
    st.title("Predictive Risk Dashboard")

    if metrics is not None:
        st.subheader("Model Validation (Test Set Performance)")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", round(metrics.get("accuracy", 0), 2))
        m2.metric("Precision", round(metrics.get("precision", 0), 2))
        m3.metric("Recall", round(metrics.get("recall", 0), 2))
        m4.metric("F1 Score", round(metrics.get("f1_score", 0), 2))
        m5.metric("ROC-AUC", round(metrics.get("roc_auc", 0), 2))

        st.caption(
            "Metrics are calculated on a hold-out test set during model training. "
            "Dashboard performs inference using the trained model."
        )
        st.divider()

    THRESHOLD = st.slider(
        "Risk Threshold (Early Warning Sensitivity)",
        min_value=0.30,
        max_value=0.95,
        value=float(st.session_state["THRESHOLD"]),
        step=0.05,
        key="THRESHOLD"
    )

    st.caption(
        "Lower threshold = monitor more vendors (proactive). "
        "Higher threshold = focus only on critical vendors."
    )

    risk_count = int((df["low_vpi_risk_prob"] >= THRESHOLD).sum())
    critical_cnt = int((df["low_vpi_risk_prob"] >= 0.80).sum())
    high_cnt = int(((df["low_vpi_risk_prob"] >= 0.60) & (df["low_vpi_risk_prob"] < 0.80)).sum())
    moderate_cnt = int(((df["low_vpi_risk_prob"] >= 0.40) & (df["low_vpi_risk_prob"] < 0.60)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendors Above Threshold", risk_count)
    c2.metric("Critical (≥ 0.80)", critical_cnt)
    c3.metric("High (0.60–0.79)", high_cnt)
    c4.metric("Moderate (0.40–0.59)", moderate_cnt)

    if risk_count > 0:
        st.error(f"{risk_count} vendors exceed risk threshold. Preventive action recommended.")
    else:
        st.success("No vendors breach the current threshold. Portfolio is stable.")

    st.divider()

    def risk_bucket_exec(prob: float) -> str:
        if prob >= 0.80:
            return "🔴 Critical"
        elif prob >= 0.60:
            return "🟠 High"
        elif prob >= 0.40:
            return "🟡 Moderate"
        else:
            return "🟢 Low"

    df["Risk_Level_Exec"] = df["low_vpi_risk_prob"].apply(risk_bucket_exec)

    st.subheader("Portfolio Risk Distribution")

    dist = df["Risk_Level_Exec"].value_counts().reset_index()
    dist.columns = ["Risk Level", "Vendors"]

    fig_dist = px.bar(dist, x="Risk Level", y="Vendors", title="Vendor Risk Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    st.subheader("Priority Watchlist (Top 25 Vendors)")

    watchlist = (
        df[df["low_vpi_risk_prob"] >= THRESHOLD]
        .sort_values("low_vpi_risk_prob", ascending=False)
        .head(25)
        .copy()
    )

    if watchlist.empty:
        st.info("No vendors above threshold. Lower the threshold to widen monitoring.")
    else:
        def driver_reason(row):
            drivers = []
            if row.get("avg_delay_days", 0) > 5:
                drivers.append("Delivery delays")
            if row.get("avg_payment_delay", 0) > 7:
                drivers.append("Payment backlog")
            if row.get("rejection_rate_pct", 0) > 7:
                drivers.append("Quality rejections")
            if row.get("penalty_cases", 0) > 0:
                drivers.append("Contract penalties")
            return ", ".join(drivers) if drivers else "Emerging instability"

        def exec_action(row):
            if row["low_vpi_risk_prob"] >= 0.80:
                return "Immediate leadership review (Ops + SCM)"
            elif row["low_vpi_risk_prob"] >= 0.60:
                return "Preventive intervention within 30 days"
            else:
                return "Enhanced monitoring"

        watchlist["Primary Drivers"] = watchlist.apply(driver_reason, axis=1)
        watchlist["Recommended Action"] = watchlist.apply(exec_action, axis=1)

        view = watchlist[[
            "vendor_name",
            "VPI_Category",
            "Risk_Level_Exec",
            "low_vpi_risk_prob",
            "Primary Drivers",
            "Recommended Action"
        ]].copy()

        view = view.rename(columns={
            "vendor_name": "Vendor",
            "low_vpi_risk_prob": "Risk Probability"
        })
        view["Risk Probability"] = view["Risk Probability"].round(2)

        st.dataframe(view, use_container_width=True)

    st.divider()

    if hasattr(model, "feature_importances_"):
        st.subheader("Top Risk Drivers (Model Insight)")

        fi = (
            pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(8)
        )

        fig_fi = px.bar(
            fi,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Drivers Influencing Vendor Risk"
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.caption(
            "Feature importance provides directional insight into which indicators "
            "most influence risk classification."
        )

# ==========================================================
# 💸 Financial Impact (Executive – aligned with Ops & SCM)
# ==========================================================
elif dashboard_view == "💸 Financial Impact":
    st.title("Financial Impact – Executive Exposure & Savings Simulation")

    st.info(
        f"📅 Data window: {data_min_month:%b %Y} → {data_max_month:%b %Y} | "
        f"Last available month: {data_max_month:%b %Y}"
    )
    if pd.Timestamp.today().normalize() - pd.to_datetime(data_max_month).normalize() > pd.Timedelta(days=180):
        st.warning(
            "⚠️ Data is older than ~6 months. Treat impact as a baseline estimate. "
            "Refresh data for real-time prioritization."
        )

    main_col, ctrl_col = st.columns([3, 1], gap="large")

    with ctrl_col:
        st.markdown("### Controls")

        THRESHOLD = st.slider(
            "Risk Threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(st.session_state.get("THRESHOLD", 0.70)),
            step=0.05,
            key="THRESHOLD"
        )
        st.caption("Lower = more vendors included")

        with st.expander("Assumptions", expanded=True):
            annual_cost_of_capital = st.slider(
                "Cost of Capital (%)",
                5.0, 30.0,
                float(st.session_state.get("annual_cost_of_capital", 0.12) * 100),
                0.5,
                key="annual_cost_of_capital_pct"
            ) / 100.0

            delay_cost_rate_per_day = st.slider(
                "Delay Cost / Day (% CV)",
                0.00, 0.30,
                float(st.session_state.get("delay_cost_rate_per_day", 0.0005) * 100),
                0.01,
                key="delay_cost_rate_per_day_pct"
            ) / 100.0

            penalty_rate = st.slider(
                "Penalty Rate (% CV)",
                0.0, 5.0,
                float(st.session_state.get("penalty_rate", 0.005) * 100),
                0.1,
                key="penalty_rate_pct"
            ) / 100.0

            mitigation_effectiveness = st.slider(
                "Mitigation (%)",
                0, 80,
                int(st.session_state.get("mitigation_effectiveness", 0.30) * 100),
                5,
                key="mitigation_effectiveness_pct"
            ) / 100.0

        st.session_state["annual_cost_of_capital"] = annual_cost_of_capital
        st.session_state["delay_cost_rate_per_day"] = delay_cost_rate_per_day
        st.session_state["penalty_rate"] = penalty_rate
        st.session_state["mitigation_effectiveness"] = mitigation_effectiveness

    with main_col:
        st.caption(
            "This is a decision-support simulation (proxy-based). "
            "It helps leadership prioritize interventions and estimate potential savings."
        )

        risk_df = df[df["low_vpi_risk_prob"] >= THRESHOLD].copy()

        if risk_df.empty:
            st.success("No vendors breach the selected threshold. Lower the threshold to simulate broader impact.")
            st.stop()

        required_cols = ["contract_value_lakhs", "avg_delay_days", "avg_payment_delay"]
        missing = [c for c in required_cols if c not in risk_df.columns]
        if missing:
            st.error(f"Missing required columns for impact simulation: {missing}")
            st.stop()

        risk_df["working_capital_cost_lakhs"] = (
            risk_df["contract_value_lakhs"]
            * (annual_cost_of_capital / 365.0)
            * risk_df["avg_payment_delay"].clip(lower=0)
        )

        risk_df["delivery_delay_cost_lakhs"] = (
            risk_df["contract_value_lakhs"]
            * delay_cost_rate_per_day
            * risk_df["avg_delay_days"].clip(lower=0)
        )

        if "penalty_cases" in risk_df.columns:
            risk_df["penalty_exposure_lakhs"] = (
                risk_df["contract_value_lakhs"]
                * penalty_rate
                * (risk_df["penalty_cases"] > 0).astype(int)
            )
        else:
            risk_df["penalty_exposure_lakhs"] = 0.0

        risk_df["total_exposure_lakhs"] = (
            risk_df["working_capital_cost_lakhs"]
            + risk_df["delivery_delay_cost_lakhs"]
            + risk_df["penalty_exposure_lakhs"]
        )

        risk_df["estimated_savings_lakhs"] = risk_df["total_exposure_lakhs"] * mitigation_effectiveness

        total_exposure = float(risk_df["total_exposure_lakhs"].sum())
        total_savings = float(risk_df["estimated_savings_lakhs"].sum())
        avg_exposure_per_vendor = total_exposure / max(risk_df.shape[0], 1)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Vendors in Risk Set", int(risk_df.shape[0]))
        k2.metric("Total Exposure (₹ Lakhs)", round(total_exposure, 2))
        k3.metric("Estimated Savings (₹ Lakhs)", round(total_savings, 2))
        k4.metric("Avg Exposure per Vendor (₹ Lakhs)", round(avg_exposure_per_vendor, 2))

        st.divider()

        st.subheader("Exposure Breakdown (₹ Lakhs)")

        breakdown = pd.DataFrame({
            "Component": ["Working Capital Cost", "Delivery Delay Cost", "Penalty Exposure"],
            "Amount_Lakhs": [
                risk_df["working_capital_cost_lakhs"].sum(),
                risk_df["delivery_delay_cost_lakhs"].sum(),
                risk_df["penalty_exposure_lakhs"].sum()
            ]
        })

        fig_split = px.pie(
            breakdown,
            names="Component",
            values="Amount_Lakhs",
            title="Where the exposure is coming from"
        )
        st.plotly_chart(fig_split, use_container_width=True)

        st.divider()

        st.subheader("Top Vendors by Exposure (Prioritized)")

        show_cols = [
            "vendor_name",
            "VPI_Category",
            "risk_level",
            "low_vpi_risk_prob",
            "contract_value_lakhs",
            "avg_delay_days",
            "avg_payment_delay",
            "working_capital_cost_lakhs",
            "delivery_delay_cost_lakhs",
            "penalty_exposure_lakhs",
            "total_exposure_lakhs",
            "estimated_savings_lakhs"
        ]

        top_exposure = risk_df.sort_values("total_exposure_lakhs", ascending=False)[show_cols].head(20).copy()
        top_exposure["low_vpi_risk_prob"] = top_exposure["low_vpi_risk_prob"].round(2)

        top_exposure = top_exposure.rename(columns={
            "vendor_name": "Vendor",
            "low_vpi_risk_prob": "Risk Probability",
            "contract_value_lakhs": "Contract (₹ Lakhs)",
            "avg_delay_days": "Avg Delivery Delay (Days)",
            "avg_payment_delay": "Avg Payment Delay (Days)",
            "working_capital_cost_lakhs": "Working Capital Cost (₹ Lakhs)",
            "delivery_delay_cost_lakhs": "Delivery Delay Cost (₹ Lakhs)",
            "penalty_exposure_lakhs": "Penalty Exposure (₹ Lakhs)",
            "total_exposure_lakhs": "Total Exposure (₹ Lakhs)",
            "estimated_savings_lakhs": "Estimated Savings (₹ Lakhs)"
        })

        st.dataframe(top_exposure, use_container_width=True)

        st.divider()

        if "location" in risk_df.columns:
            st.subheader("Exposure by Region (₹ Lakhs)")

            region_exposure = (
                risk_df.groupby("location")["total_exposure_lakhs"]
                .sum()
                .reset_index()
                .sort_values("total_exposure_lakhs", ascending=False)
            )
            region_exposure = region_exposure.rename(columns={
                "location": "Region",
                "total_exposure_lakhs": "Total Exposure (₹ Lakhs)"
            })

            fig_region = px.bar(
                region_exposure,
                x="Region",
                y="Total Exposure (₹ Lakhs)",
                title="Total Financial Exposure by Region"
            )
            st.plotly_chart(fig_region, use_container_width=True)

        if "vendor_type" in risk_df.columns:
            st.subheader("Exposure by Vendor Type (₹ Lakhs)")

            type_exposure = (
                risk_df.groupby("vendor_type")["total_exposure_lakhs"]
                .sum()
                .reset_index()
                .sort_values("total_exposure_lakhs", ascending=False)
            )
            type_exposure = type_exposure.rename(columns={
                "vendor_type": "Vendor Type",
                "total_exposure_lakhs": "Total Exposure (₹ Lakhs)"
            })

            fig_type = px.bar(
                type_exposure,
                x="Vendor Type",
                y="Total Exposure (₹ Lakhs)",
                title="Total Financial Exposure by Vendor Type"
            )
            st.plotly_chart(fig_type, use_container_width=True)

        st.success(
            "✅ Executive takeaway: Prioritize vendors with highest Total Exposure first. "
            "Most savings come from targeting high exposure vendors with high mitigation effectiveness."
        )

# ==========================================================
# 📈 Delivery Time-Series Intelligence
# ==========================================================
elif dashboard_view == "📈 Delivery Time-Series Intelligence":
    st.title("Delivery Time-Series Intelligence")

    main_col, ctrl_col = st.columns([3, 1], gap="large")

    with ctrl_col:
        st.markdown("### Controls")

        forecast_horizon = st.selectbox(
            "Forecast horizon",
            options=[3, 6, 9, 12],
            index=1,
            key="ts_forecast_horizon"
        )

        sensitivity = st.slider(
            "Anomaly sensitivity (Z-score)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key="ts_anomaly_sensitivity"
        )

        show_conf_int = st.checkbox(
            "Show forecast confidence band",
            value=True,
            key="ts_show_conf_int"
        )

    with main_col:
        mt = monthly_trend.sort_values("delivery_month").copy()
        mt["delivery_month"] = pd.to_datetime(mt["delivery_month"])

        ts_series = mt.set_index("delivery_month")["delivery_delay_days"].astype(float)

        if ts_series.dropna().shape[0] < 6:
            st.warning("Not enough monthly data points for reliable forecasting. Add more months to enable this view.")
            st.stop()

        anomaly_df = mt.copy()
        anomaly_df["rolling_mean"] = anomaly_df["delivery_delay_days"].rolling(3, min_periods=3).mean()
        anomaly_df["rolling_std"] = anomaly_df["delivery_delay_days"].rolling(3, min_periods=3).std()

        anomaly_df["rolling_z"] = (
            (anomaly_df["delivery_delay_days"] - anomaly_df["rolling_mean"])
            / anomaly_df["rolling_std"]
        )

        anomaly_df["anomaly_flag"] = anomaly_df["rolling_z"].abs() > sensitivity
        anomalies = anomaly_df[anomaly_df["anomaly_flag"] == True].copy()

        forecast_df = None
        conf_df = None

        try:
            arima_model = ARIMA(ts_series, order=(1, 1, 1))
            model_fit = arima_model.fit()

            if show_conf_int:
                pred = model_fit.get_forecast(steps=int(forecast_horizon))
                fc_mean = pred.predicted_mean.clip(lower=0)
                fc_ci = pred.conf_int()

                lower_col = [c for c in fc_ci.columns if "lower" in c.lower()][0]
                upper_col = [c for c in fc_ci.columns if "upper" in c.lower()][0]

                conf_df = pd.DataFrame({
                    "delivery_month": fc_ci.index,
                    "lower": fc_ci[lower_col].clip(lower=0).values,
                    "upper": fc_ci[upper_col].clip(lower=0).values
                })

                forecast_df = pd.DataFrame({
                    "delivery_month": fc_mean.index,
                    "forecast_delay": fc_mean.values
                })
            else:
                fc = model_fit.forecast(steps=int(forecast_horizon)).clip(lower=0)
                forecast_df = pd.DataFrame({
                    "delivery_month": fc.index,
                    "forecast_delay": fc.values
                })

        except Exception as e:
            st.error(f"Forecast failed: {e}")
            forecast_df = None
            conf_df = None

        st.subheader("Master View – Historical + Forecast + Anomalies")

        hist_line = mt[["delivery_month", "delivery_delay_days"]].copy()
        hist_line["type"] = "Historical"
        hist_line = hist_line.rename(columns={"delivery_delay_days": "value"})

        combined = hist_line.copy()

        if forecast_df is not None and not forecast_df.empty:
            fc_line = forecast_df.rename(columns={"forecast_delay": "value"}).copy()
            fc_line["type"] = "Forecast"
            combined = pd.concat([combined, fc_line[["delivery_month", "value", "type"]]], ignore_index=True)

        fig_master = px.line(
            combined,
            x="delivery_month",
            y="value",
            color="type",
            markers=True,
            title="Delivery Delay (Days): Historical vs Forecast"
        )

        if not anomalies.empty:
            fig_master.add_scatter(
                x=anomalies["delivery_month"],
                y=anomalies["delivery_delay_days"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10)
            )

        if conf_df is not None and not conf_df.empty:
            fig_master.add_scatter(
                x=conf_df["delivery_month"],
                y=conf_df["lower"],
                mode="lines",
                name="Forecast Lower",
                line=dict(width=1, dash="dot")
            )
            fig_master.add_scatter(
                x=conf_df["delivery_month"],
                y=conf_df["upper"],
                mode="lines",
                name="Forecast Upper",
                line=dict(width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.08)"
            )

        fig_master.update_xaxes(rangemode="normal")
        fig_master.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_master, use_container_width=True)

        st.caption(
            "Why forecast can look flatter than history: ARIMA models the predictable component (trend/level). "
            "If zig-zag is mostly noise/spikes, the best estimate becomes near the recent average. "
            "Confidence band shows uncertainty around that average."
        )

        st.subheader("Operational Signal – Monthly Risk Index & % Delayed")

        signal_df = mt.copy()

        plot2 = pd.DataFrame({
            "delivery_month": signal_df["delivery_month"],
            "Monthly Risk Index": signal_df.get("monthly_risk_index", pd.NA),
            "% Delayed (frequency)": (signal_df.get("delay_flag", pd.NA) * 100.0)
        })

        plot2_melt = plot2.melt(
            id_vars=["delivery_month"],
            value_vars=["Monthly Risk Index", "% Delayed (frequency)"],
            var_name="Metric",
            value_name="Value"
        ).dropna()

        fig_signal = px.line(
            plot2_melt,
            x="delivery_month",
            y="Value",
            color="Metric",
            markers=True,
            title="Risk Index & Delay Frequency (adds information beyond avg delay)"
        )
        fig_signal.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_signal, use_container_width=True)

        st.subheader("🚨 Escalation Queue (Anomaly-triggered)")

        if not anomalies.empty:
            escalation_df = anomalies.copy()
            escalation_df["abs_z"] = escalation_df["rolling_z"].abs()

            p90 = escalation_df["abs_z"].quantile(0.90)
            p70 = escalation_df["abs_z"].quantile(0.70)

            def anomaly_severity_percentile(abs_z):
                if abs_z >= p90:
                    return "Critical"
                elif abs_z >= p70:
                    return "High"
                else:
                    return "Moderate"

            escalation_df["severity"] = escalation_df["abs_z"].apply(anomaly_severity_percentile)

            def escalation_action(sev, z):
                if sev == "Critical":
                    return "Immediate escalation to Head Ops + SCM (War Room)"
                elif sev == "High":
                    return "Trigger Ops-SCM joint review within 48 hrs"
                else:
                    if z > 0:
                        return "Ops check: site readiness + resourcing validation"
                    else:
                        return "Positive deviation: validate data & capture best practice"

            escalation_df["escalation_action"] = escalation_df.apply(
                lambda r: escalation_action(r["severity"], r["rolling_z"]),
                axis=1
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Months", int(escalation_df.shape[0]))
            c2.metric("Critical Alerts", int((escalation_df["severity"] == "Critical").sum()))
            c3.metric("High Alerts", int((escalation_df["severity"] == "High").sum()))

            st.error("Escalation Triggered: Anomalous delivery delay spikes detected.")

            esc_view = escalation_df[[
                "delivery_month",
                "delivery_delay_days",
                "rolling_z",
                "severity",
                "escalation_action"
            ]].copy()

            esc_view["delivery_month"] = pd.to_datetime(esc_view["delivery_month"]).dt.strftime("%b %Y")
            esc_view["rolling_z"] = esc_view["rolling_z"].round(2)
            esc_view["delivery_delay_days"] = esc_view["delivery_delay_days"].round(2)

            st.dataframe(
                esc_view.sort_values(["severity", "rolling_z"], ascending=[True, False]),
                use_container_width=True
            )
        else:
            st.success("No anomaly escalation triggered. Operations remain stable.")