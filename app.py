import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import os
from datetime import datetime

ts = os.path.getmtime("processed/delivery_anomalies.csv")
st.caption(f"Anomaly data last updated: {datetime.fromtimestamp(ts)}")

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Vendor Governance & Risk Intelligence",
    layout="wide"
)

# Cache clear button
st.sidebar.button(
    "🔄 Clear Cache",
    on_click=lambda: st.cache_data.clear()
)

# ==========================================================
# Session state defaults
# ==========================================================
if "THRESHOLD" not in st.session_state:
    st.session_state["THRESHOLD"] = 0.70

if "annual_cost_of_capital" not in st.session_state:
    st.session_state["annual_cost_of_capital"] = 0.12  # 12%

if "delay_cost_rate_per_day" not in st.session_state:
    st.session_state["delay_cost_rate_per_day"] = 0.0005  # 0.05% per day

if "penalty_rate" not in st.session_state:
    st.session_state["penalty_rate"] = 0.005  # 0.5%

if "mitigation_effectiveness" not in st.session_state:
    st.session_state["mitigation_effectiveness"] = 0.30  # 30%

THRESHOLD = float(st.session_state["THRESHOLD"])

# ==========================================================
# Loaders
# ==========================================================
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("processed/vendor_dashboard_master.csv")

        if "last_delivery_month" in data.columns:
            data["last_delivery_month"] = pd.to_datetime(
                data["last_delivery_month"], errors="coerce", dayfirst=True
            )

        obj_cols = data.select_dtypes(include=["object"]).columns
        data[obj_cols] = data[obj_cols].fillna("").astype("object")

        return data
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_monthly_trend():
    try:
        mt = pd.read_csv("processed/monthly_trend.csv")
        mt["delivery_month"] = pd.to_datetime(mt["delivery_month"])
        return mt
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_forecast_data():
    try:
        fc = pd.read_csv("processed/delivery_forecast.csv")
        fc["delivery_month"] = pd.to_datetime(fc["delivery_month"])
        return fc
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_anomalies_data():
    try:
        an = pd.read_csv("processed/delivery_anomalies.csv")
        an["delivery_month"] = pd.to_datetime(an["delivery_month"])
        return an
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_model_metrics():
    try:
        return pd.read_csv("processed/model_metrics.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_model_drivers():
    try:
        return pd.read_csv("processed/model_drivers.csv")
    except FileNotFoundError:
        return pd.DataFrame()


# ==========================================================
# Load exported notebook artifacts
# ==========================================================
df = load_data()
monthly_trend = load_monthly_trend()
forecast_df = load_forecast_data()
anomalies_df = load_anomalies_data()
model_metrics = load_model_metrics()
model_drivers = load_model_drivers()

# ==========================================================
# Required-file safety checks
# ==========================================================
if df.empty:
    st.error(
        "vendor_dashboard_master.csv not found or is empty. "
        "Please run the notebook export section first."
    )
    st.stop()

if monthly_trend.empty:
    st.error(
        "monthly_trend.csv not found or is empty. "
        "Please run the notebook export section first."
    )
    st.stop()

# ==========================================================
# Global date window
# ==========================================================
data_max_month = pd.to_datetime(monthly_trend["delivery_month"]).max()
data_min_month = pd.to_datetime(monthly_trend["delivery_month"]).min()

# ==========================================================
# Sidebar filters
# ==========================================================
st.sidebar.header("Filters")

active_window = st.sidebar.selectbox(
    "Active vendor window (based on last delivery month)",
    ["All", "Last 3 months", "Last 6 months", "Last 12 months", "Last 24 months"],
    index=0
)

if active_window != "All" and pd.notna(data_max_month):
    months = int(active_window.split()[1])
    cutoff = pd.to_datetime(data_max_month) - pd.DateOffset(months=months)
    df = df[df["last_delivery_month"] >= cutoff].copy()

risk_filter = st.sidebar.multiselect(
    "VPI Category",
    options=sorted([x for x in df["VPI_Category"].unique() if x != ""]),
    default=sorted([x for x in df["VPI_Category"].unique() if x != ""])
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
        "📈 Delivery Time-Series Intelligence and Anomaly Detection",
    ]
)

if df.empty:
    st.warning("No data available for the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ==========================================================
# 🏁 Executive Summary
# ==========================================================
if dashboard_view == "🏁 Executive Summary":
    st.title("Executive Summary – Vendor Governance & Risk Intelligence")

    st.info(
        f"📅 Data window: {data_min_month:%b %Y} → {data_max_month:%b %Y} | "
        f"Last available month: {data_max_month:%b %Y}"
    )

    if pd.Timestamp.today().normalize() - pd.to_datetime(data_max_month).normalize() > pd.Timedelta(days=180):
        st.warning(
            "⚠️ Data is older than ~6 months. Use this as governance baseline. "
            "For live tracking, refresh with latest 6–12 months data."
        )

    total_vendors = int(df.shape[0])
    low_vpi = int((df["VPI_Category"] == "Low").sum())

    threshold_val = float(st.session_state["THRESHOLD"])
    above_threshold = int((df["low_vpi_risk_prob"] >= threshold_val).sum())
    critical = int((df["low_vpi_risk_prob"] >= 0.80).sum())
    high = int(
        ((df["low_vpi_risk_prob"] >= 0.60) & (df["low_vpi_risk_prob"] < 0.80)).sum()
    )

    avg_ops_delay = float(df["avg_delay_days"].mean())
    avg_rejection = float(df["rejection_rate_pct"].mean())
    avg_pay_delay = float(df["avg_payment_delay"].mean())
    pct_invoices_delayed = float(df["payment_risk_ratio"].mean()) * 100.0

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

    st.subheader("Executive Watchlist – Top 10 Vendors to Intervene")

    def executive_reason(row):
        reasons = []

        if row["low_vpi_risk_prob"] >= 0.80:
            reasons.append("Critical predicted deterioration")
        if row["avg_delay_days"] > 5:
            reasons.append("Severe delivery delays")
        if row["rejection_rate_pct"] > 7:
            reasons.append("Quality rejection risk")
        if row["avg_payment_delay"] > 10:
            reasons.append("Payment backlog severity")
        if row["penalty_cases"] > 0:
            reasons.append("Penalty exposure")

        return ", ".join(reasons) if reasons else "Early risk signals"

    def executive_action(row):
        if row["low_vpi_risk_prob"] >= 0.80:
            return "War-room: Ops+SCM leadership review (weekly)"
        if row["low_vpi_risk_prob"] >= 0.60:
            return "Preventive intervention within 30 days"
        return "Enhanced monitoring"

    watch = df.sort_values("low_vpi_risk_prob", ascending=False).head(10).copy()
    watch["Why"] = watch.apply(executive_reason, axis=1)
    watch["Action"] = watch.apply(executive_action, axis=1)

    show = watch[[
        "vendor_name",
        "VPI_Score",
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
        "VPI_Score": "VPI_Score",
        "low_vpi_risk_prob": "Risk Probability",
        "_wc_exposure_lakhs": "WC Exposure (₹ Lakhs)"
    })

    show["Risk Probability"] = show["Risk Probability"].round(2)
    show["WC Exposure (₹ Lakhs)"] = show["WC Exposure (₹ Lakhs)"].round(2)

    st.dataframe(show, use_container_width=True)

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
            label="⬇️ Download Executive Watchlist (Top 10) – CSV",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name=f"Executive_Watchlist_Top10_{data_max_month:%Y_%m}.csv",
            mime="text/csv"
        )

    with b2:
        st.download_button(
            label="⬇️ Download Executive KPI Snapshot – CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Executive_KPI_Snapshot_{data_max_month:%Y_%m}.csv",
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
# 🔧 Operations Dashboard
# ==========================================================
elif dashboard_view == "🔧 Operations Dashboard":
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
    avg_delay_val = float(df["avg_delay_days"].mean())
    avg_rejection = float(df["rejection_rate_pct"].mean())
    chronic_delay_vendors = int(
       ((df["delay_frequency"] >= 0.70) & (df["avg_delay_days"] > 3)).sum()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low VPI Vendors (Performance Concern)", high_risk_low_vpi)
    c2.metric("Avg Delivery Delay (Days)", round(avg_delay_val, 2))
    c3.metric("Avg Rejection Rate (%)", round(avg_rejection, 2))
    c4.metric("Chronic Delay Vendors (≥70% frequency)", chronic_delay_vendors)

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
        if x > 2:
            return "🟡 Moderate"
        return "🟢 Stable"

    def rag_quality(x: float) -> str:
        if x > 7:
            return "🔴 High"
        if x > 3:
            return "🟡 Moderate"
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
            "VPI_Score": True,
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

    ops_cols = [
        "vendor_name",
        "VPI_Score",
        "VPI_Category",
        "avg_delay_days",
        "delay_frequency",
        "rejection_rate_pct",
        "contract_value_lakhs",
        "last_delivery_month",
        "Ops_Executive_Action"
    ]

    ops_cols = [c for c in ops_cols if c in top_ops.columns]
    top_ops_display = top_ops[ops_cols].copy()

    top_ops_display.rename(columns={
        "vendor_name": "Vendor",
        "avg_delay_days": "Avg Delay (Days)",
        "delay_frequency": "Delay Frequency (0–1)",
        "rejection_rate_pct": "Rejection Rate (%)",
        "contract_value_lakhs": "Contract Value (₹ Lakhs)",
        "last_delivery_month": "Last Delivery Month",
        "Ops_Executive_Action": "Recommended Action"
    }, inplace=True)

    if "Last Delivery Month" in top_ops_display.columns:
        top_ops_display["Last Delivery Month"] = pd.to_datetime(
            top_ops_display["Last Delivery Month"], errors="coerce"
        ).dt.strftime("%b %Y")
    top_ops_display["Delay Frequency (%)"] = (
        top_ops_display["Delay Frequency (0–1)"] * 100
    ).round(1)
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
# 💰 SCM Dashboard
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

    avg_payment_delay_val = float(df["avg_payment_delay"].mean())
    pct_invoices_delayed = float(df["payment_risk_ratio"].mean()) * 100
    vendors_with_penalty = int((df["penalty_cases"] > 0).sum())
    total_penalty_cases = int(df["penalty_cases"].sum())

    annual_cost_of_capital = float(st.session_state.get("annual_cost_of_capital", 0.12))
    df["_working_capital_cost_lakhs"] = (
        df["contract_value_lakhs"]
        * (annual_cost_of_capital / 365.0)
        * df["avg_payment_delay"].clip(lower=0)
    )
    total_wc_exposure = float(df["_working_capital_cost_lakhs"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Payment Delay (Days)", round(avg_payment_delay_val, 2))
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
        if x > 7:
            return "🟡 Moderate"
        return "🟢 Stable"

    def rag_frequency(x: float) -> str:
        if x >= 0.60:
            return "🔴 High"
        if x >= 0.30:
            return "🟡 Moderate"
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
            "penalty_cases": True,
            "contract_value_lakhs": True,
            "_working_capital_cost_lakhs": ":.2f",
            "VPI_Score": True,
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
        if row["penalty_cases"] > 0:
            return "Contract compliance review; prevent recurrence"
        return "Monitor"

    df["SCM_Executive_Action"] = df.apply(scm_action_exec, axis=1)

    st.subheader("Top Payment-Risk Vendors (Prioritized List)")

    df["_priority_score"] = (
        df["avg_payment_delay"].clip(lower=0) * 0.45
        + df["payment_risk_ratio"].clip(lower=0) * 10 * 0.35
        + (df["penalty_cases"] > 0).astype(int) * 0.10
        + df["_working_capital_cost_lakhs"].fillna(0) * 0.10
    )

    top = df.sort_values("_priority_score", ascending=False).head(20).copy()

    top_display = top[[
        "vendor_name",
        "VPI_Score",
        "VPI_Category",
        "avg_payment_delay",
        "payment_risk_ratio",
        "penalty_cases",
        "_working_capital_cost_lakhs",
        "SCM_Executive_Action"
    ]].copy()

    top_display.rename(columns={
        "vendor_name": "Vendor",
        "avg_payment_delay": "Avg Payment Delay (Days)",
        "payment_risk_ratio": "Delay Frequency (Ratio)",
        "penalty_cases": "Penalty Cases",
        "_working_capital_cost_lakhs": "Working Capital Exposure (₹ Lakhs)",
        "SCM_Executive_Action": "Recommended Action"
    }, inplace=True)

    top_display["Delay Frequency (%)"] = (
        top_display["Delay Frequency (Ratio)"] * 100
    ).round(1)
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
# 🚨 Predictive Risk Dashboard
# ==========================================================
elif dashboard_view == "🚨 Predictive Risk Dashboard":
    st.title("Predictive Risk Dashboard")

    if not model_metrics.empty:
        st.subheader("Model Validation (Test Set Performance)")

        metrics_row = model_metrics.iloc[0]
  
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", round(float(metrics_row.get("accuracy", 0)), 2))
        m2.metric("Precision", round(float(metrics_row.get("precision", 0)), 2))
        m3.metric("Recall", round(float(metrics_row.get("recall", 0)), 2))
        m4.metric("F1 Score", round(float(metrics_row.get("f1_score", 0)), 2))
        m5.metric("ROC-AUC", round(float(metrics_row.get("roc_auc", 0)), 2))

        st.caption(
            f"Metrics are calculated on a hold-out test set during model training. "
            f"Selected model: {metrics_row.get('model_name', 'Unknown')}."
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
    high_cnt = int(
        ((df["low_vpi_risk_prob"] >= 0.60) & (df["low_vpi_risk_prob"] < 0.80)).sum()
    )
    moderate_cnt = int(
        ((df["low_vpi_risk_prob"] >= 0.40) & (df["low_vpi_risk_prob"] < 0.60)).sum()
    )

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
        if prob >= 0.60:
            return "🟠 High"
        if prob >= 0.40:
            return "🟡 Moderate"
        return "🟢 Low"

    view_df = df.copy()
    view_df["Risk_Level_Exec"] = view_df["low_vpi_risk_prob"].apply(risk_bucket_exec)

    st.subheader("Portfolio Risk Distribution")
    dist = view_df["Risk_Level_Exec"].value_counts().reset_index()
    dist.columns = ["Risk Level", "Vendors"]

    fig_dist = px.bar(
        dist,
        x="Risk Level",
        y="Vendors",
        title="Vendor Risk Distribution"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.subheader("Priority Watchlist (Top 25 Vendors)")

    watchlist = (
        view_df[view_df["low_vpi_risk_prob"] >= THRESHOLD]
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
            if row["low_vpi_risk_prob"] >= 0.60:
                return "Preventive intervention within 30 days"
            return "Enhanced monitoring"

        watchlist["Primary Drivers"] = watchlist.apply(driver_reason, axis=1)
        watchlist["Recommended Action"] = watchlist.apply(exec_action, axis=1)

        view = watchlist[[
            "vendor_name",
            "VPI_Score",
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
    st.subheader("Top Risk Drivers (Model Insight)")
    if not model_drivers.empty:
        plot_df = model_drivers.sort_values("Importance", ascending=True).copy()

        fig_drivers = px.bar(
            plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Direction" if "Direction" in plot_df.columns else None,
            title="Top Drivers Influencing Vendor Risk"
        )
        st.plotly_chart(fig_drivers, use_container_width=True)

        show_cols = [c for c in ["Feature", "Coefficient", "Importance", "Direction", "Driver_Type"] if c in model_drivers.columns]
        st.dataframe(model_drivers[show_cols], use_container_width=True)
    else:
        st.info("Model driver file is not available.")

# ==========================================================
# 💸 Financial Impact
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

        financial_threshold = st.slider(
            "Risk Threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(st.session_state.get("THRESHOLD", 0.70)),
            step=0.05,
            key="financial_threshold"
        )
        st.session_state["THRESHOLD"] = financial_threshold
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

        risk_df = df[df["low_vpi_risk_prob"] >= financial_threshold].copy()

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

        risk_df["estimated_savings_lakhs"] = (
            risk_df["total_exposure_lakhs"] * mitigation_effectiveness
        )

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
            "VPI_Score",
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

        top_exposure = (
            risk_df.sort_values("total_exposure_lakhs", ascending=False)[show_cols]
            .head(20)
            .copy()
        )
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
# 📈 Delivery Time-Series Intelligence and Anomaly Detection
# ==========================================================
elif dashboard_view == "📈 Delivery Time-Series Intelligence and Anomaly Detection":
    st.title("Delivery Time-Series Intelligence and Anomaly Detection")

    # Always initialize first
    display_anomalies = pd.DataFrame()

    def anomaly_root_cause(month):
        if "last_delivery_month" not in df.columns:
            return "Portfolio delay spike", "Review vendor delivery performance"

        temp_df = df.copy()
        temp_df["last_delivery_month"] = pd.to_datetime(
            temp_df["last_delivery_month"], errors="coerce"
        )

        mdf = temp_df[
            temp_df["last_delivery_month"].dt.to_period("M") == month.to_period("M")
        ]

        if mdf.empty:
            return "Portfolio delay spike", "Review vendor delivery performance"

        avg_delay = mdf["avg_delay_days"].mean()
        avg_rejection = mdf["rejection_rate_pct"].mean()
        avg_payment = mdf["avg_payment_delay"].mean()

        if avg_delay > 5:
            return "Delivery delays increased", "Ops intervention required"
        if avg_rejection > 7:
            return "Quality rejection spike", "Vendor quality audit"
        if avg_payment > 10:
            return "Payment backlog", "SCM payment cycle review"

        return "General instability", "Monitor vendor performance"

    def classify_severity(dev_pct):
        if pd.isna(dev_pct):
            return "Low"
        if abs(dev_pct) >= 30:
            return "Critical"
        elif abs(dev_pct) >= 20:
            return "High"
        elif abs(dev_pct) >= 10:
            return "Medium"
        else:
            return "Low"

    severity_order = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    severity_size_map = {"Low": 8, "Medium": 11, "High": 15, "Critical": 20}

    main_col, ctrl_col = st.columns([3, 1], gap="large")

    with ctrl_col:
        st.markdown("### Controls")

        show_conf_int = st.checkbox(
            "Show forecast confidence band",
            value=True,
            key="ts_show_conf_int"
        )

        z_threshold = st.slider(
            "Anomaly Z-score threshold",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="ts_z_threshold"
        )

        pct_threshold = st.slider(
            "Anomaly % change threshold",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="ts_pct_threshold"
        )

    # Prepare anomaly dataframe
    if "anomalies_df" in globals() and anomalies_df is not None and not anomalies_df.empty:
        display_anomalies = anomalies_df.copy()

        if "delivery_month" in display_anomalies.columns:
            display_anomalies["delivery_month"] = pd.to_datetime(
                display_anomalies["delivery_month"],
                errors="coerce",
                dayfirst=True
            )

        numeric_cols = [
            "delivery_delay_days",
            "rolling_mean",
            "rolling_z",
            "pct_change"
        ]
        for col in numeric_cols:
            if col in display_anomalies.columns:
                display_anomalies[col] = pd.to_numeric(
                    display_anomalies[col], errors="coerce"
                )

        if "anomaly_flag" in display_anomalies.columns:
            display_anomalies["anomaly_flag"] = (
                display_anomalies["anomaly_flag"]
                .astype(str)
                .str.strip()
                .str.upper()
                .isin(["TRUE", "1", "YES"])
            )

            display_anomalies = display_anomalies[
                display_anomalies["anomaly_flag"]
            ].copy()

        elif {"rolling_z", "pct_change"}.issubset(display_anomalies.columns):
            display_anomalies = display_anomalies[
                (display_anomalies["rolling_z"].abs() >= z_threshold) |
                (display_anomalies["pct_change"].abs() >= pct_threshold)
            ].copy()

        elif "rolling_z" in display_anomalies.columns:
            display_anomalies = display_anomalies[
                display_anomalies["rolling_z"].abs() >= z_threshold
            ].copy()

        # Create deviation % if not already available
        if "deviation_pct" not in display_anomalies.columns:
            if {"delivery_delay_days", "rolling_mean"}.issubset(display_anomalies.columns):
                display_anomalies["deviation_pct"] = np.where(
                    display_anomalies["rolling_mean"].fillna(0) != 0,
                    (
                        (display_anomalies["delivery_delay_days"] - display_anomalies["rolling_mean"])
                        / display_anomalies["rolling_mean"]
                    ) * 100,
                    np.nan
                )
            elif "pct_change" in display_anomalies.columns:
                display_anomalies["deviation_pct"] = display_anomalies["pct_change"] * 100
            else:
                display_anomalies["deviation_pct"] = np.nan

        # Build severity if missing
        if "severity" not in display_anomalies.columns:
            display_anomalies["severity"] = display_anomalies["deviation_pct"].apply(classify_severity)
        else:
            display_anomalies["severity"] = (
                display_anomalies["severity"]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Critial": "Critical"})
            )

        # Marker size for chart
        display_anomalies["marker_size"] = (
            display_anomalies["severity"]
            .map(severity_size_map)
            .fillna(10)
        )

        # Fallback escalation action if missing
        if "escalation_action" not in display_anomalies.columns:
            def derive_escalation_action(row):
                sev = row.get("severity", "Low")
                dev = row.get("deviation_pct", np.nan)

                if pd.isna(dev):
                    return "Review vendor delivery performance"

                if dev > 0:
                    if sev == "Critical":
                        return "Immediate vendor and ops escalation"
                    elif sev == "High":
                        return "Ops review and dispatch bottleneck check"
                    elif sev == "Medium":
                        return "Review delivery cycle slippage"
                    else:
                        return "Continue monitoring"
                else:
                    if sev == "Critical":
                        return "Validate unusual improvement / backlog closure"
                    elif sev == "High":
                        return "Check bulk dispatch closure trend"
                    else:
                        return "Monitor for consistency"

            display_anomalies["escalation_action"] = display_anomalies.apply(
                derive_escalation_action, axis=1
            )

    with main_col:
        st.subheader("Master View – Historical + Forecast + Anomalies")

        mt = monthly_trend.sort_values("delivery_month").copy()
        mt["type"] = "Historical"
        mt = mt.rename(columns={"delivery_delay_days": "value"})

        combined = mt[["delivery_month", "value", "type"]].copy()

        if not forecast_df.empty:
            fc_plot = forecast_df.copy()
            fc_plot["type"] = "Forecast"
            fc_plot = fc_plot.rename(columns={"forecast_delay": "value"})
            combined = pd.concat(
                [combined, fc_plot[["delivery_month", "value", "type"]]],
                ignore_index=True
            )

        fig_master = px.line(
            combined,
            x="delivery_month",
            y="value",
            color="type",
            markers=True,
            title="Delivery Delay (Days): Historical vs Forecast"
        )

        if forecast_df.empty:
            st.info("Forecast file not available. Showing historical trend only.")

        if (
            not display_anomalies.empty
            and "delivery_month" in display_anomalies.columns
            and "delivery_delay_days" in display_anomalies.columns
        ):
            fig_master.add_scatter(
                x=display_anomalies["delivery_month"],
                y=display_anomalies["delivery_delay_days"],
                mode="markers",
                name="Anomaly",
                text=display_anomalies["severity"],
                marker=dict(
                    color="red",
                    size=display_anomalies["marker_size"],
                    symbol="x"
                ),
                hovertemplate=(
                    "<b>Month:</b> %{x|%b %Y}<br>"
                    "<b>Delay:</b> %{y:.2f}<br>"
                    "<b>Severity:</b> %{text}<extra></extra>"
                )
            )

        if show_conf_int and {"lower", "upper"}.issubset(forecast_df.columns):
            fig_master.add_scatter(
                x=forecast_df["delivery_month"],
                y=forecast_df["lower"],
                mode="lines",
                name="Forecast Lower",
                line=dict(width=1, dash="dot")
            )

            fig_master.add_scatter(
                x=forecast_df["delivery_month"],
                y=forecast_df["upper"],
                mode="lines",
                name="Forecast Upper",
                line=dict(width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.08)"
            )

        fig_master.update_xaxes(rangemode="normal")
        fig_master.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_master, use_container_width=True)

        st.subheader("Operational Signal – Monthly Risk Index & % Delayed")

        plot2 = pd.DataFrame({
            "delivery_month": monthly_trend["delivery_month"],
            "Monthly Risk Index": monthly_trend.get("monthly_risk_index", pd.NA),
            "% Delayed (frequency)": monthly_trend.get("delay_flag", pd.NA) * 100.0
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
            title="Risk Index & Delay Frequency"
        )
        fig_signal.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_signal, use_container_width=True)

        st.subheader("🚨 Escalation Queue (Anomaly-triggered)")

        if not display_anomalies.empty:
            latest_month = (
                display_anomalies["delivery_month"].max().strftime("%b %Y")
                if "delivery_month" in display_anomalies.columns
                and display_anomalies["delivery_month"].notna().any()
                else "N/A"
            )

            worst_severity = "Low"
            if "severity" in display_anomalies.columns:
                worst_rank = display_anomalies["severity"].map(severity_order).fillna(0).max()
                for sev, rank in severity_order.items():
                    if rank == worst_rank:
                        worst_severity = sev
                        break

            top_action = (
                display_anomalies["escalation_action"].mode().iloc[0]
                if "escalation_action" in display_anomalies.columns
                and not display_anomalies["escalation_action"].mode().empty
                else "Review anomalies"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anomaly Months", int(display_anomalies.shape[0]))
            c2.metric("Latest Anomaly", latest_month)
            c3.metric(
                "Critical Alerts",
                int((display_anomalies["severity"] == "Critical").sum())
                if "severity" in display_anomalies.columns else 0
            )
            c4.metric("Top Action", top_action)

            st.markdown("### AI Anomaly Intelligence")

            esc_cols = [
                "delivery_month",
                "delivery_delay_days",
                "rolling_mean",
                "rolling_z",
                "pct_change",
                "deviation_pct",
                "severity",
                "escalation_action"
            ]
            esc_cols = [col for col in esc_cols if col in display_anomalies.columns]

            esc_view = display_anomalies[esc_cols].copy()

            esc_view["Root Cause"] = esc_view["delivery_month"].apply(
                lambda x: anomaly_root_cause(x)[0] if pd.notna(x) else "Unknown"
            )
            esc_view["Recommended Action"] = esc_view["delivery_month"].apply(
                lambda x: anomaly_root_cause(x)[1] if pd.notna(x) else "Review required"
            )

            # Prefer exported action if available
            if "escalation_action" in esc_view.columns:
                esc_view["Recommended Action"] = esc_view["escalation_action"].fillna(
                    esc_view["Recommended Action"]
                )

            esc_view["delivery_month"] = pd.to_datetime(
                esc_view["delivery_month"], errors="coerce"
            ).dt.strftime("%b %Y")

            if "delivery_delay_days" in esc_view.columns:
                esc_view["delivery_delay_days"] = esc_view["delivery_delay_days"].round(2)
            if "rolling_mean" in esc_view.columns:
                esc_view["rolling_mean"] = esc_view["rolling_mean"].round(2)
            if "rolling_z" in esc_view.columns:
                esc_view["rolling_z"] = esc_view["rolling_z"].round(2)
            if "pct_change" in esc_view.columns:
                esc_view["pct_change"] = (esc_view["pct_change"] * 100).round(2)
            if "deviation_pct" in esc_view.columns:
                esc_view["deviation_pct"] = esc_view["deviation_pct"].round(2)

            esc_view = esc_view.rename(columns={
                "delivery_month": "Month",
                "delivery_delay_days": "Delivery Delay (Days)",
                "rolling_mean": "Rolling Mean",
                "rolling_z": "Z Score",
                "pct_change": "% Change",
                "deviation_pct": "Deviation %",
                "severity": "Severity",
                "escalation_action": "Escalation Action"
            })

            preferred_order = [
                "Month",
                "Delivery Delay (Days)",
                "Rolling Mean",
                "Z Score",
                "% Change",
                "Deviation %",
                "Severity",
                "Root Cause",
                "Recommended Action",
                "Escalation Action"
            ]
            esc_view = esc_view[[col for col in preferred_order if col in esc_view.columns]]

            st.dataframe(esc_view, use_container_width=True)

            st.info(f"Worst observed severity in current view: {worst_severity}")

        else:
            st.success("No anomaly escalation triggered. Operations remain stable.")