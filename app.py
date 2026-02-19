import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from statsmodels.tsa.arima.model import ARIMA

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

THRESHOLD = st.sidebar.slider(
    "Risk Threshold (for Watchlist)",
    min_value=0.30,
    max_value=0.95,
    value=0.70,
    step=0.05
)

# ==========================================================
# Financial Impact Simulation - Impact Assumptions Slidebars
# ==========================================================

st.sidebar.header("Impact Assumptions")

annual_cost_of_capital = st.sidebar.slider(
    "Annual Cost of Capital (%)",
    5.0, 30.0, 12.0, 0.5
) / 100.0

delay_cost_rate_per_day = st.sidebar.slider(
    "Delay Cost Rate per Day (% of Contract Value)",
    0.00, 0.30, 0.05, 0.01
) / 100.0

penalty_rate = st.sidebar.slider(
    "Penalty Rate (% of Contract Value)",
    0.0, 5.0, 0.5, 0.1
) / 100.0

mitigation_effectiveness = st.sidebar.slider(
    "Mitigation Effectiveness (%)",
    0, 80, 30, 5
) / 100.0

# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def load_model():
    bundle = joblib.load("models/risk_model.pkl")
    return bundle["model"], bundle["features"]

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

model, features = load_model()
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
# Predict Deterioration Risk
# ==========================================================
df["deterioration_risk_prob"] = model.predict_proba(df[features])[:, 1]

df["deterioration_risk_flag"] = (df["deterioration_risk_prob"] >= THRESHOLD).astype(int)

def risk_bucket(prob: float) -> str:
    if prob >= 0.80:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.40:
        return "Moderate"
    else:
        return "Low"

df["risk_level"] = df["deterioration_risk_prob"].apply(risk_bucket)

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

dashboard_view = st.sidebar.radio(
    "Select Dashboard",
    [
        "🔧 Operations Dashboard",
        "💰 SCM Dashboard",
        "📉 Predictive Risk",
        "💸 Financial Impact",
        "📈 Delivery Time-Series Intelligence"
    ]
)

# ==========================================================
# 🔧 OPERATIONS DASHBOARD
# ==========================================================
if dashboard_view == "🔧 Operations Dashboard":

    st.title("Operations – Vendor Action Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk Vendors (Low VPI)", df[df["VPI_Category"] == "Low"].shape[0])
    col2.metric("Avg Delivery Delay (Days)", round(df["avg_delay_days"].mean(), 2))
    col3.metric("Avg Rejection Rate (%)", round(df["rejection_rate_pct"].mean(), 2))

    st.subheader("Operations Risk Drivers")
    ops_risk = df[["avg_delay_days", "delay_frequency", "rejection_rate_pct"]].mean().reset_index()
    ops_risk.columns = ["Risk Factor", "Impact"]
    fig_ops = px.bar(ops_risk, x="Impact", y="Risk Factor", orientation="h", title="Operations Risk Contribution")
    st.plotly_chart(fig_ops, use_container_width=True)

    def ops_action(row):
        if row["avg_delay_days"] > 5:
            return "Re-baseline delivery schedule"
        if row["rejection_rate_pct"] > 7:
            return "Conduct quality audit"
        if row["delay_frequency"] > 0.5:
            return "Assign dedicated Ops SPOC"
        return "Monitor"

    df["Ops_Action"] = df.apply(ops_action, axis=1)

    st.subheader("Operations Action List")
    st.dataframe(
        df[["vendor_name", "VPI_Score", "VPI_Category", "avg_delay_days", "rejection_rate_pct", "Ops_Action"]]
        .sort_values("VPI_Score"),
        use_container_width=True
    )
    st.caption("This dashboard uses real delivery history data to track operational risk evolution over time.")
# ==========================================================
# 💰 SCM DASHBOARD
# ==========================================================
elif dashboard_view == "💰 SCM Dashboard":

    st.title("SCM – Vendor Action Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Payment Risk Vendors", df[df["avg_payment_delay"] > 7].shape[0])
    col2.metric("Avg Payment Delay (Days)", round(df["avg_payment_delay"].mean(), 2))
    col3.metric("Penalty Cases", int(df["penalty_cases"].sum()))

    st.subheader("SCM Risk Drivers")
    scm_risk = df[["avg_payment_delay", "payment_risk_ratio", "penalty_cases"]].mean().reset_index()
    scm_risk.columns = ["Risk Factor", "Impact"]
    fig_scm = px.bar(scm_risk, x="Impact", y="Risk Factor", orientation="h", title="SCM Risk Contribution")
    st.plotly_chart(fig_scm, use_container_width=True)

    def scm_action(row):
        if row["avg_payment_delay"] > 10:
            return "Fast-track pending invoices"
        if row["payment_risk_ratio"] > 0.6:
            return "SCM escalation"
        if row["penalty_cases"] > 1:
            return "Contract compliance review"
        return "Monitor"

    df["SCM_Action"] = df.apply(scm_action, axis=1)

    st.subheader("SCM Action List")
    st.dataframe(
        df[["vendor_name", "VPI_Score", "VPI_Category", "avg_payment_delay", "payment_risk_ratio", "penalty_cases", "SCM_Action"]]
        .sort_values("VPI_Score"),
        use_container_width=True
    )

# ==========================================================
# 📉 PREDICTIVE RISK DASHBOARD
# ==========================================================
elif dashboard_view == "📉 Predictive Risk":

    st.title("Predictive Vendor Deterioration Risk (Early Warning)")

    above_threshold = int((df["deterioration_risk_prob"] >= THRESHOLD).sum())
    critical_cnt = int((df["risk_level"] == "Critical").sum())
    high_cnt = int((df["risk_level"] == "High").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendors Above Threshold", above_threshold)
    c2.metric("Critical Vendors (≥ 0.80)", critical_cnt)
    c3.metric("High Vendors (0.60–0.79)", high_cnt)
    c4.metric("Risk Threshold", THRESHOLD)

    st.subheader("Risk Level Distribution")
    dist = df["risk_level"].value_counts().reset_index()
    dist.columns = ["risk_level", "count"]
    fig_pie = px.pie(dist, names="risk_level", values="count", title="Risk Level Split")
    st.plotly_chart(fig_pie, use_container_width=True)

    def proactive_action(row):
        if row["deterioration_risk_prob"] < THRESHOLD:
            return "Preventive monitoring"

        actions = []
        if row["avg_delay_days"] > 5:
            actions.append("Stabilize delivery schedule")
        if row["avg_payment_delay"] > 7:
            actions.append("Review payment cycle & clear backlog")
        if row["rejection_rate_pct"] > 7:
            actions.append("Initiate quality audit")
        if row["penalty_cases"] > 1:
            actions.append("Review contract compliance")

        return " | ".join(actions) if actions else "Monitor closely"

    st.subheader("High-Risk Vendor Watchlist")
    watchlist = df[df["deterioration_risk_prob"] >= THRESHOLD] \
        .sort_values("deterioration_risk_prob", ascending=False) \
        .copy()

    if watchlist.empty:
        st.info("No vendors above the current threshold. Try lowering the threshold if you want a broader watchlist.")
    else:
        watchlist["Recommended_Action"] = watchlist.apply(proactive_action, axis=1)
        watchlist_view = watchlist[
            ["vendor_name", "VPI_Score", "VPI_Category", "risk_level", "deterioration_risk_prob", "Recommended_Action"]
        ].head(25)
        st.dataframe(watchlist_view, use_container_width=True)

    st.subheader("Risk Probability Distribution")
    fig_hist = px.histogram(df, x="deterioration_risk_prob", nbins=20, title="Deterioration Risk Probability Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.subheader("Key Risk Drivers (Feature Importance)")
        fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Model Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

# ==========================================================
# 💸 Financial Impact Simulation
# ==========================================================

elif dashboard_view == "💸 Financial Impact":

    st.title("Financial Impact Simulation (₹ Exposure & Savings)")

    # Filter to vendors above threshold for “at risk” view
    risk_df = df[df["deterioration_risk_prob"] >= THRESHOLD].copy()

    if risk_df.empty:
        st.info("No vendors above the current threshold. Lower the threshold to simulate broader impact.")
        st.stop()

    # -----------------------------
    # SAFETY: Ensure required columns exist
    # -----------------------------
    required_cols = ["contract_value_lakhs", "avg_delay_days", "avg_payment_delay"]
    missing = [c for c in required_cols if c not in risk_df.columns]
    if missing:
        st.error(f"Missing required columns for impact simulation: {missing}")
        st.stop()

    # -----------------------------
    # 1) Working capital cost (payment delay)
    # WorkingCapitalCost = InvoiceValue * (AnnualRate/365) * PaymentDelayDays
    # We do not have invoice total always; use contract_value_lakhs as proxy exposure
    # -----------------------------
    risk_df["working_capital_cost_lakhs"] = (
        risk_df["contract_value_lakhs"]
        * (annual_cost_of_capital / 365.0)
        * risk_df["avg_payment_delay"].clip(lower=0)
    )

    # -----------------------------
    # 2) Delivery delay cost exposure (proxy)
    # DelayExposure = ContractValue * DelayCostRatePerDay * AvgDelayDays
    # -----------------------------
    risk_df["delivery_delay_cost_lakhs"] = (
        risk_df["contract_value_lakhs"]
        * delay_cost_rate_per_day
        * risk_df["avg_delay_days"].clip(lower=0)
    )

    # -----------------------------
    # 3) Penalty exposure (proxy)
    # PenaltyExposure = ContractValue * PenaltyRate * min(penalty_cases, 1)
    # (Keeps it simple; not compounding)
    # -----------------------------
    if "penalty_cases" in risk_df.columns:
        risk_df["penalty_exposure_lakhs"] = (
            risk_df["contract_value_lakhs"]
            * penalty_rate
            * (risk_df["penalty_cases"] > 0).astype(int)
        )
    else:
        risk_df["penalty_exposure_lakhs"] = 0.0

    # -----------------------------
    # Total exposure + savings scenario
    # -----------------------------
    risk_df["total_exposure_lakhs"] = (
        risk_df["working_capital_cost_lakhs"]
        + risk_df["delivery_delay_cost_lakhs"]
        + risk_df["penalty_exposure_lakhs"]
    )

    risk_df["estimated_savings_lakhs"] = risk_df["total_exposure_lakhs"] * mitigation_effectiveness

    # -----------------------------
    # KPI Cards
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Vendors in Risk Set", risk_df.shape[0])
    c2.metric("Total Exposure (₹ Lakhs)", round(risk_df["total_exposure_lakhs"].sum(), 2))
    c3.metric("Estimated Savings (₹ Lakhs)", round(risk_df["estimated_savings_lakhs"].sum(), 2))
    c4.metric("Assumed Mitigation (%)", int(mitigation_effectiveness * 100))

    # -----------------------------
    # Breakdown chart
    # -----------------------------
    st.subheader("Exposure Breakdown (₹ Lakhs)")

    breakdown = pd.DataFrame({
        "Component": ["Working Capital Cost", "Delivery Delay Cost", "Penalty Exposure"],
        "Amount_Lakhs": [
            risk_df["working_capital_cost_lakhs"].sum(),
            risk_df["delivery_delay_cost_lakhs"].sum(),
            risk_df["penalty_exposure_lakhs"].sum()
        ]
    })

    fig = px.pie(breakdown, names="Component", values="Amount_Lakhs", title="Total Exposure Split")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Top vendors by exposure
    # -----------------------------
    st.subheader("Top Vendors by Exposure")

    show_cols = [
        "vendor_name",
        "VPI_Score",
        "VPI_Category",
        "deterioration_risk_prob",
        "contract_value_lakhs",
        "avg_delay_days",
        "avg_payment_delay",
        "total_exposure_lakhs",
        "estimated_savings_lakhs"
    ]

    if "risk_level" in risk_df.columns:
        show_cols.insert(4, "risk_level")

    top_exposure = risk_df.sort_values("total_exposure_lakhs", ascending=False)[show_cols].head(20)

    st.dataframe(top_exposure, use_container_width=True)

# -----------------------------
# Exposure by Region
# -----------------------------
    if "location" in risk_df.columns:
       st.subheader("Exposure by Region (₹ Lakhs)")

       region_exposure = (
           risk_df.groupby("location")["total_exposure_lakhs"]
           .sum()
           .reset_index()
           .sort_values("total_exposure_lakhs", ascending=False)
       )

       fig_region = px.bar(
           region_exposure,
           x="location",
           y="total_exposure_lakhs",
           title="Total Financial Exposure by Region",
       )

       st.plotly_chart(fig_region, use_container_width=True)

# -----------------------------
# Exposure by Vendor Type
# -----------------------------
    if "vendor_type" in risk_df.columns:
       st.subheader("Exposure by Vendor Type (₹ Lakhs)")

       type_exposure = (
           risk_df.groupby("vendor_type")["total_exposure_lakhs"]
           .sum()
           .reset_index()
           .sort_values("total_exposure_lakhs", ascending=False)
       )

       fig_type = px.bar(
           type_exposure,
           x="vendor_type",
           y="total_exposure_lakhs",
           title="Total Financial Exposure by Vendor Type",
       )

       st.plotly_chart(fig_type, use_container_width=True)

    st.caption(
        "Note: This is a decision-support simulation using proxy assumptions. "
        "It helps leadership prioritize interventions and estimate potential savings."
    )

# ==========================================================
# 📈 Delivery Time-Series Intelligence
# ==========================================================
elif dashboard_view == "📈 Delivery Time-Series Intelligence":

    st.title("Delivery Time-Series Intelligence")

    # 1️⃣ Avg Delivery Delay Trend
    st.subheader("Average Delivery Delay Trend")

    fig_delay = px.line(
        monthly_trend,
        x="delivery_month",
        y="delivery_delay_days",
        markers=True,
        title="Average Delivery Delay Over Time"
    )
    st.plotly_chart(fig_delay, use_container_width=True)

    # --------------------------------------------------
    # ARIMA Forecast
    # --------------------------------------------------
    st.subheader("ARIMA Forecast (Next 3 Months)")

    ts_series = monthly_trend.set_index("delivery_month")["delivery_delay_days"]

    try:
        arima_model = ARIMA(ts_series, order=(1, 1, 1))
        model_fit = arima_model.fit()
        forecast = model_fit.forecast(steps=3)

        future_dates = pd.date_range(
            start=ts_series.index.max() + pd.offsets.MonthEnd(1),
            periods=3,
            freq="M"
        )

        forecast_df = pd.DataFrame({
            "delivery_month": future_dates,
            "forecast_delay": forecast.values
        })

        hist_df = monthly_trend.copy().rename(columns={"delivery_delay_days": "value"})
        hist_df["type"] = "Historical"

        forecast_plot_df = forecast_df.rename(columns={"forecast_delay": "value"})
        forecast_plot_df["type"] = "Forecast"

        combined = pd.concat([
            hist_df[["delivery_month", "value", "type"]],
            forecast_plot_df[["delivery_month", "value", "type"]]
        ], ignore_index=True)

        fig_forecast = px.line(
            combined,
            x="delivery_month",
            y="value",
            color="type",
            markers=True,
            title="Delivery Delay: Historical vs Forecast"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.success("Forecast generated successfully for next 3 months.")

    except Exception as e:
        st.error(f"Forecast failed: {e}")

    st.caption("ARIMA(1,1,1) model used for short-term delivery delay forecasting.")

    # --------------------------------------------------
    # Anomaly Detection Controls
    # --------------------------------------------------
    st.subheader("Delay Spike Anomaly Detection")

    sensitivity = st.slider(
        "Anomaly Sensitivity (Z-score threshold)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

    anomaly_df = monthly_trend.copy()
    anomaly_df["rolling_mean"] = anomaly_df["delivery_delay_days"].rolling(3).mean()
    anomaly_df["rolling_std"] = anomaly_df["delivery_delay_days"].rolling(3).std()

    anomaly_df["rolling_z"] = (
        (anomaly_df["delivery_delay_days"] - anomaly_df["rolling_mean"])
        / anomaly_df["rolling_std"]
    )

    anomaly_df["anomaly_flag"] = anomaly_df["rolling_z"].abs() > sensitivity
    anomalies = anomaly_df[anomaly_df["anomaly_flag"] == True].copy()

    # Plot anomalies
    fig_anomaly = px.line(
        anomaly_df,
        x="delivery_month",
        y="delivery_delay_days",
        markers=True,
        title="Delivery Delay with Anomaly Detection"
    )

    if not anomalies.empty:
        fig_anomaly.add_scatter(
            x=anomalies["delivery_month"],
            y=anomalies["delivery_delay_days"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Anomaly"
        )

    st.plotly_chart(fig_anomaly, use_container_width=True)

    # Show anomaly table
    if not anomalies.empty:
        st.warning("Anomalies detected in the following months:")

        anomalies_display = anomalies.copy()
        anomalies_display["delivery_month"] = anomalies_display["delivery_month"].dt.strftime("%b %Y")

        st.dataframe(
            anomalies_display[["delivery_month", "delivery_delay_days", "rolling_z"]],
            use_container_width=True
        )
    else:
        st.info("No anomalies detected under current sensitivity setting.")

    # --------------------------------------------------
    # Automatic Escalation Logic
    # --------------------------------------------------
    st.subheader("🚨 Automatic Escalation (Anomaly-triggered)")

    if not anomalies.empty:
        escalation_df = anomalies.copy()
        escalation_df["abs_z"] = escalation_df["rolling_z"].abs()

        # Percentile thresholds inside detected anomalies
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
                # Moderate: differentiate by direction
                if z > 0:
                    return "Ops check: site readiness + resourcing validation"
                else:
                    return "Positive deviation: validate data & capture best practice"

        escalation_df["escalation_action"] = escalation_df.apply(
                lambda r: escalation_action(r["severity"], r["rolling_z"]),
                axis=1
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomaly Months", escalation_df.shape[0])
        c2.metric("Critical Alerts", int((escalation_df["severity"] == "Critical").sum()))
        c3.metric("High Alerts", int((escalation_df["severity"] == "High").sum()))

        st.error("Escalation Triggered: Anomalous delivery delay spikes detected.")

        st.markdown("### Escalation Queue")
        st.dataframe(
            escalation_df[[
                "delivery_month",
                "delivery_delay_days",
                "rolling_z",
                "severity",
                "escalation_action"
            ]].sort_values(["severity", "rolling_z"], ascending=[True, False]),
            use_container_width=True
        )

    else:
        st.success("No anomaly escalation triggered. Operations remain stable.")

