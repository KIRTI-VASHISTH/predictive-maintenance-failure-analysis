import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "Models"

X_test = joblib.load(MODEL_DIR / "X_test.joblib")
y_test = joblib.load(MODEL_DIR / "y_test.joblib")



DATA_CANDIDATES = [
    PROJECT_DIR / "data" / "ai4i2020.csv",
    PROJECT_DIR / "data" /
    "ai4i+2020+predictive+maintenance+dataset" / "ai4i2020.csv",
]

RF_MODEL_PATH = PROJECT_DIR / "Models" / "best_random_forest.joblib"
IF_MODEL_PATH = PROJECT_DIR / "Models" / "isolation_forest.joblib"
THRESHOLD_PATH = PROJECT_DIR / "Models" / "anomaly_threshold.joblib"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]


# --------------------------------------------------
# 2. LOAD DATA AND MODELS
# --------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL_PATH)
    if_model = joblib.load(IF_MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    return rf_model, if_model, threshold


st.title("⚙️ Predictive Maintenance Dashboard")
st.caption(
    "Machine failure risk, anomaly detection and maintenance prioritisation"
)

data_path = next(
    (path for path in DATA_CANDIDATES if path.exists()),
    None
)

if data_path is None:
    st.error(
        "Dataset not found. Check DATA_CANDIDATES near the top of app.py "
        "and update it to your actual CSV location."
    )
    st.stop()

df = load_dataset(str(data_path))

# Keep the same feature order used when training the models.
missing_features = [c for c in FEATURES if c not in df.columns]

if missing_features:
    st.error(f"Missing model features in dataset: {missing_features}")
    st.stop()

if "Machine failure" not in df.columns:
    st.error("The dataset must contain the Machine failure column.")
    st.stop()


# --------------------------------------------------
# 3. FIVE DASHBOARD TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Machine Simulator",
    "Maintenance Recommendations",
    "Explainability",
    "Model Performance",
])


# --------------------------------------------------
# 4. OVERVIEW
# --------------------------------------------------
with tab1:
    st.header("Maintenance Overview")

    try:
        rf_model, if_model, anomaly_threshold = load_models()

        X = df[FEATURES].copy()

        # Failure probability for class 1, regardless of class order.
        failure_index = list(rf_model.classes_).index(1)
        df["Failure Probability"] = rf_model.predict_proba(X)[
            :, failure_index
        ]

        # Higher score means more anomalous.
        df["Anomaly Score"] = -if_model.decision_function(X)
        df["Anomaly Detected"] = (
            df["Anomaly Score"] >= anomaly_threshold
        )

        # Same provisional decision rules as the project decision engine.
        risk = df["Failure Probability"]
        anomaly = df["Anomaly Detected"]

        df["Maintenance Priority"] = np.select(
            [
                (risk >= 0.50) & anomaly,
                (risk >= 0.50) | ((risk >= 0.10) & anomaly),
                (risk >= 0.10) | anomaly,
            ],
            ["Critical", "High", "Medium"],
            default="Low",
        )

        df["Maintenance Needed"] = (
            df["Maintenance Priority"] != "Low"
        )

        # Show the sensor parameter furthest from its dataset median.
        # This is a review hint, NOT a causal explanation.
        median_values = df[NUMERIC_FEATURES].median()
        std_values = df[NUMERIC_FEATURES].std().replace(0, 1)

        deviation = (
            df[NUMERIC_FEATURES] - median_values
        ).abs().div(std_values)

        df["Parameter to Review"] = deviation.idxmax(axis=1)

        def describe_task(row):
            parameter = row["Parameter to Review"]
            priority = row["Maintenance Priority"]

            if priority == "Critical":
                action = "Urgent inspection recommended"
            elif priority == "High":
                action = "Schedule inspection soon"
            elif priority == "Medium":
                action = "Review operating conditions"
            else:
                action = "Routine monitoring"

            return f"{action}; review {parameter}"

        df["Task Description"] = df.apply(describe_task, axis=1)

        # ------------------------------------------
        # TOP KPIs
        # ------------------------------------------
        total_records = len(df)
        predicted_high_risk = int((risk >= 0.50).sum())
        maintenance_count = int(df["Maintenance Needed"].sum())
        anomaly_count = int(df["Anomaly Detected"].sum())
        avg_risk = float(risk.mean() * 100)

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("Records Scored", f"{total_records:,}")
        k2.metric("High-Risk Predictions", f"{predicted_high_risk:,}")
        k3.metric("Maintenance Flags", f"{maintenance_count:,}")
        k4.metric("Anomalies", f"{anomaly_count:,}")
        k5.metric("Average Predicted Risk", f"{avg_risk:.2f}%")

        st.caption(
            "These are predictions on the loaded dataset, not live machine "
            "readings. Risk thresholds are provisional decision rules. "
            "Average predicted risk is not the observed failure rate."
        )

        st.divider()

        # ------------------------------------------
        # CHARTS
        # ------------------------------------------
        left, right = st.columns(2)

        with left:
            st.subheader("Maintenance Workload by Priority")

            priority_counts = (
                df["Maintenance Priority"]
                .value_counts()
                .reindex(PRIORITY_ORDER, fill_value=0)
                .rename_axis("Priority")
                .reset_index(name="Records")
            )

            fig = px.bar(
                priority_counts,
                x="Priority",
                y="Records",
                category_orders={"Priority": PRIORITY_ORDER},
                text="Records",
                title="Records by Maintenance Priority",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Predicted Failure Risk")

            fig = px.histogram(
                df,
                x="Failure Probability",
                nbins=30,
                title="Distribution of Predicted Failure Probabilities",
                labels={
                    "Failure Probability": "Predicted failure probability"
                },
            )
            fig.add_vline(
                x=0.10,
                line_dash="dash",
                annotation_text="Medium-risk threshold",
            )
            fig.add_vline(
                x=0.50,
                line_dash="dash",
                annotation_text="High-risk threshold",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ------------------------------------------
        # MAINTENANCE WORKLIST
        # ------------------------------------------
        st.subheader("Prioritised Maintenance Worklist")

        filter_col, search_col = st.columns([1, 2])

        with filter_col:
            selected_priorities = st.multiselect(
                "Filter by priority",
                PRIORITY_ORDER,
                default=["Critical", "High", "Medium"],
            )

        with search_col:
            search_text = st.text_input(
                "Search record or parameter",
                placeholder="e.g. Torque or 125"
            )

        worklist = df.copy()

        worklist["_priority_sort"] = worklist[
            "Maintenance Priority"
        ].map({
            "Critical": 0,
            "High": 1,
            "Medium": 2,
            "Low": 3,
        })

        worklist = worklist[
            worklist["Maintenance Priority"].isin(selected_priorities)
        ]

        if search_text:
            mask = (
                worklist["Parameter to Review"].str.contains(
                    search_text, case=False, na=False
                )
                | worklist["Task Description"].str.contains(
                    search_text, case=False, na=False
                )
            )

            if "UDI" in worklist.columns:
                mask |= worklist["UDI"].astype(str).str.contains(
                    search_text, case=False, na=False
                )

            worklist = worklist[mask]

        worklist = worklist.sort_values(
            ["_priority_sort", "Failure Probability"],
            ascending=[True, False],
        )

        display_columns = []

        if "UDI" in worklist.columns:
            display_columns.append("UDI")

        display_columns += [
            "Type",
            "Failure Probability",
            "Anomaly Detected",
            "Maintenance Priority",
            "Parameter to Review",
            "Task Description",
        ]

        display_df = worklist[display_columns].copy()

        display_df["Failure Probability"] = (
            display_df["Failure Probability"].map(
                lambda value: f"{value:.1%}"
            )
        )

        display_df = display_df.rename(columns={
            "UDI": "Record ID",
            "Type": "Product Type",
            "Failure Probability": "Predicted Failure Risk",
            "Anomaly Detected": "Anomaly Flag",
            "Maintenance Priority": "Priority",
            "Parameter to Review": "Parameter to Inspect",
            "Task Description": "Suggested Task",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "'Parameter to Inspect' identifies the numeric sensor parameter "
            "furthest from its dataset median after scaling by standard "
            "deviation. It is a screening hint, not proof that the parameter "
            "caused a failure. We will replace it with a local SHAP "
            "explanation when we connect the Explainability tab."
        )

        csv = worklist[display_columns].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download maintenance worklist as CSV",
            data=csv,
            file_name="maintenance_worklist.csv",
            mime="text/csv",
        )

    except FileNotFoundError as error:
        st.error(
            "A saved model or anomaly threshold file could not be found. "
            "Expected files:\n"
            "- models/best_random_forest.joblib\n"
            "- models/isolation_forest.joblib\n"
            "- models/anomaly_threshold.joblib"
        )
        st.code(str(error))

    except Exception as error:
        st.error("The Overview could not finish scoring the dataset.")
        st.exception(error)

with tab2:
    st.header("Machine Simulator")
    st.write(
        "Adjust the machine's operating parameters to simulate "
        "a machine condition and evaluate its maintenance risk."
    )

    st.info(
        "This is a what-if simulator using the trained models. "
        "It does not connect to a live machine."
    )

    # Input controls
    st.subheader("Machine Operating Parameters")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            index=0,
            help="L = Low, M = Medium, H = High product quality type."
        )

        air_temp = st.slider(
            "Air Temperature (K)",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.1
        )

        process_temp = st.slider(
            "Process Temperature (K)",
            min_value=305.0,
            max_value=315.0,
            value=310.0,
            step=0.1
        )

        rotational_speed = st.slider(
            "Rotational Speed (rpm)",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=10
        )

    with col2:
        torque = st.slider(
            "Torque (Nm)",
            min_value=3.0,
            max_value=80.0,
            value=40.0,
            step=0.5
        )

        tool_wear = st.slider(
            "Tool Wear (min)",
            min_value=0,
            max_value=250,
            value=100,
            step=1
        )

    # Create a single-row DataFrame with the exact training feature names
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    st.divider()

    # Predict when the user changes a parameter
    try:
        failure_class_index = list(rf_model.classes_).index(1)

        failure_probability = float(
            rf_model.predict_proba(input_data)[0][failure_class_index]
        )

        anomaly_score = float(
            -if_model.decision_function(input_data)[0]
        )

        anomaly_detected = anomaly_score >= anomaly_threshold

        # Apply the same priority rules used in the dashboard
        if failure_probability >= 0.50 and anomaly_detected:
            priority = "Critical"
        elif (
            failure_probability >= 0.50
            or (failure_probability >= 0.10 and anomaly_detected)
        ):
            priority = "High"
        elif failure_probability >= 0.10 or anomaly_detected:
            priority = "Medium"
        else:
            priority = "Low"

        # Results
        st.subheader("Simulation Results")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Predicted Failure Risk",
            f"{failure_probability:.1%}"
        )

        metric2.metric(
            "Anomaly Score",
            f"{anomaly_score:.4f}"
        )

        metric3.metric(
            "Maintenance Priority",
            priority
        )

        # Risk interpretation
        if failure_probability >= 0.50:
            st.error(
                "High predicted failure risk. Arrange prompt inspection "
                "and verify the machine's operating condition."
            )
        elif failure_probability >= 0.10:
            st.warning(
                "Elevated predicted failure risk. Review the machine "
                "condition and plan an inspection as appropriate."
            )
        else:
            st.success(
                "The model predicts a relatively low failure risk. "
                "Continue routine monitoring; this is not a guarantee "
                "that the machine is safe."
            )

        if anomaly_detected:
            st.warning(
                "Anomaly detected: the anomaly score meets or exceeds "
                "the configured threshold."
            )
        else:
            st.info(
                "No anomaly detected under the configured threshold."
            )

        # Show the exact inputs used for this prediction
        with st.expander("View input values used by the models"):
            st.dataframe(input_data, use_container_width=True)

        # Visualize failure probability
        st.subheader("Failure Risk Gauge")

        risk_percent = failure_probability * 100

        risk_chart = {
            "Risk (%)": ["Predicted risk", "Remaining"],
            "Percentage": [
                risk_percent,
                max(0, 100 - risk_percent)
            ]
        }

        risk_df = pd.DataFrame(risk_chart)

        fig = px.pie(
            risk_df,
            names="Risk (%)",
            values="Percentage",
            hole=0.65,
            title="Predicted Failure Probability"
        )

        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
        )

        fig.update_layout(
            showlegend=False,
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download simulated input and prediction
        result = input_data.copy()
        result["Failure Probability"] = failure_probability
        result["Anomaly Score"] = anomaly_score
        result["Anomaly Detected"] = anomaly_detected
        result["Maintenance Priority"] = priority

        st.download_button(
            label="Download Simulation Result",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="machine_simulation_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Could not run the simulation: {e}")


# --------------------------------------------------
# TAB 3: MAINTENANCE RECOMMENDATIONS
# --------------------------------------------------

with tab3:
    st.header("Maintenance Recommendations")
    st.write(
        "Prioritize machine records using predicted failure risk, "
        "anomaly detection, and recorded failure-mode evidence."
    )

    st.info(
        "Failure probabilities and priorities are model-derived. "
        "Failure-mode evidence comes from recorded dataset labels; "
        "it does not mean the model has predicted a specific failure mode."
    )

    # Required dataset columns
    failure_mode_columns = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
    }

    # Verify source columns
    missing_features = [col for col in FEATURES if col not in df.columns]

    if missing_features:
        st.error(
            f"Required model features are missing from the dataset: "
            f"{missing_features}"
        )
        st.stop()

    # Work with a fresh copy of the source dataset
    maintenance_df = df.copy()

    # Use the existing trained models and threshold
    X_maintenance = maintenance_df[FEATURES]

    failure_class_index = list(rf_model.classes_).index(1)

    maintenance_df["Failure Probability"] = (
        rf_model.predict_proba(X_maintenance)[:, failure_class_index]
    )

    maintenance_df["Anomaly Score"] = (
        -if_model.decision_function(X_maintenance)
    )

    maintenance_df["Anomaly Detected"] = (
        maintenance_df["Anomaly Score"] >= anomaly_threshold
    )

    # Existing maintenance decision rules
    risk = maintenance_df["Failure Probability"]
    anomaly = maintenance_df["Anomaly Detected"]

    maintenance_df["Maintenance Priority"] = np.select(
        [
            (risk >= 0.50) & anomaly,
            (risk >= 0.50) | ((risk >= 0.10) & anomaly),
            (risk >= 0.10) | anomaly,
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )

    maintenance_df["Maintenance Needed"] = (
        maintenance_df["Maintenance Priority"] != "Low"
    )

    # Maintenance action labels derived from priority rules
    action_map = {
        "Critical": "Prompt inspection; verify machine condition",
        "High": "Prioritize inspection and review operating conditions",
        "Medium": "Schedule inspection and monitor sensor readings",
        "Low": "Continue routine monitoring",
    }

    maintenance_df["Suggested Action"] = (
        maintenance_df["Maintenance Priority"].map(action_map)
    )

    # --------------------------------------------------
    # SECTION 1: FILTERS
    # --------------------------------------------------

    st.subheader("Filter Maintenance Worklist")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_priorities = st.multiselect(
            "Maintenance Priority",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium"],
        )

    with filter_col2:
        product_types = sorted(
            maintenance_df["Type"].dropna().unique().tolist()
        )

        selected_types = st.multiselect(
            "Product Type",
            options=product_types,
            default=product_types,
        )

    with filter_col3:
        anomaly_filter = st.selectbox(
            "Anomaly Status",
            options=["All", "Anomaly detected", "No anomaly"],
        )

    filtered_df = maintenance_df[
        maintenance_df["Maintenance Priority"].isin(selected_priorities)
        & maintenance_df["Type"].isin(selected_types)
    ].copy()

    if anomaly_filter == "Anomaly detected":
        filtered_df = filtered_df[
            filtered_df["Anomaly Detected"]
        ]
    elif anomaly_filter == "No anomaly":
        filtered_df = filtered_df[
            ~filtered_df["Anomaly Detected"]
        ]

    # --------------------------------------------------
    # SECTION 2: WORKLIST KPIs
    # --------------------------------------------------

    st.subheader("Maintenance Worklist Summary")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        "Records in Worklist",
        f"{len(filtered_df):,}",
    )

    kpi2.metric(
        "Critical Priority",
        int(
            (filtered_df["Maintenance Priority"] == "Critical").sum()
        ),
    )

    kpi3.metric(
        "High Priority",
        int(
            (filtered_df["Maintenance Priority"] == "High").sum()
        ),
    )

    kpi4.metric(
        "Anomalies Detected",
        int(filtered_df["Anomaly Detected"].sum()),
    )

    # --------------------------------------------------
    # SECTION 3: PRIORITIZED WORKLIST
    # --------------------------------------------------

    st.subheader("Prioritized Maintenance Worklist")

    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }

    filtered_df["_Priority Order"] = (
        filtered_df["Maintenance Priority"].map(priority_order)
    )

    filtered_df = filtered_df.sort_values(
        by=["_Priority Order", "Failure Probability"],
        ascending=[True, False],
    )

    # Show only columns available in the dataset
    worklist_columns = [
        col for col in [
            "UDI",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Failure Probability",
            "Anomaly Score",
            "Anomaly Detected",
            "Maintenance Priority",
            "Suggested Action",
        ]
        if col in filtered_df.columns
    ]

    display_worklist = filtered_df[worklist_columns].copy()

    display_worklist["Failure Probability"] = (
        display_worklist["Failure Probability"].map(
            lambda x: f"{x:.1%}"
        )
    )

    display_worklist["Anomaly Score"] = (
        display_worklist["Anomaly Score"].map(
            lambda x: f"{x:.4f}"
        )
    )

    st.dataframe(
        display_worklist,
        use_container_width=True,
        hide_index=True,
    )

    # Download numeric values without display formatting
    download_columns = [
        col for col in worklist_columns
        if col in filtered_df.columns
    ]

    st.download_button(
        label="Download Maintenance Worklist (CSV)",
        data=filtered_df[download_columns].to_csv(
            index=False
        ).encode("utf-8"),
        file_name="maintenance_worklist.csv",
        mime="text/csv",
    )

    # --------------------------------------------------
    # SECTION 4: FAILURE-MODE EVIDENCE
    # --------------------------------------------------

    st.divider()
    st.subheader("Recorded Failure-Mode Evidence")

    available_modes = {
        col: label
        for col, label in failure_mode_columns.items()
        if col in maintenance_df.columns
    }

    if not available_modes:
        st.warning(
            "No failure-mode label columns were found in the source dataset."
        )
    else:
        st.caption(
            "These charts use recorded failure-mode labels. "
            "They describe historical records and are not failure-mode "
            "predictions from the Random Forest."
        )

        # Historical failure-mode counts in the source dataset
        mode_counts = pd.DataFrame([
            {
                "Failure Mode": label,
                "Recorded Cases": int(
                    (pd.to_numeric(
                        maintenance_df[col], errors="coerce"
                    ) == 1).sum()
                ),
            }
            for col, label in available_modes.items()
        ]).sort_values(
            "Recorded Cases",
            ascending=False,
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Failure Modes — Entire Dataset**")

            fig_modes = px.bar(
                mode_counts,
                x="Failure Mode",
                y="Recorded Cases",
                title="Recorded Failure-Mode Counts",
                text="Recorded Cases",
            )

            fig_modes.update_layout(
                xaxis_title="Recorded Failure Mode",
                yaxis_title="Number of Records",
            )

            st.plotly_chart(
                fig_modes,
                use_container_width=True,
            )

        # Recorded failure-mode labels among maintenance-flagged records
        with chart_col2:
            st.markdown("**Failure Modes — Maintenance-Flagged Records**")

            flagged_df = maintenance_df[
                maintenance_df["Maintenance Needed"]
            ]

            flagged_counts = pd.DataFrame([
                {
                    "Failure Mode": label,
                    "Recorded Cases": int(
                        (
                            pd.to_numeric(
                                flagged_df[col],
                                errors="coerce",
                            ) == 1
                        ).sum()
                    ),
                }
                for col, label in available_modes.items()
            ]).sort_values(
                "Recorded Cases",
                ascending=False,
            )

            fig_flagged = px.bar(
                flagged_counts,
                x="Failure Mode",
                y="Recorded Cases",
                title="Recorded Failure Modes in Flagged Records",
                text="Recorded Cases",
            )

            fig_flagged.update_layout(
                xaxis_title="Recorded Failure Mode",
                yaxis_title="Number of Records",
            )

            st.plotly_chart(
                fig_flagged,
                use_container_width=True,
            )

        # --------------------------------------------------
        # SECTION 5: FAILURE RATE BY PRODUCT TYPE
        # --------------------------------------------------

        if "Machine failure" in maintenance_df.columns:
            st.subheader("Recorded Failure Rate by Product Type")

            product_failure = (
                maintenance_df.groupby("Type", dropna=False)
                .agg(
                    Total_Records=("Machine failure", "size"),
                    Recorded_Failures=("Machine failure", "sum"),
                )
                .reset_index()
            )

            product_failure["Failure Rate (%)"] = (
                100
                * product_failure["Recorded_Failures"]
                / product_failure["Total_Records"].replace(0, np.nan)
            )

            fig_product = px.bar(
                product_failure,
                x="Type",
                y="Failure Rate (%)",
                text=product_failure["Failure Rate (%)"].map(
                    lambda x: f"{x:.2f}%"
                ),
                title="Observed Machine Failure Rate by Product Type",
                hover_data=[
                    "Total_Records",
                    "Recorded_Failures",
                ],
            )

            fig_product.update_layout(
                xaxis_title="Product Type",
                yaxis_title="Observed Failure Rate (%)",
            )

            st.plotly_chart(
                fig_product,
                use_container_width=True,
            )

        # --------------------------------------------------
        # SECTION 6: SELECT A RECORD FOR EVIDENCE
        # --------------------------------------------------

        st.subheader("Inspect a Maintenance Record")

        if len(filtered_df) > 0:
            if "UDI" in filtered_df.columns:
                selected_udi = st.selectbox(
                    "Select UDI",
                    options=filtered_df["UDI"].tolist(),
                )

                selected_row = filtered_df[
                    filtered_df["UDI"] == selected_udi
                ].iloc[0]
            else:
                selected_index = st.selectbox(
                    "Select record",
                    options=filtered_df.index.tolist(),
                )

                selected_row = filtered_df.loc[selected_index]

            evidence_col1, evidence_col2 = st.columns(2)

            with evidence_col1:
                st.markdown("**Model-derived results**")
                st.write(
                    f"Predicted failure probability: "
                    f"{selected_row['Failure Probability']:.2%}"
                )
                st.write(
                    f"Anomaly detected: "
                    f"{bool(selected_row['Anomaly Detected'])}"
                )
                st.write(
                    f"Maintenance priority: "
                    f"{selected_row['Maintenance Priority']}"
                )

            with evidence_col2:
                st.markdown("**Recorded failure-mode labels**")

                recorded_modes = []

                for col, label in available_modes.items():
                    value = pd.to_numeric(
                        pd.Series([selected_row[col]]),
                        errors="coerce",
                    ).iloc[0]

                    if pd.notna(value) and value == 1:
                        recorded_modes.append(label)

                if recorded_modes:
                    for mode in recorded_modes:
                        st.write(f"- {mode}")
                else:
                    st.write(
                        "No failure-mode label is recorded as 1 "
                        "for this record."
                    )

                if "Machine failure" in selected_row.index:
                    st.write(
                        "Recorded machine failure: "
                        f"{int(selected_row['Machine failure'])}"
                    )

                st.caption(
                    "Recorded labels are historical outcomes, not "
                    "predictions of what will fail next."
                )
        else:
            st.info(
                "No records match the selected filters. "
                "Change the filters to view the worklist."
            )



# --------------------------------------------------
# TAB 4: EXPLAINABILITY & FEATURE ANALYSIS
# --------------------------------------------------

with tab4:
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance

    st.header("Explainability & Feature Analysis")
    st.write(
        "Understand which machine operating parameters influence "
        "the trained model's failure-risk predictions."
    )

    st.info(
        "Feature importance and SHAP values explain model behaviour. "
        "They do not establish physical causation."
    )

    # --------------------------------------------------
    # SECTION 1: VALIDATE INPUTS
    # --------------------------------------------------

    missing_features = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:
        st.error(
            f"Missing required model features: {missing_features}"
        )
        st.stop()

    X_explain = df[FEATURES].copy()

    # Use the recorded target only for evaluating feature importance.
    # It is NOT passed to the model as an input feature.
    has_target = "Machine failure" in df.columns

    if has_target:
        y_explain = pd.to_numeric(
            df["Machine failure"], errors="coerce"
        )

        valid_rows = y_explain.notna()

        X_explain = X_explain.loc[valid_rows].copy()
        y_explain = y_explain.loc[valid_rows].astype(int).copy()

    failure_class_index = list(rf_model.classes_).index(1)

    # Keep analysis responsive on large datasets
    MAX_ROWS = 1500

    if len(X_explain) > MAX_ROWS:
        if has_target:
            sample_indices = (
                X_explain.sample(
                    n=MAX_ROWS,
                    random_state=42
                ).index
            )
            X_analysis = X_explain.loc[sample_indices]
            y_analysis = y_explain.loc[sample_indices]
        else:
            X_analysis = X_explain.sample(
                n=MAX_ROWS,
                random_state=42
            )
            y_analysis = None
    else:
        X_analysis = X_explain.copy()
        y_analysis = y_explain if has_target else None

    # --------------------------------------------------
    # SECTION 2: GLOBAL PERMUTATION IMPORTANCE
    # --------------------------------------------------

    st.subheader("Global Feature Importance")

    st.write(
        "Permutation importance measures how much model performance "
        "changes when one input feature is shuffled."
    )

    if has_target and y_analysis is not None:
        @st.cache_data(show_spinner="Calculating feature importance...")
        def get_permutation_importance(_model, X_data, y_data):
            return permutation_importance(
                _model,
                X_data,
                y_data,
                scoring="average_precision",
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )

        try:
            importance_result = get_permutation_importance(
                rf_model,
                X_analysis,
                y_analysis,
            )

            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": importance_result.importances_mean,
                "Standard Deviation": importance_result.importances_std,
            }).sort_values(
                "Importance",
                ascending=True,
            )

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                error_x="Standard Deviation",
                title="Permutation Importance — Average Precision",
                labels={
                    "Importance": "Decrease in Average Precision",
                    "Feature": "Input Feature",
                },
            )

            fig_importance.update_layout(
                height=420,
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )

            st.caption(
                "Positive values indicate that shuffling a feature "
                "reduced model performance. Values near zero indicate "
                "little measured impact in this evaluation sample."
            )

            with st.expander("View importance values"):
                st.dataframe(
                    importance_df.sort_values(
                        "Importance", ascending=False
                    ).round(5),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.warning(
                f"Could not calculate permutation importance: {e}"
            )
    else:
        st.warning(
            "The Machine failure target column is unavailable. "
            "Supervised permutation importance cannot be calculated."
        )

    # --------------------------------------------------
    # SECTION 3: SHAP EXPLANATIONS
    # --------------------------------------------------

    st.divider()
    st.subheader("SHAP Model Explanations")

    try:
        import shap

        if len(X_explain) > 0:

            # Encode Type for SHAP calculations
            type_categories = sorted(
                X_explain["Type"].astype(str).unique()
            )

            type_to_code = {
                value: index
                for index, value in enumerate(type_categories)
            }

            def encode_shap_data(data):
                if isinstance(data, pd.DataFrame):
                    result = data[FEATURES].copy()
                else:
                    result = pd.DataFrame(
                        data,
                        columns=FEATURES,
                    )

                result["Type"] = (
                    result["Type"]
                    .astype(str)
                    .map(type_to_code)
                )

                return result.astype(float)

            def predict_shap_probability(values):
                if isinstance(values, pd.DataFrame):
                    batch = values[FEATURES].copy()
                else:
                    batch = pd.DataFrame(
                        values,
                        columns=FEATURES,
                    )

                type_codes = np.rint(
                    batch["Type"].to_numpy()
                ).astype(int)

                batch["Type"] = [
                    type_categories[code]
                    if 0 <= code < len(type_categories)
                    else type_categories[0]
                    for code in type_codes
                ]

                return rf_model.predict_proba(
                    batch[FEATURES]
                )[:, failure_class_index]

            # Background and explanation samples
            background = X_explain.sample(
                n=min(30, len(X_explain)),
                random_state=42,
            ).reset_index(drop=True)

            shap_sample = X_explain.sample(
                n=min(40, len(X_explain)),
                random_state=7,
            ).reset_index(drop=True)

            background_encoded = encode_shap_data(background)
            sample_encoded = encode_shap_data(shap_sample)

            # Create one explainer for global and local explanations
            explainer = shap.PermutationExplainer(
                predict_shap_probability,
                background_encoded,
                feature_names=FEATURES,
                seed=42,
            )

            with st.spinner("Calculating SHAP values..."):
                shap_values = explainer(
                    sample_encoded,
                    max_evals=2 * len(FEATURES) + 1,
                )

            shap_array = np.asarray(shap_values.values)

            if shap_array.ndim == 3:
                shap_array = shap_array[:, :, 0]

            # Global SHAP importance
            mean_abs_shap = np.abs(shap_array).mean(axis=0)

            shap_importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "Mean Absolute SHAP Value": mean_abs_shap,
            }).sort_values(
                "Mean Absolute SHAP Value",
                ascending=True,
            )

            fig_shap_importance = px.bar(
                shap_importance_df,
                x="Mean Absolute SHAP Value",
                y="Feature",
                orientation="h",
                title="Global SHAP Importance",
            )

            fig_shap_importance.update_layout(height=420)

            st.plotly_chart(
                fig_shap_importance,
                use_container_width=True,
            )

            # SHAP summary plot
            st.markdown("**SHAP Summary Plot**")

            plt.figure(figsize=(10, 5))

            shap.summary_plot(
                shap_array,
                sample_encoded,
                feature_names=FEATURES,
                show=False,
                plot_type="dot",
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True,
            )
            plt.close("all")

            # --------------------------------------------------
            # SECTION 4: INDIVIDUAL RECORD EXPLANATION
            # --------------------------------------------------

            st.divider()
            st.subheader("Individual Prediction Explanation")

            available_indices = X_explain.index.tolist()

            if "UDI" in df.columns:
                selected_index = st.selectbox(
                    "Select a source record (UDI)",
                    options=available_indices,
                    format_func=lambda idx: str(
                        df.loc[idx, "UDI"]
                    ),
                    key="tab4_selected_record",
                )
            else:
                selected_index = st.selectbox(
                    "Select a source record",
                    options=available_indices,
                    key="tab4_selected_record",
                )

            selected_input = X_explain.loc[
                [selected_index], FEATURES
            ]

            # Use original input for the trained model
            selected_probability = float(
                rf_model.predict_proba(
                    selected_input
                )[0, failure_class_index]
            )

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Predicted Failure Probability",
                f"{selected_probability:.2%}",
            )

            result_col2.metric(
                "Recorded Machine Failure",
                (
                    str(int(df.loc[
                        selected_index, "Machine failure"
                    ]))
                    if "Machine failure" in df.columns
                    and pd.notna(
                        df.loc[selected_index, "Machine failure"]
                    )
                    else "Not available"
                ),
            )

            st.markdown("**Input Values**")

            st.dataframe(
                selected_input,
                use_container_width=True,
                hide_index=True,
            )

            # Encode selected record before explaining it
            selected_encoded = encode_shap_data(
                selected_input.reset_index(drop=True)
            )

            with st.spinner("Explaining selected record..."):
                local_values = explainer(
                    selected_encoded,
                    max_evals=2 * len(FEATURES) + 1,
                )

            local_array = np.asarray(local_values.values)

            if local_array.ndim == 3:
                local_array = local_array[:, :, 0]

            local_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP Value": local_array[0],
            })

            local_df = local_df.sort_values(
                "SHAP Value",
                ascending=True,
            )

            fig_local = px.bar(
                local_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                title="Feature Contributions for Selected Record",
                color="SHAP Value",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )

            fig_local.update_layout(
                height=400,
                coloraxis_showscale=False,
                xaxis_title="SHAP Value (probability scale)",
            )

            st.plotly_chart(
                fig_local,
                use_container_width=True,
            )

    except ImportError:
        st.warning(
            "SHAP is not installed. Run: python -m pip install shap"
        )

    except Exception as e:
        st.warning(f"SHAP analysis failed: {e}")


    # --------------------------------------------------
    # SECTION 5: SENSOR CORRELATION HEATMAP
    # --------------------------------------------------


    st.divider()
    st.subheader("Sensor Correlation Heatmap")

    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    available_numeric = [
        col for col in numeric_features
        if col in df.columns
    ]

    if len(available_numeric) >= 2:
        correlation_matrix = df[
            available_numeric
        ].corr(numeric_only=True)

        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Between Sensor Measurements",
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True,
        )

        st.caption(
            "Correlations range from -1 to +1. Correlation does not "
            "establish that one sensor measurement causes another "
            "or causes machine failure."
        )
    else:
        st.warning(
            "At least two numerical sensor columns are needed "
            "to calculate the correlation heatmap."
        )



# --------------------------------------------------
# TAB 5: MODEL PERFORMANCE
# --------------------------------------------------
with tab5:
    st.header("Model Performance")

    # Load the trained model
    rf_model, if_model, threshold = load_models()

    # Generate predictions on the saved test set
    y_true = np.asarray(y_test).ravel()
    y_pred = rf_model.predict(X_test)

    # Get the probability of machine failure
    probabilities = rf_model.predict_proba(X_test)
    classes = list(rf_model.classes_)

    failure_class_index = classes.index(1)
    y_probability = probabilities[:, failure_class_index]

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
    )

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_true, y_probability)
    pr_auc = average_precision_score(y_true, y_probability)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.2%}")
    c2.metric("Precision", f"{precision:.2%}")
    c3.metric("Recall", f"{recall:.2%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1 Score", f"{f1:.2%}")
    c5.metric("ROC AUC", f"{roc_auc:.3f}")
    c6.metric("PR AUC", f"{pr_auc:.3f}")

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Normal", "Predicted Failure"],
        y=["Actual Normal", "Actual Failure"],
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Count",
        },
        color_continuous_scale="Blues",
        aspect="auto",
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        "Rows represent actual classes; columns represent predicted classes."
    )

    st.divider()

    # Precision-recall curve
    st.subheader("Precision-Recall Curve")

    from sklearn.metrics import precision_recall_curve

    pr_precision, pr_recall, _ = precision_recall_curve(
        y_true, y_probability
    )

    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=pr_recall,
            y=pr_precision,
            mode="lines",
            name="Random Forest",
        )
    )
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    # ROC curve
    st.subheader("ROC Curve")

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_probability)

    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="Random Forest",
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash"),
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(fig_roc, use_container_width=True)
