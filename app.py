import streamlit as st
import numpy as np
import pandas as pd
import dowhy
import dowhy.gcm as gcm
from dowhy import CausalModel
import statsmodels.api as sm
import warnings
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="SC CausalML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .vertical-divider {
            border-left: 2px solid #e0e0e0;
            height: 100%;
            padding-left: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Interactive Causal Analysis App")


# --------------------------------------------------------
# SIDEBAR NAVIGATION (BUTTON-STYLE WITH SYMBOLS)
# --------------------------------------------------------

st.sidebar.header("Navigation")
st.sidebar.write("---")

# Initialize page in session state
if "page" not in st.session_state:
    st.session_state.page = "Root Cause Analysis"

def nav_button(label, page_name):
    is_active = (st.session_state.page == page_name)
    button_style = (
        f"background-color: #e8e8e8; font-weight: 600;" if is_active else "background-color: transparent;"
    )
    html = f"""
    <style>
        .nav-button {{
            width: 100%;
            text-align: left;
            border: none;
            background: none;
            padding: 8px 4px;
            font-size: 15px;
        }}
        .nav-button:hover {{
            background-color: #f2f2f2;
        }}
    </style>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)
    clicked = st.sidebar.button(f"{label}", key=page_name)
    if clicked:
        st.session_state.page = page_name

# Menu items
nav_button("Root Cause Analysis", "Root Cause Analysis")
nav_button("Intervention Analysis", "Intervention Analysis")
nav_button("Anomaly Attribution", "Anomaly Attribution")


page = st.session_state.page


# --------------------------------------------------------
# ROOT CAUSE ANALYSIS PAGE  (Your original code)
# --------------------------------------------------------
if page == "Root Cause Analysis":

    DAG1_edges = [
        ("SameNode_Production", "SameNode_DaysOfCover"),
        ("SameNode_ForecastAccuracy", "SameNode_DaysOfCover"),
        ("SameNode_Procurement", "SameNode_DaysOfCover"),
        ("SameNode_ForecastAccuracy", "SameNode_Procurement"),
        ("SameNode_ForecastAccuracy", "SameNode_Production"),
        ("SameNode_Procurement", "SameNode_Production"),
    ]

    DAG2_edges = [
        ("SameNode_Production", "SameNode_DaysOfCover"),
        ("SameNode_ForecastAccuracy", "SameNode_DaysOfCover"),
        ("SameNode_Procurement", "SameNode_DaysOfCover"),
        ("SameNode_ForecastAccuracy", "SameNode_Procurement"),
        ("SameNode_ForecastAccuracy", "SameNode_Production"),
    ]

    st.sidebar.header("Configuration")
    st.sidebar.subheader("Select Variables")

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    columns = []
    ate_placeholder = None

    if "estimate_value" not in st.session_state:
        st.session_state.estimate_value = None

    if "tornado_values" not in st.session_state:
        st.session_state.tornado_values = {
            "SameNode_Procurement": 0,
            "SameNode_Production": 0,
            "SameNode_ForecastAccuracy": 0
        }

    if uploaded_file is not None:
        df_causal = pd.read_csv(uploaded_file)
        columns = list(df_causal.columns)

    if columns:
        treatment_var = st.sidebar.selectbox(
            "Treatment Variable",
            options=columns,
            index=columns.index("SameNode_Procurement") if "SameNode_Procurement" in columns else 0
        )

        outcome_var = st.sidebar.selectbox(
            "Outcome Variable",
            options=columns,
            index=columns.index("SameNode_DaysOfCover") if "SameNode_DaysOfCover" in columns else 0
        )

        st.sidebar.divider()
        st.sidebar.info(f"**Treatment:** {treatment_var}\n\n**Outcome:** {outcome_var}")

        if st.session_state.estimate_value is not None:
            st.sidebar.divider()
            st.sidebar.metric("Estimated Causal Effect",
                              f"{st.session_state.estimate_value:.4f}")
    else:
        st.sidebar.warning("⬆️ Upload a CSV file to configure variables")
        ate_placeholder = None

    if ate_placeholder is None and not columns:
        st.sidebar.metric("Estimated ATE", "Upload file first")

    if uploaded_file is not None:
        tab1, tab2 = st.tabs(["DAG 1", "DAG 2"])
    else:
        tab1, tab2 = st.tabs(["DAG 1", "DAG 2"])

    def run_causal_analysis(dag_edges, dag_name, dag_image, df, treatment, outcome, run_analysis):

        G = nx.DiGraph()
        G.add_edges_from(dag_edges)
        causal_graph = G

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Causal Graph Diagram")
            st.image(dag_image, caption=f"Causal Relationships ({dag_name})", width=600)

        with col2:
            st.markdown('<div class="vertical-divider">', unsafe_allow_html=True)
            st.subheader("Data")
            st.write(f"Shape: {df.shape}")
            st.dataframe(df, width='stretch', height=400)
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        if run_analysis:
            try:
                with st.spinner("Running causal analysis..."):
                    np.random.seed(42)

                    df = df.drop(
                        ['Item.[Item]', 'Location.[Location]', 'Time.[Week]'],
                        axis=1
                    )

                    causal_model = gcm.StructuralCausalModel(causal_graph)
                    gcm.auto.assign_causal_mechanisms(causal_model, df)
                    gcm.fit(causal_model, df)

                    # --- Dynamic interventions based on selected treatment variable ---
                    alternative = {treatment: (lambda arr: arr + 1)}
                    reference = {treatment: (lambda arr: arr)}


                    estimated_value = gcm.average_causal_effect(
                        causal_model,
                        'SameNode_DaysOfCover',
                        interventions_alternative=alternative,
                        interventions_reference=reference,
                        num_samples_to_draw=10000
                    )

                    st.success("Analysis Complete!")
                    st.session_state.estimate_value = estimated_value

                    # --------------------------
                    # Compute Tornado Chart Values
                    # --------------------------
                    drivers = [
                        "SameNode_Procurement",
                        "SameNode_Production",
                        "SameNode_ForecastAccuracy"
                    ]

                    tornado_result = {}

                    for driver in drivers:
                        try:
                            alt = {driver: (lambda arr: arr + 1)}
                            ref = {driver: (lambda arr: arr)}

                            effect = gcm.average_causal_effect(
                                causal_model,
                                outcome,
                                interventions_alternative=alt,
                                interventions_reference=ref,
                                num_samples_to_draw=4000
                            )
                            tornado_result[driver] = float(effect)
                        except Exception:
                            tornado_result[driver] = 0.0

                    st.session_state.tornado_values = tornado_result

                    st.rerun()

            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")

    if uploaded_file is not None:
        with tab1:
            run_dag1 = st.button("Run Causal Analysis", key="run_dag1")
            run_causal_analysis(DAG1_edges, "DAG1", "casual_graph_new.png",
                                df_causal, treatment_var, outcome_var, run_dag1)
            # ====== ADD TORNADO CHART HERE ======
            # Tornado Chart
            plt.rcParams.update({'font.size': 7})
            st.subheader("Impact of Drivers on Outcome")

            tornado_data = st.session_state.tornado_values

            df_tornado = pd.DataFrame({
                "Driver": list(tornado_data.keys()),
                "Effect": list(tornado_data.values())
            }).sort_values(by="Effect", ascending=True)

            fig, ax = plt.subplots(figsize=(3, 2), dpi=90)

            bars = ax.barh(df_tornado["Driver"], df_tornado["Effect"])

            # Add numeric labels to each bar
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + (0.02 * np.sign(width)),  # slight offset left/right
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}",
                    va='center',
                    ha='left' if width >= 0 else 'right',
                    fontsize=9
                )

            ax.set_xlabel("Causal Impact on Outcome (ATE)")
            ax.set_ylabel("Driver")
            ax.set_title("Driver Sensitivity")

            #plt.tight_layout()
            #fig, ax = plt.subplots(figsize=(4.5, 2.8))
            st.pyplot(fig,width='content')

        with tab2:
            run_dag2 = st.button("Run Causal Analysis", key="run_dag2")
            run_causal_analysis(DAG2_edges, "DAG2", "causal_graph_dag2_new.png",
                                df_causal, treatment_var, outcome_var, run_dag2)
            # ====== ADD TORNADO CHART HERE ======
            # Tornado Chart
            plt.rcParams.update({'font.size': 7})
            st.subheader("Impact of Drivers on Outcome")

            tornado_data = st.session_state.tornado_values

            df_tornado = pd.DataFrame({
                "Driver": list(tornado_data.keys()),
                "Effect": list(tornado_data.values())
            }).sort_values(by="Effect", ascending=True)

            fig, ax = plt.subplots(figsize=(3, 2), dpi=90)

            bars = ax.barh(df_tornado["Driver"], df_tornado["Effect"])

            # Add numeric labels to each bar
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + (0.02 * np.sign(width)),  # slight offset left/right
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}",
                    va='center',
                    ha='left' if width >= 0 else 'right',
                    fontsize=9
                )

            ax.set_xlabel("Causal Impact on Outcome (ATE)")
            ax.set_ylabel("Driver")
            ax.set_title("Driver Sensitivity")

            #plt.tight_layout()
            st.pyplot(fig,width='content')


elif page == "Intervention Analysis":
    st.header("Intervention Analysis")
    st.write("Upload data, choose a driver, and simulate an intervention.")

    # --- File upload ---
    data_file = st.file_uploader("Upload your training (normal) CSV", type=["csv"], key="intervention_train")

    if data_file is not None:
        df_int = pd.read_csv(data_file)

        # Drop unused columns safely
        for col in ['Item.[Item]', 'Location.[Location]', 'Time.[Week]']:
            if col in df_int.columns:
                df_int = df_int.drop(columns=[col])

        columns = list(df_int.columns)

        st.subheader("Intervention Configuration")

        colA, colB = st.columns(2)

        with colA:
            treatment_sel = st.selectbox("Select Treatment Variable", columns)
        with colB:
            outcome_sel = st.selectbox("Select Outcome Variable", columns)

        colC, colD = st.columns(2)
        with colC:
            intervention_type = st.radio(
                "Intervention Type",
                ["Shift value (x + Δ)", "Set to constant"],
                horizontal=True
            )
        with colD:
            intervention_value = st.number_input("Value (Δ or constant)", value=1.0)

        st.divider()
        run_intervention = st.button("Run Intervention Simulation")

        if run_intervention:
            try:
                st.info("Fitting causal model and running intervention...")

                # ----- Build DAG (same as RCA) -----
                intervention_edges = [
                    ("SameNode_Production", "SameNode_DaysOfCover"),
                    ("SameNode_ForecastAccuracy", "SameNode_DaysOfCover"),
                    ("SameNode_Procurement", "SameNode_DaysOfCover"),
                    ("SameNode_ForecastAccuracy", "SameNode_Procurement"),
                    ("SameNode_ForecastAccuracy", "SameNode_Production"),
                    ("SameNode_Procurement", "SameNode_Production"),
                ]

                G_int = nx.DiGraph()
                G_int.add_edges_from(intervention_edges)

                causal_model_int = gcm.StructuralCausalModel(G_int)
                gcm.auto.assign_causal_mechanisms(causal_model_int, df_int)
                gcm.fit(causal_model_int, df_int)

                # ----- Build intervention dict -----
                if intervention_type == "Shift value (x + Δ)":
                    alt = {treatment_sel: lambda arr: arr + intervention_value}
                    ref = {treatment_sel: lambda arr: arr}
                else:
                    # Hard intervention
                    const_val = intervention_value
                    alt = {treatment_sel: lambda arr: np.full_like(arr, const_val)}
                    ref = {treatment_sel: lambda arr: arr}

                # ----- Compute ACE -----
                ace_val = gcm.average_causal_effect(
                    causal_model_int,
                    outcome_sel,
                    interventions_alternative=alt,
                    interventions_reference=ref,
                    num_samples_to_draw=8000
                )

                st.success("Intervention simulation complete!")

                # ----- Show Results -----
                st.metric(
                    label=f"Estimated Impact on {outcome_sel}",
                    value=f"{ace_val:.4f}"
                )

            except Exception as e:
                st.error(f"Error during intervention analysis: {e}")

    else:
        st.info("Please upload the training dataset to begin.")



# --------------------------------------------------------
# ANOMALY ATTRIBUTION PAGE (BOOTSTRAPPED, shows dowhy.utils.bar_plot)
# --------------------------------------------------------
elif page == "Anomaly Attribution":
    st.header("Anomaly Attribution")
    st.write("Upload a training (normal) CSV and an anomaly CSV")

    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload training set (normal data)", type=["csv"], key="anom_train")
    with col2:
        anomaly_file = st.file_uploader("Upload anomaly set (anomalous samples)", type=["csv"], key="anom_samples")

    run_attrib = st.button("Run Attribution", key="run_anom_boot")

    if run_attrib:
        if (train_file is None) or (anomaly_file is None):
            st.error("Please upload both training and anomaly CSV files before running.")
        else:
            try:
                df = pd.read_csv(train_file)
                df_anomaly = pd.read_csv(anomaly_file)

                # Safe drop if present (same as elsewhere)
                for col in ['Item.[Item]', 'Location.[Location]', 'Time.[Week]']:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                    if col in df_anomaly.columns:
                        df_anomaly = df_anomaly.drop(columns=[col])

                st.info("Fitting causal model and computing attribution (may take time)...")

                # Build DAG edges (same as Root Cause)
                causal_edges = [
                    ("SameNode_Production", "SameNode_DaysOfCover"),
                    ("SameNode_ForecastAccuracy", "SameNode_DaysOfCover"),
                    ("SameNode_Procurement", "SameNode_DaysOfCover"),
                    ("SameNode_ForecastAccuracy", "SameNode_Procurement"),
                    ("SameNode_ForecastAccuracy", "SameNode_Production"),
                    ("SameNode_Procurement", "SameNode_Production"),
                ]

                G = nx.DiGraph()
                G.add_edges_from(causal_edges)

                causal_model = gcm.StructuralCausalModel(G)
                gcm.auto.assign_causal_mechanisms(causal_model, df)
                gcm.fit(causal_model, df)

                # ---- YOUR EXACT BOOTSTRAP + bar_plot CALL ----
                with st.spinner("Computing attribution + confidence intervals..."):
                    median_attribs, uncertainty_attribs = gcm.confidence_intervals(
                        gcm.fit_and_compute(
                            gcm.attribute_anomalies,
                            causal_model,
                            df,
                            target_node='SameNode_DaysOfCover',
                            anomaly_samples=df_anomaly
                        ),
                        num_bootstrap_resamples=10
                    )

                st.success("Attribution complete.")

                # Use dowhy's bar_plot to draw the chart (it draws on current matplotlib figure)
                from dowhy.utils import bar_plot
                # Clear current figure (defensive)
                plt.close('all')
                bar_plot(median_attribs, uncertainty_attribs, 'Attribution Score')

                # Grab the current matplotlib figure and render in Streamlit
                fig = plt.gcf()
                fig.set_size_inches(4, 2.5)
                st.pyplot(fig, use_container_width=False)

                # Show raw outputs for inspection
                with st.expander("Show raw results (median, uncertainty)"):
                    st.write("Median attribution (per feature):")
                    st.write(median_attribs)
                    st.write("Uncertainty (per feature):")
                    st.write(uncertainty_attribs)

            except Exception as e:
                st.error(f"Error during"
                         f""
                         f" attribution: {e}")




