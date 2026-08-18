from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from intelligent_cyber_assistant import run_intelligent_ids_from_dataframe
from intelligent_cyber_assistant import run_intelligent_ids_pipeline
from intelligent_cyber_assistant import run_live_network_pipeline
from live_data_adapter import build_live_dataframe


st.set_page_config(page_title="Intelligent Cybersecurity Assistant", layout="wide")
load_dotenv()


def resolve_default_data_path() -> Path:
    processed_dir = project_root / "data" / "processed"
    preferred = processed_dir / "multi_dataset_combined.csv"
    fallback = processed_dir / "clean_data.csv"
    return preferred if preferred.exists() else fallback


data_path = resolve_default_data_path()


def load_project_preview(max_rows: int = 5000) -> pd.DataFrame:
    return pd.read_csv(data_path, nrows=max_rows, low_memory=False)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%);
            color: white;
            padding: 1.2rem 1.4rem;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
        }
        .hero-subtitle {
            margin-top: 0.4rem;
            color: rgba(255,255,255,0.85);
            font-size: 0.98rem;
        }
        .metric-note {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: -0.25rem;
            margin-bottom: 0.5rem;
        }
        .soft-panel {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.9rem;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Intelligent Cybersecurity Assistant</div>
        <div class="hero-subtitle">Hybrid detection, correlated incidents, live monitoring, and explanation-ready outputs.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run Settings")
    st.caption("Upload a CSV, or use the built-in dataset. For Wireshark data, export to CSV first.")
    mode = st.selectbox("Data mode", options=["Historical CSV", "Real-time Network"], index=0)
    fast_mode = st.toggle("Fast mode (lower latency)", value=True)
    sample_size = st.slider("Sample size", min_value=2000, max_value=30000, value=4000, step=1000)
    realtime_window = st.slider("Real-time window rows", min_value=100, max_value=1000, value=300, step=50)
    auto_refresh = st.toggle("Auto-refresh live mode", value=False)
    refresh_seconds = st.slider("Refresh every (seconds)", min_value=2, max_value=30, value=5, step=1)
    threshold = st.slider("Detection threshold", min_value=0.20, max_value=0.90, value=0.50, step=0.01)
    use_llm = st.toggle("Enable LLM reasoning", value=True)
    llm_provider = st.selectbox("LLM provider", options=["auto", "openai", "gemini"], index=0)
    enable_honeybadger = st.toggle("Enable HoneyBadger defense layer", value=True)
    llm_model = st.text_input("LLM model", value="gpt-4o-mini")
    uploaded_dataset = st.file_uploader("Upload your own CSV dataset", type=["csv"])
    run_pipeline = st.button("Run Analysis", type="primary")
    clear_cache = st.button("Clear cached results")

    st.markdown("---")
    st.markdown("**Supported custom dataset format**")
    st.caption("Best case: CSV with a `label` column and network-flow style features. Wireshark PCAP files are not used directly here.")
    st.caption("If you export from Wireshark, convert the capture into CSV/flow features first.")
    st.caption(f"Default dataset: {data_path.name}")

if "live_history" not in st.session_state:
    st.session_state["live_history"] = []

if "run_result_cache" not in st.session_state:
    st.session_state["run_result_cache"] = {}

if clear_cache:
    st.session_state["run_result_cache"] = {}
    st.cache_data.clear()
    st.success("Cached results cleared.")


@st.cache_data(show_spinner=False)
def load_project_data() -> pd.DataFrame:
    return load_project_preview()


def validate_uploaded_dataset(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if "label" not in df.columns:
        issues.append("Missing required 'label' column. The model needs this column to train and evaluate.")

    if "attack_cat" not in df.columns:
        issues.append("Optional 'attack_cat' column not found. The attack-category chart will be skipped.")

    if "id" in df.columns:
        issues.append("'id' column detected. It will be ignored by the model, but it is safe to keep in the file.")

    if df.empty:
        issues.append("The uploaded file is empty.")

    return issues


def build_confusion_frame(y_true: list[int], y_pred: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame({"True": y_true, "Predicted": y_pred})
    confusion = pd.crosstab(frame["True"], frame["Predicted"])
    return confusion.reindex(index=[0, 1], columns=[0, 1], fill_value=0)


def build_threshold_sweep(y_true: list[int], scores: list[float]) -> pd.DataFrame:
    thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    y_true_arr = np.asarray(y_true)
    score_arr = np.asarray(scores)

    for threshold_value in thresholds:
        predicted = (score_arr >= threshold_value).astype(int)
        tp = int(np.sum((y_true_arr == 1) & (predicted == 1)))
        tn = int(np.sum((y_true_arr == 0) & (predicted == 0)))
        fp = int(np.sum((y_true_arr == 0) & (predicted == 1)))
        fn = int(np.sum((y_true_arr == 1) & (predicted == 0)))

        accuracy = (tp + tn) / max(1, len(y_true_arr))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        rows.append(
            {
                "threshold": round(float(threshold_value), 2),
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "alerts": int(np.sum(predicted)),
            }
        )

    return pd.DataFrame(rows)


def render_project_visuals(df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Project Visual Gallery")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Overview", "Distributions", "Correlation"])

    with chart_tab1:
        overview_col1, overview_col2, overview_col3 = st.columns(3)

        overview_col1.metric("Rows", value=f"{df.shape[0]:,}")
        overview_col2.metric("Columns", value=f"{df.shape[1]:,}")
        overview_col3.metric("Numeric Features", value=f"{len(df.select_dtypes(include='number').columns):,}")

        o1, o2 = st.columns(2)
        with o1:
            st.markdown("**Label Distribution**")
            if "label" in df.columns:
                label_counts = df["label"].value_counts().sort_index()
                st.bar_chart(label_counts)
            else:
                st.info("Label column not found in the dataset.")
        with o2:
            st.markdown("**Missing Values by Column**")
            missing_counts = df.isna().sum()
            missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
            if missing_counts.empty:
                st.success("No missing values found in the dataset.")
            else:
                st.bar_chart(missing_counts)

    with chart_tab2:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Top 10 Attack Categories**")
            if "attack_cat" in df.columns:
                attack_counts = df["attack_cat"].value_counts().head(10)
                st.bar_chart(attack_counts)
            else:
                st.info("Attack category column not found in the dataset.")
        with d2:
            st.markdown("**Top 12 Numeric Feature Means**")
            numeric_df = df.select_dtypes(include="number")
            if not numeric_df.empty:
                mean_df = numeric_df.mean().abs().sort_values(ascending=False).head(12)
                st.bar_chart(mean_df)
            else:
                st.info("No numeric columns available for summary charts.")

    with chart_tab3:
        st.markdown("**Top Numeric Feature Correlations**")
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr(numeric_only=True)
            top_features = corr.abs().sum().sort_values(ascending=False).head(12).index
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(corr.loc[top_features, top_features], cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")


def _risk_from_history(hist_df: pd.DataFrame) -> tuple[str, str, str]:
    if hist_df.empty:
        return ("LOW", "#1f9d55", "No live events yet")

    last = hist_df.iloc[-1]
    sev = str(last.get("top_severity", "none")).lower()
    events = int(last.get("correlated_events", 0))

    if sev == "high" or events >= 6:
        return ("HIGH", "#c62828", "Immediate analyst attention required")
    if sev == "medium" or events >= 2:
        return ("MEDIUM", "#ef6c00", "Elevated activity - monitor closely")
    return ("LOW", "#2e7d32", "Stable traffic profile")

auto_run = mode == "Real-time Network" and auto_refresh
should_run = run_pipeline or auto_run

if uploaded_dataset is not None:
    current_df = pd.read_csv(uploaded_dataset)
    upload_issues = validate_uploaded_dataset(current_df)
    upload_blocked = any(
        issue.startswith("Missing required") or "empty" in issue.lower() for issue in upload_issues
    )
    st.success(f"Uploaded dataset loaded: {current_df.shape[0]} rows x {current_df.shape[1]} columns")
    st.caption("Required for training: `label` column. Optional: `attack_cat`, `id`.")
    if upload_issues:
        with st.expander("Upload checks", expanded=True):
            for issue in upload_issues:
                if "Missing required" in issue or "empty" in issue:
                    st.error(issue)
                else:
                    st.info(issue)
    st.dataframe(current_df.head(10), use_container_width=True)
    target_candidates = list(current_df.columns)
    if "label" in target_candidates:
        default_target_index = target_candidates.index("label")
    elif "attack_cat" in target_candidates:
        default_target_index = target_candidates.index("attack_cat")
    else:
        default_target_index = 0
    target_col = st.selectbox(
        "Target / label column",
        options=target_candidates,
        index=default_target_index,
        help="Choose the column that contains the class you want the model to learn.",
    )
else:
    current_df = load_project_data()
    upload_blocked = False
    target_col = "label"

with st.expander("Dataset Visual Gallery (Heavy)", expanded=False):
    render_project_visuals(current_df)

if uploaded_dataset is not None and upload_blocked and mode == "Historical CSV":
    st.stop()

if should_run:
    cache_store = st.session_state.get("run_result_cache")
    if not isinstance(cache_store, dict):
        cache_store = {}
        st.session_state["run_result_cache"] = cache_store

    if uploaded_dataset is not None:
        dataset_tag = f"upload:{uploaded_dataset.name}:{uploaded_dataset.size}:{current_df.shape[0]}:{current_df.shape[1]}"
    else:
        dataset_tag = f"default:{data_path.name}:{int(data_path.stat().st_mtime)}"

    run_key = (
        mode,
        dataset_tag,
        sample_size,
        realtime_window,
        fast_mode,
        threshold,
        use_llm,
        llm_provider,
        llm_model,
        enable_honeybadger,
        target_col,
    )

    result = None
    if not auto_run:
        result = cache_store.get(run_key)
        if result is not None:
            st.caption("Using cached analysis result for current settings.")

    if result is None:
        with st.spinner("Running hybrid detection pipeline..."):
            if mode == "Historical CSV" and uploaded_dataset is not None:
                result = run_intelligent_ids_from_dataframe(
                    df=current_df,
                    sample_size=sample_size,
                    target_col=target_col,
                    fast_mode=fast_mode,
                    use_llm=use_llm,
                    llm_model=llm_model,
                    llm_provider=llm_provider,
                    enable_honeybadger=enable_honeybadger,
                    threshold=threshold,
                )
            elif mode == "Historical CSV":
                result = run_intelligent_ids_pipeline(
                    csv_path=str(data_path),
                    sample_size=sample_size,
                    target_col=target_col,
                    fast_mode=fast_mode,
                    use_llm=use_llm,
                    llm_model=llm_model,
                    llm_provider=llm_provider,
                    enable_honeybadger=enable_honeybadger,
                    threshold=threshold,
                )
            else:
                live_df = build_live_dataframe(
                    schema_path=str(data_path),
                    window_rows=realtime_window,
                )
                result = run_live_network_pipeline(
                    live_df=live_df,
                    baseline_csv_path=str(data_path),
                    baseline_sample_size=sample_size,
                    fast_mode=fast_mode,
                    use_llm=use_llm,
                    llm_model=llm_model,
                    llm_provider=llm_provider,
                    enable_honeybadger=enable_honeybadger,
                    threshold=min(threshold, 0.40),
                )

        if not auto_run:
            cache_store[run_key] = result
            while len(cache_store) > 4:
                oldest_key = next(iter(cache_store))
                del cache_store[oldest_key]

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_col1.metric("Samples Tested", value=result["total_samples_tested"])
    metrics_col2.metric("Attack Predictions", value=result["attack_predictions"])
    metrics_col3.metric("Correlated Events", value=result["correlated_events"])
    metrics_col4.metric("ROC-AUC", value=result["metrics"]["roc_auc"])
    st.caption(f"HoneyBadger enabled: {result['honeybadger_enabled']}")
    st.caption(f"Data mode: {mode}")
    if auto_run:
        st.caption(f"Live auto-refresh active: every {refresh_seconds}s")

    if mode == "Real-time Network":
        events_df_for_hist = pd.DataFrame(result["events_table"])
        if events_df_for_hist.empty:
            top_severity = "none"
            top_event_type = "none"
        else:
            top_severity = str(events_df_for_hist["severity"].value_counts().index[0])
            top_event_type = str(events_df_for_hist["event_type"].value_counts().index[0])

        st.session_state["live_history"].append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "attack_predictions": int(result["attack_predictions"]),
                "correlated_events": int(result["correlated_events"]),
                "top_severity": top_severity,
                "top_event_type": top_event_type,
            }
        )
        st.session_state["live_history"] = st.session_state["live_history"][-10:]

    eval1, eval2, eval3, eval4 = st.columns(4)
    eval1.metric("Accuracy", value=result["metrics"]["accuracy"])
    eval2.metric("Precision", value=result["metrics"]["precision"])
    eval3.metric("Recall", value=result["metrics"]["recall"])
    eval4.metric("F1 Score", value=result["metrics"]["f1_score"])

    st.markdown('<div class="metric-note">Tip: use the chart tabs above to compare your dataset before running the model.</div>', unsafe_allow_html=True)

    with st.expander("Model Lab (Heavy)", expanded=False):
        st.subheader("Model Lab")

        lab_tab1, lab_tab2, lab_tab3 = st.tabs(["Confusion Matrix", "Threshold Sensitivity", "Confidence View"])

        with lab_tab1:
            if result.get("y_true") and result.get("y_pred"):
                confusion_df = build_confusion_frame(result["y_true"], result["y_pred"])
                fig, ax = plt.subplots(figsize=(5.8, 4.8))
                sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Confusion matrix is available only for historical runs with ground-truth labels.")

        with lab_tab2:
            if result.get("y_true") and result.get("confidence_scores"):
                sweep_df = build_threshold_sweep(result["y_true"], result["confidence_scores"])
                left_col, right_col = st.columns(2)
                with left_col:
                    st.line_chart(sweep_df.set_index("threshold")[["accuracy", "precision", "recall", "f1_score"]])
                with right_col:
                    st.line_chart(sweep_df.set_index("threshold")["alerts"])
                st.dataframe(sweep_df, use_container_width=True)
            else:
                st.info("Threshold sensitivity analysis is available only for historical runs.")

        with lab_tab3:
            if result.get("confidence_scores"):
                conf_df = pd.DataFrame({"confidence": result["confidence_scores"]})
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Confidence Distribution**")
                    fig, ax = plt.subplots(figsize=(7.0, 4.5))
                    sns.histplot(conf_df["confidence"], bins=20, kde=True, ax=ax, color="#0f766e")
                    ax.set_xlabel("Confidence")
                    ax.set_ylabel("Count")
                    st.pyplot(fig, clear_figure=True)
                with c2:
                    st.markdown("**Top Confidence Samples**")
                    conf_preview = conf_df.sort_values("confidence", ascending=False).head(10)
                    st.dataframe(conf_preview, use_container_width=True)
            else:
                st.info("Confidence profiling is available only after running an analysis.")

    with st.expander("Project Graphs (Heavy)", expanded=False):
        st.subheader("Project Graphs")

        visual_col1, visual_col2 = st.columns(2)

        with visual_col1:
            if result.get("model_accuracy"):
                model_accuracy_df = pd.DataFrame(result["model_accuracy"]).set_index("model")
                st.markdown("**Training vs Testing Accuracy**")
                st.line_chart(model_accuracy_df[["train_accuracy", "test_accuracy"]])
            else:
                st.info("Model accuracy comparison is not available for this run.")

        with visual_col2:
            roc_points = result["metrics"].get("roc_curve_points", [])
            if roc_points:
                roc_df = pd.DataFrame(roc_points)
                st.markdown("**ROC Curve**")
                st.line_chart(roc_df.set_index("fpr")["tpr"])
            else:
                st.info("ROC curve data is not available for this run.")

        if result.get("events_table"):
            events_df_viz = pd.DataFrame(result["events_table"])
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.markdown("**Correlated Events by Severity**")
                severity_df = events_df_viz["severity"].value_counts().sort_index()
                st.bar_chart(severity_df)

            with viz_col2:
                st.markdown("**Correlated Events by Type**")
                event_type_df = events_df_viz["event_type"].value_counts().sort_values(ascending=False)
                st.bar_chart(event_type_df)

    if mode == "Real-time Network" and st.session_state["live_history"]:
        st.markdown("---")
        st.subheader("Live Cycle History (Last 10)")
        hist_df = pd.DataFrame(st.session_state["live_history"])

        risk_label, risk_color, risk_msg = _risk_from_history(hist_df)
        st.markdown(
            f"""
            <div style='display:flex;gap:12px;align-items:center;margin-bottom:8px;'>
              <span style='background:{risk_color};color:white;padding:6px 14px;border-radius:999px;font-weight:700;'>SOC RISK: {risk_label}</span>
              <span style='color:#475569;font-weight:600;'>{risk_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        severity_counts = hist_df["top_severity"].value_counts().to_dict()
        b1, b2, b3 = st.columns(3)
        b1.markdown(
            f"<span style='background:#c62828;color:white;padding:4px 10px;border-radius:999px;'>High cycles: {severity_counts.get('High', 0)}</span>",
            unsafe_allow_html=True,
        )
        b2.markdown(
            f"<span style='background:#ef6c00;color:white;padding:4px 10px;border-radius:999px;'>Medium cycles: {severity_counts.get('Medium', 0)}</span>",
            unsafe_allow_html=True,
        )
        b3.markdown(
            f"<span style='background:#2e7d32;color:white;padding:4px 10px;border-radius:999px;'>Low/None cycles: {severity_counts.get('Low', 0) + severity_counts.get('none', 0)}</span>",
            unsafe_allow_html=True,
        )

        st.dataframe(hist_df, use_container_width=True)
        st.line_chart(hist_df.set_index("timestamp")[["attack_predictions", "correlated_events"]])

    st.markdown("---")
    st.subheader("Detected Incident")
    outputs = result["final_outputs"]
    if outputs:
        incident = outputs[0]
        st.write(f"Type: {incident['attack_type']}")
        st.write(f"Source IP: {incident.get('source_ip', 'unknown-source')}")
        st.write(f"Severity: {incident['severity']}")

        st.subheader("Model Output")
        st.write(f"Classification: {incident.get('model_classification', 'Attack')}")
        st.write(f"Confidence Score: {incident['confidence']}")
        st.write(f"Anomaly Score: {incident.get('anomaly_score', 'n/a')}")

        st.subheader("Explanation")
        st.write(incident["reason"])

        st.subheader("Recommended Action")
        st.write(incident["recommended_action"])
    else:
        st.info("No incidents were correlated in this run.")

    st.subheader("Correlated Attack Events")
    events_df = pd.DataFrame(result["events_table"])

    if events_df.empty:
        st.info("No correlated attack events found in this run.")
    else:
        severity_filter = st.multiselect(
            "Filter severity",
            options=sorted(events_df["severity"].unique().tolist()),
            default=sorted(events_df["severity"].unique().tolist()),
        )
        type_filter = st.multiselect(
            "Filter event type",
            options=sorted(events_df["event_type"].unique().tolist()),
            default=sorted(events_df["event_type"].unique().tolist()),
        )

        filtered_df = events_df[
            events_df["severity"].isin(severity_filter)
            & events_df["event_type"].isin(type_filter)
        ]
        st.dataframe(filtered_df, use_container_width=True)

    st.subheader("LLM/Reasoning Outputs")
    if not outputs:
        st.warning("No reasoning output available.")
    else:
        for item in outputs:
            with st.container(border=True):
                st.markdown(f"### {item['event_id']}")
                c1, c2, c3 = st.columns(3)
                c1.write(f"Attack type: {item['attack_type']}")
                c2.write(f"Severity: {item['severity']}")
                c3.write(f"LLM used: {item['llm_used']}")
                st.write(f"Confidence: {item['confidence']}")
                st.write(f"Reason: {item['reason']}")
                st.write(f"Recommended action: {item['recommended_action']}")
                if result["honeybadger_enabled"]:
                    st.write(f"HoneyBadger mode: {item.get('honeybadger_mode', 'n/a')}")
                    st.write(f"HoneyBadger target: {item.get('honeybadger_target', 'n/a')}")
                    st.write(f"HoneyBadger objective: {item.get('honeybadger_objective', 'n/a')}")
                    st.write(f"HoneyBadger action: {item.get('honeybadger_action', 'n/a')}")

    st.subheader("Export Incident Report")
    incident_json = json.dumps(result, indent=2)
    st.download_button(
        "Download JSON report",
        data=incident_json,
        file_name="incident_report.json",
        mime="application/json",
    )

    if result["events_table"]:
        events_csv = pd.DataFrame(result["events_table"]).to_csv(index=False)
        st.download_button(
            "Download events CSV",
            data=events_csv,
            file_name="correlated_events.csv",
            mime="text/csv",
        )

    if auto_run:
        time.sleep(refresh_seconds)
        st.rerun()
else:
    st.info("Choose settings and click Run Analysis. For live streaming, enable Real-time Network + Auto-refresh.")
