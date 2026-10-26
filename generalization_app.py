"""Streamlit interface for evaluating generalization to unseen typhoon events.
Run with: streamlit run generalization_app.py
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from experiments.compare_qkn_approaches import run_comparison
from experiments.compare_qkn_kernel import split_data
st.set_page_config(page_title="QKN | Unseen Typhoon Evaluation", layout="wide")
st.title("QKN Generalization Lab")
st.caption("Quantum-Triad • Event-level evaluation on typhoons excluded from training")
st.markdown("An interactive workspace for matched Approach A, Approach B, and classical evaluation. Storms—not individual snapshots—must be separated between training and test.")
with st.sidebar:
    st.header("Dataset")
    source = st.radio("Dataset source", ["Repository CSV", "Upload event-aware CSV"])
    uploaded = None
    if source == "Upload event-aware CSV":
        uploaded = st.file_uploader("Upload CSV containing storm/event ID", type=["csv"])
        data_path = None
    else:
        data_path = st.text_input("CSV path", "data/processed/final_quantum_dataset.csv")
    st.divider()
    st.header("Evaluation setup")
    test_size = st.slider("Held-out storm fraction", 0.10, 0.40, 0.25, 0.05)
    max_samples = st.number_input("Maximum rows for simulator runtime (0 = all rows)", min_value=0, max_value=10000, value=250, step=50)
    layers = st.slider("Approach A circuit layers", 1, 4, 2)
    C = st.select_slider("SVM regularization C", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
if source == "Upload event-aware CSV":
    if uploaded is None:
        st.info("Upload the processed feature table exported from the updated original app.")
        st.stop()
    frame = pd.read_csv(uploaded)
    original_name = uploaded.name
else:
    path = Path(data_path)
    if not path.exists():
        st.error(f"Dataset not found: {path}. Run the original app's extraction and export phases first.")
        st.stop()
    frame = pd.read_csv(path)
    original_name = str(path)
tabs = st.tabs(["1 · Event audit", "2 · Matched benchmark", "3 · Protocol and interpretation"])
with tabs[0]:
    st.subheader("Event and label audit")
    st.write(f"Loaded **{len(frame):,} rows** and **{len(frame.columns)} columns** from `{original_name}`.")
    st.dataframe(frame.head(20), use_container_width=True)
    id_candidates = [c for c in frame.columns if c.lower() in {"storm_id", "seq_id", "event_id", "typhoon_id", "storm", "event"}]
    label_candidates = [c for c in frame.columns if c.lower() in {"failure_label", "label", "target", "y"}]
    group_col = st.selectbox("Typhoon/event identifier", options=list(frame.columns), index=(list(frame.columns).index(id_candidates[0]) if id_candidates else 0), key="group_col")
    label_col = st.selectbox("Prediction target", options=list(frame.columns), index=(list(frame.columns).index(label_candidates[0]) if label_candidates else 0), key="label_col")
    group_count = int(frame[group_col].nunique(dropna=True))
    confirm_event_id = st.checkbox(
        "I confirm this column uniquely identifies the complete source typhoon/event for every row.",
        value=False,
        help="Do not select a feature, snapshot number, or bus ID. Use storm_id/seq_id/event_id that is constant for all observations from one typhoon."
    )
    c1, c2 = st.columns(2)
    c1.metric("Distinct events", group_count)
    c2.metric("Target classes", int(frame[label_col].nunique(dropna=True)))
    if group_col == label_col:
        st.error("The event identifier and prediction target must be different columns.")
    elif group_count < 3:
        st.error("At least 3 distinct typhoon events are required for a basic train/test experiment; more are strongly recommended.")
    else:
        event_sizes = frame.groupby(group_col, dropna=False).size().rename("rows").reset_index()
        st.plotly_chart(px.bar(event_sizes, x=group_col, y="rows", title="Rows per event"), use_container_width=True)
        class_by_event = pd.crosstab(frame[group_col], frame[label_col], normalize="index")
        st.plotly_chart(px.imshow(class_by_event, aspect="auto", title="Label proportions by event"), use_container_width=True)
    st.warning("If the selected identifier is not a true storm/event identifier, this is not an unseen-typhoon test. Rows from the same typhoon must never appear in both training and test.")
with tabs[1]:
    st.subheader("Matched Approach A vs Approach B vs classical baseline")
    st.markdown("- **Approach A:** QKN-style Pauli-Z expectation features + RBF-SVC.\n- **Approach B:** quantum fidelity kernel + precomputed-kernel SVC.\n- **Classical control:** standardized input features + RBF-SVC.\n\nAll three use the same event-level partition. This compares quantum representations; it does not claim to run the full spatiotemporal GNN.")
    excluded = {group_col, label_col, "snapshot_id", "bus_id"}
    numeric_candidates = [c for c in frame.select_dtypes(include=np.number).columns if c not in excluded]
    default_features = [c for c in numeric_candidates if c.endswith("(rad)")]
    if len(default_features) < 2:
        default_features = numeric_candidates[:3]
    features = st.multiselect("Input features (one qubit per selected feature)", numeric_candidates, default=default_features[:3])
    st.caption(f"Selected features: {len(features)} → {len(features)} qubits.")
    can_run = confirm_event_id and group_col != label_col and group_count >= 3 and len(features) >= 2 and len(features) <= 8 and frame[label_col].nunique(dropna=True) == 2 and frame[group_col].notna().all()
    if len(features) > 8:
        st.warning("Select 8 or fewer features for this simulator-based benchmark.")
    if st.button("Run unseen-typhoon benchmark", type="primary", disabled=not can_run):
        work = frame.dropna(subset=features + [label_col, group_col]).copy()
        try:
            work[label_col] = work[label_col].astype(int)
        except (ValueError, TypeError):
            st.error("The target must be binary and encoded as 0/1.")
            st.stop()
        if set(work[label_col].unique()) != {0, 1}:
            st.error("The target must be binary and encoded as 0/1.")
            st.stop()
        if max_samples and len(work) > max_samples:
            # Keep every row from a selected event; never downsample individual bus rows.
            rng = np.random.RandomState(int(seed))
            event_ids = work[group_col].drop_duplicates().to_numpy()
            rng.shuffle(event_ids)
            selected_events = []
            selected_rows = 0
            for event_id in event_ids:
                event_rows = int((work[group_col] == event_id).sum())
                if selected_rows and selected_rows + event_rows > int(max_samples):
                    continue
                selected_events.append(event_id)
                selected_rows += event_rows
                if selected_rows >= int(max_samples):
                    break
            work = work[work[group_col].isin(selected_events)].copy()
        if work[group_col].nunique() < 3:
            st.error("The row budget left fewer than 3 complete events. Increase the row budget.")
            st.stop()
        if work[label_col].nunique() < 2:
            st.error("The selected complete events do not contain both target classes. Increase the row budget or select different events.")
            st.stop()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
            temp_path = tmp.name
            work.to_csv(tmp, index=False)
        try:
            with st.spinner("Running quantum circuits and evaluating held-out typhoon groups..."):
                results = run_comparison(data_path=temp_path, output_path="outputs/qkn_unseen_typhoon_comparison.json", features=features, label=label_col, test_size=float(test_size), random_state=int(seed), n_qubits=len(features), C=float(C), group_column=group_col, max_samples=0, layers=int(layers))
            from sklearn.model_selection import GroupShuffleSplit
            X, y, groups = work[features].to_numpy(dtype=float), work[label_col].to_numpy(), work[group_col].to_numpy()
            train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed)).split(X, y, groups))
            train_events, test_events = sorted(map(str, pd.unique(groups[train_idx]))), sorted(map(str, pd.unique(groups[test_idx])))
            if set(train_events) & set(test_events):
                st.error("Leakage check failed: an event appears in both partitions.")
                st.stop()
            st.session_state["last_qkn_generalization_result"] = results
            st.session_state["last_qkn_event_split"] = {"train_events": train_events, "test_events": test_events, "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx))}
        except Exception as exc:
            st.exception(exc)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    if "last_qkn_generalization_result" in st.session_state:
        result, split = st.session_state["last_qkn_generalization_result"], st.session_state["last_qkn_event_split"]
        st.success("Benchmark completed. Event-level separation check passed.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Training events", len(split["train_events"]))
        c2.metric("Held-out test events", len(split["test_events"]))
        c3.metric("Held-out test rows", split["test_rows"])
        st.write("**Training event IDs:**", ", ".join(split["train_events"]))
        st.write("**Held-out event IDs:**", ", ".join(split["test_events"]))
        metrics = pd.DataFrame([
            {"model": name, **{key: value for key, value in values.items() if key != "per_storm_metrics"}}
            for name, values in result["models"].items()
        ])
        st.subheader("Pooled held-out metrics")
        st.dataframe(metrics, use_container_width=True)
        per_storm_rows = []
        for model_name, values in result["models"].items():
            for storm_id, storm_metrics in values.get("per_storm_metrics", {}).items():
                per_storm_rows.append({"model": model_name, "storm_id": storm_id, **storm_metrics})
        if per_storm_rows:
            st.subheader("Metrics by held-out storm")
            per_storm = pd.DataFrame(per_storm_rows)
            st.dataframe(per_storm, use_container_width=True)
            st.download_button(
                "Download per-storm metrics CSV",
                per_storm.to_csv(index=False),
                file_name="qkn_per_storm_metrics.csv",
                mime="text/csv",
            )
        metric_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in metrics.columns]
        if metric_cols:
            st.plotly_chart(px.bar(metrics, x="model", y=metric_cols, barmode="group", title="Performance on held-out typhoon events"), use_container_width=True)
        st.download_button("Download JSON report", json.dumps(result, indent=2), file_name="qkn_unseen_typhoon_comparison.json", mime="application/json")
        st.download_button("Download metrics CSV", metrics.to_csv(index=False), file_name="qkn_unseen_typhoon_metrics.csv", mime="text/csv")
with tabs[2]:
    st.subheader("Recommended generalization protocol")
    st.markdown("1. Extract multiple complete typhoon tracks and preserve original `seq_id` as `storm_id`.\n2. Split by storm ID before model fitting: training storms, calibration/validation storms, and a final held-out test set.\n3. Fit feature scaling and feature selection on training storms only. Tune hyperparameters using training/validation storms only.\n4. Freeze the model and evaluate once on held-out storms. Report per-storm metrics as well as pooled metrics.\n5. Keep all snapshots and all 33 bus rows from a snapshot together; these are correlated observations, not independent events.\n6. For stronger evidence, repeat group holdout across seeds or use leave-one-storm-out cross-validation.")
    st.warning("The older processed CSV may not contain an event ID because earlier exports discarded storm identity. Re-run the updated original app's extraction/export phase to generate `storm_id`, `snapshot_id`, and `bus_id`. Do not substitute a random row split when these identifiers are missing.")
