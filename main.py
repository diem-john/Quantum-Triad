import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import copy
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import LineString
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pennylane as qml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Custom Modules
from src.mapping import ChiayiMicrogridMapper, QuantumWalkIslandingMapper
from src.utils import haversine, rankine_vortex, vulnerability_curve
from src.qkn import QuantumSpatiotemporalGNN
from src.qkn_backend import create_qkn
from src.qcp import QuantumConformalPredictor


def label_distribution_frame(label_sets):
    """Return auditable binary class counts without modifying the labels."""
    rows = []
    for name, labels in label_sets.items():
        values = np.asarray(labels).reshape(-1)
        total = int(values.size)
        safe = int(np.sum(values == 0))
        failure = int(np.sum(values == 1))
        rows.append({
            "Dataset": name,
            "Total": total,
            "Safe (0)": safe,
            "Failure (1)": failure,
            "Safe (%)": round(100.0 * safe / total, 2) if total else 0.0,
            "Failure (%)": round(100.0 * failure / total, 2) if total else 0.0,
            "Other labels": int(np.sum(~np.isin(values, [0, 1]))),
        })
    return pd.DataFrame(rows)



def stratified_storm_group_split(storm_ids, y_tensor, fractions=(0.60, 0.15, 0.10, 0.15), random_state=42, trials=12000):
    """Allocate whole storms using randomized multi-start assignment and pair-swap refinement.

    The objective balances total snapshots, positive bus labels, and the number of
    failure-containing storms against each partition's target. Storms are never split.
    """
    storm_ids = np.asarray(storm_ids).astype(str)
    labels = np.asarray(y_tensor)
    names = ("train", "validation", "calibration", "test")
    fractions = np.asarray(fractions, dtype=float)
    fractions = fractions / fractions.sum()
    groups = np.unique(storm_ids)
    if len(groups) < len(names):
        raise ValueError(f"Need at least {len(names)} distinct storms; found {len(groups)}.")

    stats = {}
    for group in groups:
        mask = storm_ids == group
        positive_count = int(np.sum(labels[mask] == 1))
        stats[group] = np.array([int(mask.sum()), positive_count, int(positive_count > 0)], dtype=float)
    positive_groups = [g for g in groups if stats[g][2] > 0]
    if len(positive_groups) < len(names):
        raise ValueError(
            f"Only {len(positive_groups)} storms contain Failure labels, but four independent "
            "partitions are requested. Add genuine failure-containing storms or reduce partition "
            "count; do not synthesize labels or split storms across sets."
        )

    total = np.sum(list(stats.values()), axis=0)
    target = fractions[:, None] * total
    # Class-positive storm count is a feasibility guard and part of the objective.
    weights = np.array([1.0, 1.5, 0.75], dtype=float)
    denom = np.maximum(target, 1.0)

    def score_assignment(assignment):
        current = np.zeros_like(target)
        for p, partition in enumerate(assignment):
            for group in partition:
                current[p] += stats[group]
        relative_error = (current - target) / denom
        score = float(np.sum(relative_error ** 2 * weights))
        # Penalize partitions without positive storms heavily; those partitions cannot
        # provide meaningful failure-class metrics.
        for partition in assignment:
            if not any(stats[g][2] > 0 for g in partition):
                score += 1e6
        if any(len(partition) == 0 for partition in assignment):
            score += 1e6
        return score, current

    rng = np.random.default_rng(random_state)
    best_assignment, best_score = None, float("inf")
    group_array = np.asarray(groups, dtype=object)

    for _ in range(max(1, int(trials))):
        order = group_array.copy()
        rng.shuffle(order)
        # Vary ordering while still placing large/positive storms early enough to avoid
        # greedy starvation of the small calibration and test partitions.
        priority_noise = rng.random(len(order))
        order = sorted(order, key=lambda g: (
            stats[g][2], stats[g][1], stats[g][0], priority_noise[list(group_array).index(g)]
        ), reverse=True)

        assignment = [[] for _ in names]
        # Seed each partition with a different positive-containing storm.
        seeded = rng.choice(np.asarray(positive_groups, dtype=object), size=len(names), replace=False)
        for p, group in enumerate(seeded):
            assignment[p].append(group)
        seeded_set = set(seeded.tolist())
        for group in order:
            if group in seeded_set:
                continue
            # Test every destination against the global score, not only local target
            # error, so each decision considers all partitions simultaneously.
            candidates = []
            for p in range(len(names)):
                proposal = [part.copy() for part in assignment]
                proposal[p].append(group)
                candidates.append((score_assignment(proposal)[0] + rng.random() * 1e-8, p))
            assignment[min(candidates)[1]].append(group)

        current_score, _ = score_assignment(assignment)
        # Pair-swap local search repairs greedy allocations while preserving whole storms.
        improved = True
        passes = 0
        while improved and passes < 4:
            improved = False
            passes += 1
            for p in range(len(names)):
                for q in range(p + 1, len(names)):
                    for gp in list(assignment[p]):
                        for gq in list(assignment[q]):
                            proposal = [part.copy() for part in assignment]
                            proposal[p].remove(gp)
                            proposal[q].remove(gq)
                            proposal[p].append(gq)
                            proposal[q].append(gp)
                            proposal_score, _ = score_assignment(proposal)
                            if proposal_score + 1e-12 < current_score:
                                assignment = proposal
                                current_score = proposal_score
                                improved = True
            # continue until no pair swap improves the global score or pass cap reached

        final_score, _ = score_assignment(assignment)
        if final_score < best_score:
            best_assignment, best_score = [part.copy() for part in assignment], final_score

    if best_assignment is None or any(not any(stats[g][2] > 0 for g in part) for part in best_assignment):
        raise ValueError("Could not allocate failure-containing storms to all four partitions. Add more independent storms.")

    indices = {name: np.flatnonzero(np.isin(storm_ids, part)) for name, part in zip(names, best_assignment)}
    groups_by_partition = {name: sorted(map(str, part)) for name, part in zip(names, best_assignment)}
    return indices, groups_by_partition


def render_label_distribution_audit(label_sets, title="Label Distribution Audit"):
    """Display and persist class counts for the supplied partitions."""
    report = label_distribution_frame(label_sets)
    st.markdown(f"#### {title}")
    st.dataframe(report, use_container_width=True, hide_index=True)
    os.makedirs("data/processed", exist_ok=True)
    report.to_csv("data/processed/label_distribution_audit.csv", index=False)
    st.download_button(
        "Download Label Distribution Audit (CSV)",
        data=report.to_csv(index=False),
        file_name="label_distribution_audit.csv",
        mime="text/csv",
        key=f"label_distribution_download_{title.lower().replace(' ', '_')}",
    )
    return report


# --- HELPER: CSV EXPORT ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to a UTF-8 encoded CSV."""
    return df.to_csv(index=True).encode('utf-8')


# --- PAGE CONFIG ---
st.set_page_config(page_title="Q-Rating Chaos", layout="wide")
st.title(" Q-Rating Chaos: A Tri-Partite Quantum Framework for Typhoon Modeling and Microgrid Resilience")
st.subheader("ⓒ Engr. D.J. Medina 2026")
st.markdown(
    "Interactive POC: Quantum Kernel Networks, Conformal Prediction, and Quantum Walks for Typhoon Risk Modeling in Chiayi, Taiwan.")


# --- HELPER: GEOSPATIAL MAPBOX VISUALIZATION ---
def plot_interactive_map(mapper, title="Microgrid Topology", highlighted_nodes=None, node_colors=None):
    """Converts the NetworkX graph into an interactive Plotly Mapbox overlay."""
    edge_lon, edge_lat = [], []
    for edge in mapper.graph.edges():
        # FIX: Unpack the (lon, lat) tuples directly instead of using .x and .y
        x0, y0 = mapper.bus_coords[edge[0]]
        x1, y1 = mapper.bus_coords[edge[1]]
        edge_lon.extend([x0, x1, None])
        edge_lat.extend([y0, y1, None])

    edge_trace = go.Scattermapbox(
        lon=edge_lon, lat=edge_lat,
        mode='lines', line=dict(width=2, color='#555'), hoverinfo='none'
    )

    node_lon, node_lat, node_text, node_color_list = [], [], [], []
    for node in mapper.graph.nodes():
        # FIX: Access the tuple indices [0] for lon and [1] for lat
        node_lon.append(mapper.bus_coords[node][0])
        node_lat.append(mapper.bus_coords[node][1])
        node_text.append(f"Bus {node}")

        if node_colors and node in node_colors:
            node_color_list.append(node_colors[node])
        elif highlighted_nodes and node in highlighted_nodes:
            node_color_list.append('red')
        else:
            node_color_list.append('#1f77b4')  # Default Plotly Blue

    node_trace = go.Scattermapbox(
        lon=node_lon, lat=node_lat,
        mode='markers+text', text=[str(n) for n in mapper.graph.nodes()],
        textposition="top right", hoverinfo='text', hovertext=node_text,
        marker=dict(size=12, color=node_color_list)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False, hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        mapbox=dict(
                            style="carto-positron",
                            center=dict(lat=mapper.base_lat, lon=mapper.base_lon),
                            zoom=12
                        )
                    ))
    return fig


# --- SESSION STATE INITIALIZATION ---
if 'mapper' not in st.session_state:
    st.session_state.mapper = ChiayiMicrogridMapper()
    st.session_state.mapper.generate_topology()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'qkn_trained' not in st.session_state:
    st.session_state.qkn_trained = False
if 'qcp_calibrated' not in st.session_state:
    st.session_state.qcp_calibrated = False
if 'bus_train' not in st.session_state:
    st.session_state.bus_train = []

# --- SIDEBAR CONTROLS ---
# st.sidebar.header("Pipeline Controls")
# # n_samples = st.sidebar.slider("Sample Size (POC Speed)", 50, 500, 150)
# qkn_qubits = st.sidebar.selectbox("QKN Qubits", [1, 2, 3, 4, 5], index=0)
# qkn_layers = st.sidebar.slider("QKN Entangling Layers", 1, 5, 2)
# target_coverage = st.sidebar.slider("QCP Target Coverage (%)", 80, 99, 90) / 100.0

# --- TAB NAVIGATION ---
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     " Phase 1: Geo-Extraction",
#     " Phase 2: Quantum Training",
#     " Phase 3: Uncertainty Calibration",
#     " Phase 4: Inference Testing",
#     " Phase 5: CTQW Islanding"
# ])

# --- TAB NAVIGATION ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    " Phase 1: Geo-Extraction",
    " Phase 2: Global Optimization",
    " Phase 3: Quantum Training",
    " Phase 4: Uncertainty Calibration",
    " Phase 5: Inference Testing",
    " Phase 6: CTQW Islanding",
    " Evaluation: QKN Benchmark"
])

# --- TAB 1: MAPPING & DATA EXTRACTION (GNN COMPATIBLE) ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_interactive_map(st.session_state.mapper, "Enhanced IEEE 33-Bus System in Chiayi"),
                        use_container_width=True)
    with col2:
        st.info(
            "**Geospatial & Temporal Setup**\n\nThe 33-bus system is mapped to Chiayi. The extraction pipeline now builds full **Grid Snapshots** required for Graph Neural Network message passing.")
        ids = int(st.slider("How Many Typhoons to Process:", 4, 10, 8, help="Use enough distinct storm events to support independent train, validation, calibration, and test partitions."))
        forecast_horizon = st.slider("Forecast Horizon (Lead-Time Steps):", 1, 6, 2)
        target_snapshots = st.slider("QPU Snapshot Budget:", 10, 100, 25,
                                     help="How many full grid-states to extract. 25 snapshots = 825 individual bus evaluations.")

        if st.button("Extract Grid Snapshots", type="primary"):
            with st.spinner("Applying Rankine Vortex, Fatigue Models, and Assembling Graph Tensors..."):
                import pandas as pd
                import numpy as np

                # (Assuming haversine, rankine_vortex, vulnerability_curve are imported from src.utils)
                from src.utils import haversine, rankine_vortex, vulnerability_curve

                try:
                    df = pd.read_csv("data/raw/typhoon_data.csv")
                    df_tw = df[(df['lat'] >= 21) & (df['lat'] <= 26) & (df['lng'] >= 118) & (df['lng'] <= 123)].copy()

                    if df_tw.empty:
                        st.error(" No historical typhoons found near Taiwan.")
                        st.stop()

                    taiwan_seq_ids = df_tw['seq_id'].unique()[-ids:]
                    df_events = df_tw[df_tw['seq_id'].isin(taiwan_seq_ids)].sort_index()

                    chiayi_lat, chiayi_lng = 23.48, 120.44
                    coastline_lng = 120.15

                    bus_coords = st.session_state.mapper.bus_coords

                    grid_snapshots = []
                    grid_labels = []
                    time_steps = 4

                    BOUNDS = {
                        'Wind_Speed': (0, 70),
                        'Delta_Wind': (-20, 20),
                        'Distance_to_Eye': (0, 500),
                        'Coastal_Exposure': (0, 1)
                    }


                    def scale_to_phase(val, feature_name):
                        min_v, max_v = BOUNDS[feature_name]
                        clipped = np.clip(val, min_v, max_v)
                        return -np.pi + 2 * np.pi * ((clipped - min_v) / (max_v - min_v))


                    # --- GRID-CENTRIC EXTRACTION ---
                    for seq_id, storm_track in df_events.groupby('seq_id'):
                        storm_track = storm_track.reset_index(drop=True)
                        n_records = len(storm_track)

                        # Initialize fatigue trackers for all 33 buses
                        fatigue_trackers = {bus_id: 0 for bus_id in range(1, 34)}

                        # We iterate chronologically through the storm
                        for t in range(time_steps - 1, n_records - forecast_horizon):
                            snapshot_features = []
                            snapshot_labels = []
                            snapshot_has_stress = False

                            # Build the full 33-bus graph state for this time window
                            for bus_id in range(1, 34):
                                bus_lon, bus_lat = bus_coords[bus_id]
                                coastal_exposure = max(0,
                                                       1 - ((bus_lon - coastline_lng) / (chiayi_lng - coastline_lng)))

                                bus_history = []
                                # Gather the 4-step history for this specific bus
                                for step in range(t - time_steps + 1, t + 1):
                                    row = storm_track.iloc[step]
                                    dist_km = haversine(row['lat'], row['lng'], bus_lat, bus_lon)
                                    wind = rankine_vortex(row['wind'] * 0.51444, dist_km)
                                    bus_history.append({'Wind': wind, 'Dist': dist_km})

                                # Process Fatigue and Labels based on the *Forecast Horizon* step
                                future_row = storm_track.iloc[t + forecast_horizon]
                                future_dist = haversine(future_row['lat'], future_row['lng'], bus_lat, bus_lon)
                                future_wind = rankine_vortex(future_row['wind'] * 0.51444, future_dist)

                                if future_wind >= 17.0:
                                    fatigue_trackers[bus_id] += 1
                                else:
                                    fatigue_trackers[bus_id] = max(0, fatigue_trackers[bus_id] - 1)

                                fatigue_penalty = min(0.30, fatigue_trackers[bus_id] * 0.05)
                                noisy_wind = future_wind + np.random.normal(0, 3.0)
                                base_fail_prob = vulnerability_curve(noisy_wind)
                                surge_penalty = coastal_exposure * 0.40

                                total_risk = base_fail_prob + surge_penalty + fatigue_penalty
                                label = 1 if total_risk >= 0.45 else 0

                                if label == 1: snapshot_has_stress = True

                                # Scale Features for the Quantum Kernel
                                delta_wind = bus_history[-1]['Wind'] - bus_history[0]['Wind']
                                seq_scaled = []
                                for step_data in bus_history:
                                    seq_scaled.append([
                                        scale_to_phase(step_data['Wind'], 'Wind_Speed'),
                                        scale_to_phase(delta_wind, 'Delta_Wind'),
                                        scale_to_phase(step_data['Dist'], 'Distance_to_Eye'),
                                        scale_to_phase(coastal_exposure, 'Coastal_Exposure')
                                    ])

                                snapshot_features.append(seq_scaled)  # Shape: (4, 4) per bus
                                snapshot_labels.append(label)

                            # Append the entire 33-bus graph snapshot
                            # Snapshot features shape: (33, 4, 4) -> (Nodes, Time_Steps, Features)
                            grid_snapshots.append({
                                'storm_id': str(seq_id),
                                'features': np.array(snapshot_features),
                                'labels': np.array(snapshot_labels),
                                'has_stress': snapshot_has_stress
                            })

                    # --- STRATIFIED SNAPSHOT SELECTION ---
                    # Separate snapshots into stressful (contains at least one failure) and safe
                    df_snapshots = pd.DataFrame(grid_snapshots)

                    # Audit all generated snapshots before budget selection or data splitting.
                    all_generated_labels = np.concatenate(df_snapshots["labels"].to_list())
                    render_label_distribution_audit(
                        {"All generated bus labels (before snapshot selection/splitting)": all_generated_labels},
                        title="Pre-Split Label Distribution"
                    )

                    df_stress = df_snapshots[df_snapshots['has_stress'] == True]
                    df_safe = df_snapshots[df_snapshots['has_stress'] == False]

                    # Do not fabricate labels if the risk rule produces no failures.
                    # Artificially forcing labels here would contaminate evaluation and hide
                    # a threshold/data-generation problem.
                    if df_stress.empty:
                        st.error(
                            "The current Failure_Label rule produced no failure snapshots. "
                            "Review vulnerability_curve(), the total_risk threshold (0.45), "
                            "wind/risk scaling, and the selected storms. No labels were fabricated."
                        )
                        st.dataframe(
                            label_distribution_frame({"All extracted bus labels": np.concatenate(df_snapshots["labels"].to_list())}),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.stop()

                    # Preserve every failure-containing snapshot when the budget permits.
                    # Then reserve one snapshot per storm, and fill the remaining budget.
                    available_storms = df_snapshots['storm_id'].dropna().astype(str).unique().tolist()
                    budget = min(int(target_snapshots), len(df_snapshots))
                    rng = np.random.default_rng(42)
                    stress_indices = df_snapshots.index[df_snapshots['has_stress'] == True].to_list()
                    mandatory_indices = []
                    for storm_id in available_storms:
                        storm_rows = df_snapshots.index[df_snapshots['storm_id'].astype(str) == storm_id].to_numpy()
                        if len(storm_rows):
                            mandatory_indices.append(int(rng.choice(storm_rows)))
                    # Prioritize positive snapshots so random selection cannot silently omit all failures.
                    prioritized = list(dict.fromkeys(stress_indices + mandatory_indices))
                    if len(prioritized) > budget:
                        st.warning(
                            "The snapshot budget is smaller than the number of failure-containing/mandatory snapshots. "
                            "Some storms may be omitted; increase the snapshot budget for a reliable split."
                        )
                        prioritized = prioritized[:budget]
                    remaining_budget = max(0, budget - len(prioritized))
                    remaining_indices = df_snapshots.index.difference(prioritized).to_numpy()
                    extra_indices = rng.choice(
                        remaining_indices,
                        size=min(remaining_budget, len(remaining_indices)),
                        replace=False
                    ).tolist() if remaining_budget else []
                    selected_indices = list(dict.fromkeys(prioritized + extra_indices))
                    final_snapshots = df_snapshots.loc[selected_indices].sample(frac=1, random_state=42).reset_index(drop=True)
                    if final_snapshots['storm_id'].nunique() < 3:
                        st.warning('Fewer than three storms are represented in the selected snapshots. Increase the snapshot budget or process more typhoons.')

                    # Store as numpy arrays.
                    # X_tensor shape: (Snapshots, 33 Nodes, 4 Timesteps, 4 Features)
                    # Y_tensor shape: (Snapshots, 33 Nodes)
                    st.session_state.X_grid_tensor = np.stack(final_snapshots['features'].values)
                    st.session_state.Y_grid_tensor = np.stack(final_snapshots['labels'].values)
                    st.session_state.storm_ids = final_snapshots['storm_id'].astype(str).to_numpy()

                    st.session_state.feature_cols_all = ['Wind_Speed', 'Delta_Wind', 'Distance_to_Eye',
                                                         'Coastal_Exposure']
                    st.session_state.coresets_generated = True
                    st.success(
                        f"Extracted {len(final_snapshots)} Graph Snapshots (Total: {len(final_snapshots) * 33} bus evaluations).")

                except Exception as e:
                    st.error(f" An error occurred: {e}")

    # --- PRE-OPTIMIZATION RAW SNAPSHOT VIEWER ---
    if st.session_state.get('coresets_generated', False):
        st.divider()
        st.markdown("###  Raw Grid Snapshot Viewer")
        st.write(
            "Inspect the physical distribution of quantum-scaled features (in radians) and failure labels across the grid *before* applying feature selection.")

        import plotly.graph_objects as go

        view_col1, view_col2 = st.columns(2)
        with view_col1:
            snap_idx = st.slider("Select Target Snapshot:", 0, len(st.session_state.X_grid_tensor) - 1, 0,
                                 help="Scroll through the timeline of extracted weather events.")
        with view_col2:
            feat_name = st.selectbox("Select Node Feature (Most Recent Timestep):", st.session_state.feature_cols_all)

        feat_idx = st.session_state.feature_cols_all.index(feat_name)
        mapper = st.session_state.mapper

        fig_viewer = go.Figure()

        # 1. Draw Physical Power Lines (Edges)
        for u, v in mapper.graph.edges():
            x0, y0 = mapper.bus_coords[u]
            x1, y1 = mapper.bus_coords[v]
            fig_viewer.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines', line=dict(color='lightgrey', width=1),
                showlegend=False, hoverinfo='none'
            ))

        # 2. Extract Data for Selected Snapshot & Draw Nodes
        node_x, node_y, node_vals, node_labels, hover_texts = [], [], [], [], []

        for i, node in enumerate(mapper.graph.nodes()):
            x, y = mapper.bus_coords[node]
            node_x.append(x)
            node_y.append(y)

            # Extract the specific feature value at the most recent time step (-1)
            val = st.session_state.X_grid_tensor[snap_idx, i, -1, feat_idx]
            is_fail = st.session_state.Y_grid_tensor[snap_idx, i]

            node_vals.append(val)
            node_labels.append(is_fail)
            hover_texts.append(
                f"<b>Bus {node}</b><br>{feat_name}: {val:.2f} rad<br>State: {'FAILED' if is_fail else 'Safe'}")

        fig_viewer.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(
                size=12,
                color=node_vals,
                colorscale='Viridis',  # Cool to warm color mapping
                colorbar=dict(title=f"{feat_name} (rad)"),
                showscale=True,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=hover_texts, hoverinfo='text', name='Substations'
        ))

        # 3. Explicitly Mark Grid Failures
        fail_x = [x for i, x in enumerate(node_x) if node_labels[i] == 1]
        fail_y = [y for i, y in enumerate(node_y) if node_labels[i] == 1]
        if fail_x:
            fig_viewer.add_trace(go.Scatter(
                x=fail_x, y=fail_y, mode='markers',
                marker=dict(size=18, symbol='x', color='red', line=dict(width=2)),
                name='Grid Failure', hoverinfo='skip'
            ))

        fig_viewer.update_layout(
            title=f"Topology State Map: Snapshot {snap_idx}",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig_viewer, use_container_width=True)
    # --- META-HEURISTIC FEATURE SELECTION (CGWO) ---
    if st.session_state.get('coresets_generated', False):
        st.divider()
        st.markdown("###  Sequence-Aware Information Bottleneck (Binary CGWO)")
        st.write(
            "A Chaotic Grey Wolf Optimizer hunting for the optimal combination of temporal features across the entire graph topology.")

        feature_cols_all = st.session_state.feature_cols_all

        if st.button("Run Chaotic Grey Wolf Optimization"):
            with st.spinner("Wolves are evaluating graph feature topologies..."):
                import numpy as np
                from sklearn.linear_model import RidgeClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import balanced_accuracy_score

                X_raw = st.session_state.X_grid_tensor  # (N, 33, 4, 4)
                Y_raw = st.session_state.Y_grid_tensor  # (N, 33)

                # For the fast proxy evaluator, flatten the Nodes into the Batch dimension
                # Proxy shape: (N*33, 4, 4) -> (N*33, 16)
                X_proxy = X_raw.reshape(-1, X_raw.shape[2], X_raw.shape[3]).reshape(-1, 16)
                Y_proxy = Y_raw.reshape(-1)

                if len(np.unique(Y_proxy)) < 2:
                    st.error(" Data Imbalance: No failures detected. Adjust sliders.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X_proxy, Y_proxy, test_size=0.3, random_state=42, stratify=Y_proxy
                )

                num_features = len(feature_cols_all)
                num_wolves = 5
                max_iter = 50

                wolves = np.random.randint(2, size=(num_wolves, num_features))
                for w in range(num_wolves):
                    if np.sum(wolves[w]) == 0: wolves[w, np.random.randint(num_features)] = 1

                alpha_pos, beta_pos, delta_pos = np.zeros(num_features), np.zeros(num_features), np.zeros(num_features)
                alpha_score, beta_score, delta_score = -float("inf"), -float("inf"), -float("inf")
                chaotic_map = 0.5

                progress_bar = st.progress(0)

                for it in range(max_iter):
                    chaotic_map = 4.0 * chaotic_map * (1.0 - chaotic_map)
                    a = (2.0 - (it * (2.0 / max_iter))) * chaotic_map

                    for i in range(num_wolves):
                        mask_2d = np.repeat(wolves[i], 4)
                        X_sub_train = X_train[:, mask_2d == 1]
                        X_sub_test = X_test[:, mask_2d == 1]

                        if X_sub_train.shape[1] > 0:
                            try:
                                clf = RidgeClassifier(class_weight='balanced')
                                clf.fit(X_sub_train, y_train)
                                preds = clf.predict(X_sub_test)
                                score = balanced_accuracy_score(y_test, preds)
                                if np.isnan(score): score = 0.0
                            except:
                                score = 0.0
                        else:
                            score = 0.0

                        if score > alpha_score:
                            delta_score, delta_pos = beta_score, beta_pos.copy()
                            beta_score, beta_pos = alpha_score, alpha_pos.copy()
                            alpha_score, alpha_pos = score, wolves[i].copy()
                        elif score > beta_score:
                            delta_score, delta_pos = beta_score, beta_pos.copy()
                            beta_score, beta_pos = score, wolves[i].copy()
                        elif score > delta_score:
                            delta_score, delta_pos = score, wolves[i].copy()

                    for i in range(num_wolves):
                        for j in range(num_features):
                            r1, r2 = np.random.random(), np.random.random()
                            A1, C1 = 2 * a * r1 - a, 2 * r2
                            X1 = alpha_pos[j] - A1 * abs(C1 * alpha_pos[j] - wolves[i, j])

                            r1, r2 = np.random.random(), np.random.random()
                            A2, C2 = 2 * a * r1 - a, 2 * r2
                            X2 = beta_pos[j] - A2 * abs(C2 * beta_pos[j] - wolves[i, j])

                            r1, r2 = np.random.random(), np.random.random()
                            A3, C3 = 2 * a * r1 - a, 2 * r2
                            X3 = delta_pos[j] - A3 * abs(C3 * delta_pos[j] - wolves[i, j])

                            X_new = (X1 + X2 + X3) / 3.0
                            prob = 1 / (1 + np.exp(-10 * (X_new - 0.5)))
                            wolves[i, j] = 1 if np.random.random() < prob else 0

                        if np.sum(wolves[i]) < 2:
                            wolves[i, np.random.choice(num_features, 2, replace=False)] = 1

                    progress_bar.progress((it + 1) / max_iter)

                st.session_state.cgwo_best_mask = alpha_pos
                st.session_state.cgwo_best_score = alpha_score
                st.success(f"Optimization Complete! Peak Balanced Accuracy: {alpha_score:.4f}")

        if 'cgwo_best_mask' in st.session_state:
            active_mask = st.session_state.cgwo_best_mask
            selected_features = [feature_cols_all[j] for j in range(len(active_mask)) if active_mask[j] == 1]

            st.info(f" **Alpha Wolf Optimal Subset:** {', '.join(selected_features)}")

            if st.button("Lock Topology & Formulate Quantum Graph Dataset"):
                import os
                import plotly.graph_objects as go
                import plotly.express as px
                from sklearn.decomposition import PCA

                # Apply the CGWO mask to the 4D Tensor
                selected_indices = [feature_cols_all.index(f) for f in selected_features]
                X_tensor = st.session_state.X_grid_tensor[:, :, :, selected_indices]  # (Snapshots, 33, 4, Selected)
                Y_tensor = st.session_state.Y_grid_tensor  # (Snapshots, 33)

                n_samples = len(X_tensor)
                storm_ids = np.asarray(st.session_state.storm_ids).astype(str)
                unique_storms = np.unique(storm_ids)

                # Hold out complete typhoon events; never split snapshots from one storm across partitions.
                if len(unique_storms) < 4:
                    st.error(
                        f"Only {len(unique_storms)} distinct storm event(s) are available. "
                        "Extract at least 4 distinct storms to create independent train, validation, "
                        "calibration, and test partitions."
                    )
                    st.stop()

                try:
                    partition_indices, partition_storms = stratified_storm_group_split(
                        storm_ids, Y_tensor, fractions=(0.60, 0.15, 0.10, 0.15), random_state=42
                    )
                except ValueError as split_error:
                    st.error(f"Class-aware storm split could not be created: {split_error}")
                    st.stop()

                train_idx = partition_indices["train"]
                val_idx = partition_indices["validation"]
                cal_idx = partition_indices["calibration"]
                test_idx = partition_indices["test"]
                train_storms = set(partition_storms["train"])
                val_storms = set(partition_storms["validation"])
                cal_storms = set(partition_storms["calibration"])
                test_storms = set(partition_storms["test"])
                all_partition_storms = [train_storms, val_storms, cal_storms, test_storms]
                if any(
                    all_partition_storms[i] & all_partition_storms[j]
                    for i in range(len(all_partition_storms))
                    for j in range(i + 1, len(all_partition_storms))
                ):
                    st.error("Event split integrity check failed: storm overlap detected.")
                    st.stop()

                st.session_state.X_seq_train, st.session_state.y_train = X_tensor[train_idx], Y_tensor[train_idx]
                st.session_state.X_seq_val, st.session_state.y_val = X_tensor[val_idx], Y_tensor[val_idx]
                st.session_state.X_seq_cal, st.session_state.y_cal = X_tensor[cal_idx], Y_tensor[cal_idx]
                st.session_state.X_seq_test, st.session_state.y_test = X_tensor[test_idx], Y_tensor[test_idx]
                st.session_state.train_storm_ids = sorted(train_storms)
                st.session_state.val_storm_ids = sorted(val_storms)
                st.session_state.cal_storm_ids = sorted(cal_storms)
                st.session_state.test_storm_ids = sorted(test_storms)

                # Audit the complete label pool and every current event-level partition.
                # Validation is used for model selection; calibration remains held out for QCP only.
                full_labels = Y_tensor.reshape(-1)
                train_labels = Y_tensor[train_idx].reshape(-1)
                val_labels = Y_tensor[val_idx].reshape(-1)
                cal_labels = Y_tensor[cal_idx].reshape(-1)
                test_labels = Y_tensor[test_idx].reshape(-1)
                distribution_report = render_label_distribution_audit({
                    "Full dataset before split": full_labels,
                    "Training": train_labels,
                    "Validation": val_labels,
                    "Calibration": cal_labels,
                    "Held-out testing": test_labels,
                }, title="Full and Partition Label Distributions")
                st.session_state.label_distribution_report = distribution_report.to_dict("records")

                st.warning(
                    "Check the distribution table above. If any partition lacks either class, do not interpret "
                    "binary Recall/Precision/F1 for that partition as a complete evaluation. Increase the number "
                    "of storms or revise event-level allocation without splitting a storm across partitions."
                )
                st.success(
                    f"Event-level split: {len(train_storms)} train, {len(val_storms)} validation, "
                    f"{len(cal_storms)} calibration, {len(test_storms)} held-out test storms."
                )

                st.session_state.feature_cols = selected_features
                st.session_state.data_loaded = True

                # --- VISUALIZATIONS ---
                st.divider()
                st.markdown("###  Graph Dataset Visualizations")

                # Flatten just the most recent timestep for visualization
                # X_flat shape: (N*33, Selected)
                X_flat = X_tensor[:, :, -1, :].reshape(-1, len(selected_features))
                Y_flat = Y_tensor.reshape(-1)

                df_export = pd.DataFrame(X_flat, columns=[f"{feat} (rad)" for feat in selected_features])
                # Preserve storm identity for event-level generalization checks.
                df_export["storm_id"] = np.repeat(st.session_state.storm_ids.astype(str), Y_tensor.shape[1])
                df_export["snapshot_id"] = np.repeat(np.arange(len(Y_tensor)), Y_tensor.shape[1])
                df_export["bus_id"] = np.tile(np.arange(1, Y_tensor.shape[1] + 1), len(Y_tensor))
                df_export["Failure_Label"] = Y_flat

                os.makedirs("data/processed", exist_ok=True)
                df_export.to_csv("data/processed/final_quantum_dataset.csv", index=False)

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    df_safe = df_export[df_export["Failure_Label"] == 0]
                    df_fail = df_export[df_export["Failure_Label"] == 1]

                    # Use the actual feature column names in df_export.
                    radial_cols = [f"{feat} (rad)" for feat in selected_features]

                    # Calculate means using only selected feature columns.
                    mean_safe = (
                        df_safe[radial_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .mean()
                        .fillna(0)
                        .values.tolist()
                        if not df_safe.empty
                        else [0] * len(selected_features)
                    )

                    mean_fail = (
                        df_fail[radial_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .mean()
                        .fillna(0)
                        .values.tolist()
                        if not df_fail.empty
                        else [0] * len(selected_features)
                    )

                    # Close the radar chart by repeating the first feature.
                    mean_safe += [mean_safe[0]] if mean_safe else []
                    mean_fail += [mean_fail[0]] if mean_fail else []

                    radial_cols += [radial_cols[0]] if radial_cols else []

                    fig_radar = go.Figure()
                    fig_radar.add_trace(
                        go.Scatterpolar(r=mean_safe, theta=radial_cols, fill='toself', name='Safe (0)', line_color='blue'))
                    fig_radar.add_trace(go.Scatterpolar(r=mean_fail, theta=radial_cols, fill='toself', name='Failure (1)',
                                                        line_color='red'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-np.pi, np.pi])),
                                            title="Average Node Phase Profiles")
                    st.plotly_chart(fig_radar, use_container_width=True)

                with v_col2:
                    if len(selected_features) > 2:
                        pca = PCA(n_components=2)
                        components = pca.fit_transform(X_flat)
                        df_scatter = pd.DataFrame(components, columns=['Dim 1 (PCA)', 'Dim 2 (PCA)'])
                        title_scatter = "Node Distribution (PCA Reduced)"
                    else:
                        df_scatter = pd.DataFrame(X_flat, columns=[selected_features[0], selected_features[1]])
                        df_scatter.rename(columns={selected_features[0]: 'Dim 1', selected_features[1]: 'Dim 2'},
                                          inplace=True)
                        title_scatter = "Node Distribution (2D)"

                    df_scatter['State'] = Y_flat
                    df_scatter['State'] = df_scatter['State'].map({0: 'Safe', 1: 'Failure'})

                    fig_scatter = px.scatter(df_scatter, x=df_scatter.columns[0], y=df_scatter.columns[1], color='State',
                                             title=title_scatter, color_discrete_map={'Safe': 'blue', 'Failure': 'red'},
                                             opacity=0.7)
                    fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.success("Graph Matrices Prepared! You can now proceed to Phase 2 (Bayesian Optimization).")

# --- TAB 2: PHASE 2 - BAYESIAN OPTIMIZATION (OPTUNA) ---
with tab2:
    st.markdown("###  Phase 2: Bayesian Hyperparameter Optimization")
    st.info(
        "Optimize the Quantum Circuit topology and the Quantum Spatiotemporal GNN parameters using Tree-structured Parzen Estimator (TPE) Bayesian search.")

    if not st.session_state.get('data_loaded', False):
        st.warning(" Please complete the Data Extraction in Tab 1 first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Bayesian Search Space")
            n_trials = st.slider("Optimization Trials:", 5, 50, 15)
            proxy_samples = st.slider("Proxy Subset Size:", 16, 128, 32)

            run_bo_btn = st.button("Initialize Bayesian Search", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Bayesian Telemetry")
            bo_status = st.empty()
            bo_progress = st.progress(0)
            st.caption("Bayesian Convergence (Proxy Validation Loss)")
            bo_chart = st.empty()
            bo_params_display = st.empty()

        if run_bo_btn:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import pandas as pd
            import numpy as np
            import optuna

            from src.qkn import QuantumSpatiotemporalGNN, QuantumKernelNetwork

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            try:
                bo_status.info(" Bayesian Engine Deployed. Mapping Quantum-Graph-Temporal hyper-surface...")

                X_seq_raw = st.session_state.X_seq_train[:proxy_samples]
                y_train_raw = st.session_state.y_train[:proxy_samples]
                X_val_raw = st.session_state.X_seq_val[:proxy_samples]
                y_val_raw = st.session_state.y_val[:proxy_samples]

                y_proxy = torch.tensor(y_train_raw, dtype=torch.float32)
                y_val_t = torch.tensor(y_val_raw, dtype=torch.float32)

                edge_index, _ = st.session_state.mapper.export_edge_index()

                convergence_history = []
                tracker = {'best_loss': float('inf')}


                def objective(trial):
                    n_qubits = trial.suggest_int("n_qubits", 2, 4)
                    q_layers = trial.suggest_int("q_layers", 1, 3)
                    entangling_type = trial.suggest_categorical("entangling_type",
                                                                ["StronglyEntangling", "BasicEntangler"])

                    conv_out = trial.suggest_int("conv_out", 8, 32, step=8)
                    gat_heads = trial.suggest_int("gat_heads", 1, 4)
                    lstm_hidden = trial.suggest_int("lstm_hidden", 16, 64, step=16)
                    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
                    dropout = trial.suggest_float("dropout", 0.1, 0.5)

                    qkn = create_qkn(st.session_state.get("qkn_backend", "pennylane"), n_qubits=n_qubits, layers=q_layers, entangling_type=entangling_type)

                    X_proxy_q = qkn.extract_temporal_quantum_features(X_seq_raw)
                    X_val_q = qkn.extract_temporal_quantum_features(X_val_raw)

                    model = QuantumSpatiotemporalGNN(
                        in_channels=n_qubits,
                        seq_len=4,
                        conv_out=conv_out,
                        gat_heads=gat_heads,
                        lstm_hidden=lstm_hidden,
                        dropout=dropout
                    )

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.BCELoss()

                    model.train()
                    optimizer.zero_grad()
                    out = model(X_proxy_q, edge_index)

                    # FIX: view_as() guarantees the target perfectly matches the output shape
                    loss = criterion(out, y_proxy.view_as(out))
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_out = model(X_val_q, edge_index)
                        val_loss = criterion(val_out, y_val_t.view_as(val_out)).item()

                    if np.isnan(val_loss): return float('inf')

                    if val_loss < tracker['best_loss']:
                        tracker['best_loss'] = val_loss

                    convergence_history.append(tracker['best_loss'])
                    bo_chart.line_chart(pd.DataFrame({"Minimum Surrogate Loss": convergence_history}))

                    return val_loss


                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
                for i in range(n_trials):
                    study.optimize(objective, n_trials=1)
                    bo_progress.progress((i + 1) / n_trials)
                    bo_params_display.json(study.best_params)

                st.session_state.bo_best_params = study.best_params
                bo_status.success(" Bayesian Optimization Complete! Architecture locked. Proceed to Phase 3.")

            except Exception as e:
                bo_status.error(f" BO Error: {e}")

# --- TAB 3: PHASE 3 - HYBRID DL TRAINING ---
with tab3:
    st.markdown("### Phase 3: Hybrid QKN + QSTGNN Training")
    st.info(
        "Hybrid mode fuses quantum expectation-value channels (A) with fidelity-to-training-landmark channels (B), then passes the fused node/time features through temporal convolution, physical-grid GAT message passing, BiLSTM, temporal attention, and bus-level failure prediction.")

    if not st.session_state.get('data_loaded', False):
        st.warning(" Please complete the Data Extraction and Optimization in prior tabs first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Step 1: Configuration Mode")
            config_mode = st.radio(
                "How would you like to set the architecture parameters?",
                ["Manual Configuration", "Run New Bayesian Search (Optuna)", "Use Cached Optimization"]
            )

            st.markdown("#### Step 2: Model Architecture")
            model_architecture = st.selectbox(
                "Predictive architecture",
                ["QSTGNN (expectation branch A)", "Hybrid QKN (A+B) + QSTGNN"],
                index=1,
                help="Hybrid mode fuses QKN expectation values with fidelity similarities to training-only quantum landmarks before graph-temporal learning."
            )
            st.session_state.model_architecture = model_architecture
            b_fusion_weight = st.slider(
                "B-branch fusion weight",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Scales fidelity-kernel branch B before A+B concatenation. 0 disables B; 0.25 makes B a secondary signal while preserving A expectation features unchanged."
            )
            st.session_state.b_fusion_weight = float(b_fusion_weight)

            st.markdown("#### Step 3: Quantum Backend")
            qkn_backend = st.selectbox(
                "QKN Implementation Backend:",
                ["pennylane", "cudaq"],
                index=0 if st.session_state.get("qkn_backend", "pennylane") == "pennylane" else 1,
                help="Choose the Phase 3 quantum execution backend. Both backends expose the same QKN feature-extraction interface."
            )
            st.session_state.qkn_backend = qkn_backend

            st.markdown("#### Step 3: Architecture Parameters")
            if config_mode == "Manual Configuration":
                q_qubits = st.number_input("Qubits:", 2, 6, 3)
                q_layers = st.number_input("Quantum Layers:", 1, 4, 2)
                q_entangle = st.selectbox("Entanglement:", ["StronglyEntangling", "BasicEntangler"])
                c_conv = st.number_input("GNN Conv Channels:", 8, 64, 16, step=8)
                c_gat = st.number_input("GAT Heads:", 1, 8, 2)
                c_lstm = st.number_input("LSTM Hidden State:", 16, 128, 32, step=16)
                c_lr = st.number_input("Learning Rate:", value=0.005, format="%.4f")
                c_drop = st.slider("Dropout:", 0.0, 0.6, 0.3)

            elif config_mode == "Run New Bayesian Search (Optuna)":
                n_trials = st.slider("Optimization Trials:", 2, 20, 5)
                proxy_samples = st.slider("Proxy Subset Size (Speed up BO):", 16, 128, 32)
                st.caption(
                    "Optuna will map the hyper-surface and automatically find the optimal combination before training.")

            elif config_mode == "Use Cached Optimization":
                if 'bo_best_params' in st.session_state:
                    st.success(" Found cached BO parameters:")
                    st.json(st.session_state.bo_best_params)
                else:
                    st.warning(" No cached parameters found. Will run with default fallback values.")

            st.markdown("#### Step 3: Training Config")
            epochs = st.number_input("Training Epochs:", min_value=5, max_value=200, value=25, step=5)
            batch_size = st.selectbox("Batch Size:", [8, 16, 32], index=1)

            train_btn = st.button("Initialize & Begin Training Sequence", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Real-Time Telemetry")
            telemetry_status = st.empty()
            train_progress = st.progress(0)

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                loss_chart = st.empty()
            with chart_col2:
                acc_chart = st.empty()

        if train_btn:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            # --- DYNAMIC SHAPE DETECTION (Fixes the IndexError) ---
            num_nodes = st.session_state.X_seq_train.shape[1]
            dynamic_seq_len = st.session_state.X_seq_train.shape[2]

            # --- PARAMETER RESOLUTION BLOCK ---
            if config_mode == "Run New Bayesian Search (Optuna)":
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                telemetry_status.info(" Running Bayesian Search...")


                def objective(trial):
                    nq = trial.suggest_int("n_qubits", 2, 4)
                    ql = trial.suggest_int("q_layers", 1, 3)
                    qe = trial.suggest_categorical("entangling_type", ["StronglyEntangling", "BasicEntangler"])
                    cc = trial.suggest_int("conv_out", 8, 32, step=8)
                    gh = trial.suggest_int("gat_heads", 1, 4)
                    cl = trial.suggest_int("lstm_hidden", 16, 64, step=16)
                    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
                    dr = trial.suggest_float("dropout", 0.2, 0.6)

                    qkn_bo = create_qkn(st.session_state.get("qkn_backend", "pennylane"), n_qubits=nq, layers=ql, entangling_type=qe)
                    X_p_q = qkn_bo.extract_temporal_quantum_features(st.session_state.X_seq_train[:proxy_samples])
                    y_p_t = torch.tensor(st.session_state.y_train[:proxy_samples], dtype=torch.float32)

                    # Dynamic edge index based on actual node count
                    edge_idx, _ = st.session_state.mapper.export_edge_index()

                    model_bo = QuantumSpatiotemporalGNN(in_channels=nq, seq_len=dynamic_seq_len, conv_out=cc,
                                                        gat_heads=gh, lstm_hidden=cl, dropout=dr)
                    optimizer_bo = optim.Adam(model_bo.parameters(), lr=lr)
                    criterion_bo = nn.BCELoss()

                    model_bo.train()
                    for _ in range(3):
                        optimizer_bo.zero_grad()
                        out = model_bo(X_p_q, edge_idx)
                        out_clamped = torch.clamp(out, 1e-7, 1.0 - 1e-7)
                        loss = criterion_bo(out_clamped, y_p_t.view_as(out_clamped))
                        loss.backward()
                        optimizer_bo.step()
                    return loss.item()


                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=n_trials)

                # Cache the best params globally
                st.session_state.bo_best_params = study.best_params
                params = study.best_params

                # Map to local variables for training
                q_qubits, q_layers, q_entangle = params['n_qubits'], params['q_layers'], params['entangling_type']
                c_conv, c_gat, c_lstm, c_lr, c_drop = params['conv_out'], params['gat_heads'], params['lstm_hidden'], \
                params['lr'], params['dropout']
                telemetry_status.success(
                    f" BO Complete! Qubits:{q_qubits}, Entanglement:{q_entangle}, LR:{c_lr:.4f}, Lstm:{c_lstm}")

            elif config_mode == "Use Cached Optimization":
                if 'bo_best_params' in st.session_state:
                    params = st.session_state.bo_best_params
                    q_qubits, q_layers, q_entangle = params['n_qubits'], params['q_layers'], params.get(
                        'entangling_type', 'StronglyEntangling')
                    c_conv, c_gat, c_lstm, c_lr, c_drop = params['conv_out'], params['gat_heads'], params[
                        'lstm_hidden'], params['lr'], params['dropout']
                else:
                    # Fallback defaults
                    q_qubits, q_layers, q_entangle = 3, 2, "StronglyEntangling"
                    c_conv, c_gat, c_lstm, c_lr, c_drop = 16, 2, 32, 0.005, 0.3

            # --- TRAINING BLOCK ---
            telemetry_status.info(" Extracting quantum representations...")

            qkn = create_qkn(st.session_state.get("qkn_backend", "pennylane"), n_qubits=q_qubits, layers=q_layers, entangling_type=q_entangle)
            hybrid_mode = st.session_state.get("model_architecture", "Hybrid QKN (A+B) + QSTGNN") == "Hybrid QKN (A+B) + QSTGNN"
            if hybrid_mode:
                from src.hybrid_qstgnn import HybridQSTGNNFeatureEncoder
                encoder = HybridQSTGNNFeatureEncoder(
                    n_qubits=int(q_qubits),
                    n_landmarks=8,
                    random_state=42,
                    b_weight=float(st.session_state.get("b_fusion_weight", 0.25))
                ).fit(st.session_state.X_seq_train)
                X_train_q = encoder.transform(st.session_state.X_seq_train, qkn)
                X_val_q = encoder.transform(st.session_state.X_seq_val, qkn)
                X_cal_q = encoder.transform(st.session_state.X_seq_cal, qkn)
                os.makedirs("data/processed", exist_ok=True)
                torch.save(encoder.state_dict(), "data/processed/hybrid_qstgnn_encoder.pt")
                st.session_state.hybrid_encoder_state = encoder.state_dict()
                model_in_channels = encoder.output_channels
                st.session_state.hybrid_feature_channels = model_in_channels
                telemetry_status.info(
                    f" Weighted hybrid fusion complete: A has priority; {q_qubits} expectation channels "
                    f"are unchanged and {len(encoder.landmarks_)} B fidelity channels are scaled by "
                    f"{encoder.b_weight:.2f}. Total QSTGNN input channels: {model_in_channels}."
                )
            else:
                X_train_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_train)
                X_val_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_val)
                X_cal_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_cal)
                model_in_channels = int(q_qubits)

            # Save expectation/fused embeddings for visualization and audit.
            st.session_state.X_train_q_numpy = X_train_q.detach().cpu().numpy()

            # Apply label smoothing to prevent 0.000 logit saturation loss
            y_train_smooth = np.clip(st.session_state.y_train, 0.05, 0.95)
            y_val_smooth = np.clip(st.session_state.y_val, 0.05, 0.95)

            y_train_t = torch.tensor(y_train_smooth, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_smooth, dtype=torch.float32)

            train_loader = DataLoader(TensorDataset(X_train_q, y_train_t), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_q, y_val_t), batch_size=batch_size, shuffle=False)

            # Dynamic edge index for main training
            edge_index, _ = st.session_state.mapper.export_edge_index()

            telemetry_status.info(" Initializing PyTorch Graph and Beginning Training...")

            model = QuantumSpatiotemporalGNN(in_channels=model_in_channels, seq_len=dynamic_seq_len, conv_out=c_conv,
                                             gat_heads=c_gat, lstm_hidden=c_lstm, dropout=c_drop)

            optimizer = optim.Adam(model.parameters(), lr=c_lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            # Graph-preserving imbalance handling: weight genuine positive bus labels in the
            # training loss. Ordinary SMOTE creates independent rows and cannot preserve the
            # 33-node graph topology or node-to-label alignment of this QSTGNN.
            train_label_values = np.asarray(st.session_state.y_train).reshape(-1)
            n_positive = int(np.sum(train_label_values == 1))
            n_negative = int(np.sum(train_label_values == 0))
            if n_positive == 0 or n_negative == 0:
                st.error(
                    "Training partition must contain both Safe and Failure labels. "
                    "Revisit the storm split or collect more labeled storm events."
                )
                st.stop()
            positive_weight = float(n_negative / max(n_positive, 1))
            criterion = nn.BCELoss(reduction="none")
            st.info(
                f"Training-only imbalance correction: positive-class loss weight = {positive_weight:.3f} "
                f"(Safe nodes: {n_negative:,}; Failure nodes: {n_positive:,}). "
                "Validation, calibration, and test labels are not resampled."
            )

            loss_history = pd.DataFrame(columns=['Train Loss', 'Val Loss'])
            acc_history = pd.DataFrame(columns=['Val Accuracy'])
            best_val_loss = float('inf')
            best_model_weights = None

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs, edge_index)

                    # Clamp outputs to prevent math errors
                    outputs_clamped = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)

                    target_values = targets.view_as(outputs_clamped)
                    element_losses = criterion(outputs_clamped, target_values)
                    element_weights = torch.where(
                        target_values >= 0.5,
                        torch.as_tensor(positive_weight, dtype=element_losses.dtype, device=element_losses.device),
                        torch.ones_like(element_losses)
                    )
                    loss = (element_losses * element_weights).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                model.eval()
                val_loss, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs, edge_index)
                        outputs_clamped = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)

                        v_loss = criterion(outputs_clamped, targets.view_as(outputs_clamped))
                        # BCELoss(reduction="none") returns one loss per node. Reduce
                        # element-wise losses to a scalar before accumulating batch loss.
                        v_loss = v_loss.mean()
                        val_loss += v_loss.item()

                        predicted = (outputs_clamped >= 0.5).float()
                        hard_targets = (targets >= 0.5).float()

                        total += (hard_targets.size(0) * hard_targets.size(1))
                        correct += (predicted == hard_targets.view_as(predicted)).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total

                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())

                new_loss = pd.DataFrame({'Train Loss': [avg_train_loss], 'Val Loss': [avg_val_loss]}, index=[epoch])
                new_acc = pd.DataFrame({'Val Accuracy': [val_accuracy]}, index=[epoch])
                loss_history = pd.concat([loss_history, new_loss])
                acc_history = pd.concat([acc_history, new_acc])

                loss_chart.line_chart(loss_history)
                acc_chart.line_chart(acc_history)
                train_progress.progress((epoch + 1) / epochs)

            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)

            os.makedirs("data/processed", exist_ok=True)
            torch.save(model.state_dict(), "data/processed/hybrid_dl_weights.pt")

            model.eval()
            with torch.no_grad():
                st.session_state.cal_probs = model(X_cal_q, edge_index).numpy()

            st.session_state.final_model_config = {
                'qkn_backend': st.session_state.get('qkn_backend', 'pennylane'),
                'q_qubits': q_qubits, 'q_layers': q_layers, 'q_entangle': q_entangle,
                'architecture': 'hybrid_qkn_ab_qstgnn' if hybrid_mode else 'qstgnn_expectation_a',
                'model_in_channels': int(model_in_channels),
                'c_conv': c_conv, 'c_gat': c_gat, 'c_lstm': c_lstm, 'c_drop': c_drop
            }
            st.session_state.model_trained = True
            telemetry_status.success(f" Training Complete! Best Val Loss ({best_val_loss:.4f}) locked and saved.")

        # --- QUANTUM KERNEL INSPECTOR ---
        if st.session_state.get('model_trained', False) and 'X_train_q_numpy' in st.session_state:
            st.divider()
            st.markdown("###  Quantum Kernel & Embedding Inspector")
            st.write(
                "Visualize how the Quantum Circuit embedded your topological features into the Hilbert space, and see the resulting **Gram Matrix (Quantum Kernel)** that gets fed into the Graph Neural Network.")

            q_col1, q_col2 = st.columns([1, 2])
            with q_col1:
                snap_q = st.slider("Select Training Snapshot:", 0, len(st.session_state.X_train_q_numpy) - 1, 0,
                                   key='q_snap')
                time_q = st.slider("Select Sequence Timestep:", 0, st.session_state.X_train_q_numpy.shape[2] - 1, 0,
                                   key='q_time')

            # Extract specific Embeddings: Shape -> (Nodes, Qubits)
            embeds = st.session_state.X_train_q_numpy[snap_q, :, time_q, :]

            # 1. Quantum Embedding Heatmap
            fig_emb = px.imshow(
                embeds.T,
                labels=dict(x="Grid Nodes (Buses)", y="Qubit Amplitude", color="Value"),
                title="Quantum Feature Embeddings (State Vector)",
                color_continuous_scale="Viridis",
                aspect="auto"
            )

            # 2. Quantum Kernel (Gram Matrix): K(x, x') = <phi(x), phi(x')>
            kernel_matrix = np.dot(embeds, embeds.T)
            fig_kernel = px.imshow(
                kernel_matrix,
                labels=dict(x="Grid Nodes", y="Grid Nodes", color="Similarity"),
                title="Quantum Kernel / Gram Matrix (Node-to-Node Hilbert Similarity)",
                color_continuous_scale="Plasma",
                aspect="auto"
            )

            vcol1, vcol2 = st.columns(2)
            vcol1.plotly_chart(fig_emb, use_container_width=True)
            vcol2.plotly_chart(fig_kernel, use_container_width=True)

            st.caption(
                "*Left: The raw amplitude projections of features onto the qubits. Right: The dot-product similarity between every node pair within the quantum feature space.*")

# --- TAB 4: PHASE 4 - CONFORMAL CALIBRATION ---
with tab4:
    st.markdown("###  Phase 4: Conformal Calibration")
    st.info(
        "Calculate the non-conformity threshold (q_hat) using the dedicated calibration partition. "
        "Keep this partition representative of the intended deployment distribution: do not downsample "
        "Safe nodes here just to balance classes, because that changes the calibration distribution."
    )

    if not st.session_state.get('model_trained', False):
        st.warning(" Please train the model in Tab 3 first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Conformal Parameters")
            alpha = st.slider("Target Error Rate (α):", min_value=0.01, max_value=0.20, value=0.10, step=0.01,
                              help="An α of 0.10 guarantees that the true grid state will be contained in the prediction set 90% of the time.")

            calibrate_btn = st.button("Calculate Threshold & Calibrate", type="primary", use_container_width=True)

            st.markdown("---")
            st.markdown("#### The Math")
            st.latex(r"\hat{q} = \text{Quantile}\left( s_1, ..., s_n, \frac{(n+1)(1-\alpha)}{n} \right)")
            st.latex(r"\mathcal{C}(x_{test}) = \{ y \in \mathcal{Y} : s(x_{test}, y) \le \hat{q} \}")

        with col2:
            st.markdown("#### Calibration Telemetry")
            cp_status = st.empty()

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Target Coverage", f"{(1 - alpha)*100:.1f}%")
            with metrics_col2:
                q_hat_metric = st.empty()

            st.caption("Non-Conformity Score Distribution (Calibration Set)")
            dist_chart = st.empty()

        if calibrate_btn:
            import numpy as np
            import pandas as pd
            import plotly.express as px

            try:
                cp_status.info(" Calculating Non-Conformity Scores on Calibration set...")

                # Retrieve saved calibration probabilities and labels from Tab 3
                cal_probs = st.session_state.cal_probs
                y_cal = st.session_state.y_cal

                # Flatten arrays to treat each node's state as an exchangeable calibration point
                probs_flat = cal_probs.flatten()
                y_cal_flat = y_cal.flatten()

                # Calculate non-conformity scores (1 - predicted probability of the TRUE class)
                prob_true_class = probs_flat * y_cal_flat + (1 - probs_flat) * (1 - y_cal_flat)
                scores = 1.0 - prob_true_class

                # Calculate the rigorous quantile threshold (q_hat)
                n = len(scores)
                q_level = np.ceil((n + 1) * (1 - alpha)) / n
                q_level = min(q_level, 1.0)
                q_hat = np.quantile(scores, q_level)

                # Explicitly save variables for Tab 5 to consume
                st.session_state.q_hat = q_hat
                st.session_state.qcp_calibrated = True

                # Update UI Metrics
                q_hat_metric.metric("Calibrated Threshold (q_hat)", f"{q_hat:.4f}")

                # Plot the calibration distribution
                df_scores = pd.DataFrame({'Non-Conformity Score': scores})
                fig_dist = px.histogram(df_scores, x='Non-Conformity Score', nbins=50,
                                        color_discrete_sequence=['#4C78A8'])
                fig_dist.add_vline(x=q_hat, line_dash="dash", line_color="red",
                                   annotation_text=f"q_hat = {q_hat:.3f}")
                dist_chart.plotly_chart(fig_dist, use_container_width=True)

                cp_status.success(" Calibration Complete! Threshold locked. Proceed to Phase 5 for Out-of-Sample Inference.")

            except Exception as e:
                cp_status.error(f" Calibration Error: {e}")

# --- TAB 5: PHASE 5 - INFERENCE TESTING ---
with tab5:
    st.write("Final Model Configuration:")
    st.write(st.session_state.get("final_model_config", {}))

    # Ensure keys exist even if not yet populated
    if 'y_pred_raw' not in st.session_state:
        st.session_state.y_pred_raw = None

    if 'y_pred_calibrated' not in st.session_state:
        st.session_state.y_pred_calibrated = None

    if 'y_pred_qcp' not in st.session_state:
        st.session_state.y_pred_qcp = None

    if not st.session_state.get('model_trained', False):
        st.error(" Model not found. Please complete Phase 3 first.")
    else:
        # Fallback to calculate q_hat if it wasn't saved in Tab 4
        if 'q_hat' not in st.session_state:
            cal_probs = st.session_state.cal_probs.flatten()
            y_cal_flat = st.session_state.y_cal.flatten()
            scores = 1.0 - (cal_probs * y_cal_flat + (1 - cal_probs) * (1 - y_cal_flat))
            st.session_state.q_hat = np.quantile(scores, 0.90)  # Default alpha = 0.10

        st.markdown("###  Phase 5: Out-of-Sample Inference Testing")
        st.warning(
            "Use the untouched held-out test partition for primary reported metrics. Do not remove Safe "
            "nodes from it to improve class balance. If a balanced diagnostic is added, report it separately "
            "as a secondary evaluation and retain the original test results."
        )
        st.write(
            f"Evaluating model reliability on unseen test data using the established conformal threshold $q_{{\hat{{h}}}}$ = **{st.session_state.q_hat:.4f}**.")

        # 1. EXECUTE INFERENCE
        if st.button("Execute Inference & Generate Metrics", type="primary"):
            import torch
            import numpy as np
            import pandas as pd
            from src.qkn import QuantumKernelNetwork, QuantumSpatiotemporalGNN
            from torch.utils.data import DataLoader, TensorDataset

            with st.spinner("Executing Hybrid Quantum-Graph Inference..."):
                config = st.session_state.final_model_config
                optimal_T = st.session_state.get('optimal_temperature', 1.0)
                st.session_state.qkn_backend = config.get('qkn_backend', 'pennylane')

                # Rebuild QKN
                qkn = create_qkn(st.session_state.get("qkn_backend", "pennylane"), n_qubits=config['q_qubits'], layers=config['q_layers'],
                                           entangling_type=config['q_entangle'])
                if config.get('architecture') == 'hybrid_qkn_ab_qstgnn':
                    from src.hybrid_qstgnn import HybridQSTGNNFeatureEncoder
                    encoder_state = torch.load(
                        "data/processed/hybrid_qstgnn_encoder.pt",
                        map_location="cpu",
                        weights_only=True
                    )
                    encoder = HybridQSTGNNFeatureEncoder.from_state_dict(encoder_state)
                    X_test_q = encoder.transform(st.session_state.X_seq_test, qkn)
                    model_in_channels = int(config.get('model_in_channels', encoder.output_channels))
                else:
                    X_test_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_test)
                    model_in_channels = int(config.get('model_in_channels', config['q_qubits']))

                st.write("### Feature Pipeline Diagnostics")

                st.write(
                    "Selected architecture:",
                    config.get("architecture")
                )

                st.write(
                    "Configured input channels:",
                    config.get("model_in_channels")
                )

                st.write(
                    "Quantum input shape:",
                    tuple(st.session_state.X_seq_test.shape)
                )

                st.write(
                    "Encoded test feature shape:",
                    tuple(X_test_q.shape)
                )

                st.write(
                    "Feature dtype:",
                    str(X_test_q.dtype)
                )

                st.write(
                    "Feature contains NaN:",
                    bool(torch.isnan(X_test_q).any().item())
                )

                st.write(
                    "Feature contains Inf:",
                    bool(torch.isinf(X_test_q).any().item())
                )

                # Rebuild QSTGNN with the same fused channel dimension used during training.
                model = QuantumSpatiotemporalGNN(
                    in_channels=model_in_channels, seq_len=4, conv_out=config['c_conv'],
                    gat_heads=config['c_gat'], lstm_hidden=config['c_lstm'], dropout=config['c_drop']
                )
                model.load_state_dict(torch.load("data/processed/hybrid_dl_weights.pt", weights_only=True))
                model.eval()

                edge_index, _ = st.session_state.mapper.export_edge_index()

                # Format Tensors
                y_test_tensor = torch.tensor(st.session_state.y_test, dtype=torch.float32)
                test_loader = DataLoader(TensorDataset(X_test_q, y_test_tensor), batch_size=16, shuffle=False)

                test_probs_list = []
                raw_probs_list = []  # NEW: Collect raw probabilities for Tab 6

                with torch.no_grad():
                    for inputs, _ in test_loader:
                        outputs = model(inputs, edge_index)

                        # Isolate Raw Output for physical thresholding
                        eps = 1e-7
                        out_clamped = torch.clamp(outputs, eps, 1.0 - eps)
                        raw_probs_list.append(out_clamped.numpy())

                        # Apply Temperature Scaling from Tab 3 (if exists, else optimal_T=1.0)
                        test_logits = torch.log(out_clamped / (1.0 - out_clamped))
                        scaled_probs = torch.sigmoid(test_logits / optimal_T)
                        test_probs_list.append(scaled_probs.numpy())

                # Flatten arrays because grid snapshots are shape (Snapshots, 33 Nodes)
                probs_flat = np.concatenate(test_probs_list, axis=0).flatten()
                raw_probs_flat = np.concatenate(raw_probs_list, axis=0).flatten()
                y_test_flat = st.session_state.y_test.flatten()

                # Convert calibrated probabilities into binary class predictions.
                # probs_flat contains the temperature-scaled probabilities.
                y_calibrated = (probs_flat >= 0.5).astype(int)

                # Save predictions for the Phase 5 evaluation section.
                st.session_state.y_pred_calibrated = y_calibrated
                
                # Apply QCP Sets
                q_hat = st.session_state.q_hat
                prediction_sets = []
                results = []
                ambiguous_count = 0

                for i, p in enumerate(probs_flat):
                    bus_id = (i % 33) + 1
                    snap_id = i // 33
                    s = []

                    if p <= q_hat: s.append("Safe")
                    if p >= (1 - q_hat): s.append("Failure")
                    if len(s) == 0: s = ["Safe", "Failure"]  # Safety Fallback for empty sets

                    prediction_sets.append(s)
                    if len(s) > 1: ambiguous_count += 1

                    results.append({
                        "Snapshot_ID": snap_id,
                        "Bus_ID": bus_id,
                        "Truth": y_test_flat[i],
                        "Prob_Fail": p,
                        "Raw_Prob_Fail": raw_probs_flat[i],  # NEW: Save raw outputs for Tab 6 physical slider
                        "Alert_Level": "High" if "Failure" in s and "Safe" not in s else (
                            "Uncertain" if len(s) > 1 else "Low")
                    })

                total_samples = len(results)
                certain_count = total_samples - ambiguous_count

                # Set coverage is distinct from the fail-safe binary decision metric:
                # it checks whether each true label is included in its conformal set.
                qcp_covered_count = sum(
                    (("Safe" if int(y) == 0 else "Failure") in pred_set)
                    for y, pred_set in zip(y_test_flat, prediction_sets)
                )
                qcp_coverage = (
                    qcp_covered_count / total_samples if total_samples else float("nan")
                )

                # Save to Session State
                st.session_state.final_results = pd.DataFrame(results)
                # Fix: Assess raw predictions using raw_probs_flat to avoid scaled distortion
                st.session_state.y_pred_raw = (raw_probs_flat >= 0.5).astype(int)

                # For safety systems: If it's uncertain, we flag it as a 1 (Failure expected) to trigger alarms
                st.session_state.y_pred_qcp = [1 if "Failure" in s else 0 for s in prediction_sets]
                st.session_state.ambiguous_count = ambiguous_count
                st.session_state.certain_count = certain_count
                st.session_state.total_test_samples = total_samples
                st.session_state.qcp_covered_count = qcp_covered_count
                st.session_state.qcp_empirical_coverage = qcp_coverage
                st.session_state.inference_complete = True
                st.rerun()

        # 2. VISUALIZATION & METRICS
        if st.session_state.get('inference_complete', False) and st.session_state.y_pred_qcp is not None:
            import plotly.express as px
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            y_test = st.session_state.y_test.flatten()

            y_raw = np.asarray(st.session_state.y_pred_raw).flatten()
            y_calibrated = np.asarray(
                st.session_state.y_pred_calibrated
            ).flatten()
            y_qcp = np.asarray(st.session_state.y_pred_qcp).flatten()


            def get_metrics(y_true, y_pred):
                return {
                    "Acc": accuracy_score(y_true, y_pred),
                    "Pre": precision_score(y_true, y_pred, zero_division=0),
                    "Rec": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0)
                }


            m_raw = get_metrics(y_test, y_raw)
            m_calibrated = get_metrics(y_test, y_calibrated)
            m_qcp = get_metrics(y_test, y_qcp)

            # Retain per-architecture test metrics in this Streamlit session so A-only
            # and A+B can be compared after each is trained/evaluated on the same split.
            active_config = st.session_state.get("final_model_config", {})
            active_architecture = active_config.get("architecture", "unknown")
            architecture_benchmark = st.session_state.get("architecture_benchmark", {})
            architecture_benchmark[active_architecture] = {
                "Architecture": active_architecture,
                "Recall": m_calibrated["Rec"],
                "Precision": m_calibrated["Pre"],
                "F1": m_calibrated["F1"],
                "Accuracy": m_calibrated["Acc"],
                "Test samples": int(len(y_test)),
                "Ground-truth failures": int(np.sum(np.asarray(y_test).reshape(-1) == 1)),
                "Threshold": 0.5,
            }
            st.session_state.architecture_benchmark = architecture_benchmark

            st.markdown("#### A-only vs A+B Architecture Benchmark")
            st.caption(
                "Compare calibrated binary predictions on the same held-out storm split. "
                "Prioritize Recall, then Precision, F1, and Accuracy. Re-run both architectures "
                "without changing the extracted dataset or split."
            )
            benchmark_df = pd.DataFrame(architecture_benchmark.values())
            if not benchmark_df.empty:
                benchmark_df = benchmark_df[
                    ["Architecture", "Recall", "Precision", "F1", "Accuracy",
                     "Test samples", "Ground-truth failures", "Threshold"]
                ].sort_values("Recall", ascending=False)
                st.dataframe(
                    benchmark_df.style.format({
                        "Recall": "{:.2%}", "Precision": "{:.2%}",
                        "F1": "{:.2%}", "Accuracy": "{:.2%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download A-only vs A+B Benchmark (CSV)",
                    data=benchmark_df.to_csv(index=False),
                    file_name="architecture_benchmark.csv",
                    mime="text/csv",
                    key="architecture_benchmark_download",
                )

            # --- PREDICTION CONFIDENCE ---
            st.markdown("####  Conformal Set Efficiency")

            c_count = st.session_state.certain_count
            a_count = st.session_state.ambiguous_count
            t_count = st.session_state.total_test_samples

            c_pct = (c_count / t_count) * 100
            a_pct = (a_count / t_count) * 100

            col_pie, col_stats = st.columns([1, 1])

            with col_pie:
                df_pie = pd.DataFrame({
                    "Confidence": ["Certain (Sure)", "Ambiguous (Uncertain)"],
                    "Count": [c_count, a_count]
                })
                fig_pie = px.pie(
                    df_pie, values='Count', names='Confidence', hole=0.5,
                    color='Confidence',
                    color_discrete_map={"Certain (Sure)": "#2ecc71", "Ambiguous (Uncertain)": "#f39c12"}
                )
                fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_stats:
                st.write("**Model Autonomy vs. Intervention**")
                st.metric(" Certain Predictions (Model is Sure)", f"{c_pct:.1f}%", f"{c_count} samples",
                          delta_color="normal")
                st.metric(" Ambiguous Sets (Needs Intervention)", f"{a_pct:.1f}%", f"{a_count} samples",
                          delta_color="inverse")
                st.metric(
                    "Empirical QCP Set Coverage",
                    f"{100 * st.session_state.get('qcp_empirical_coverage', 0.0):.2f}%",
                    f"{st.session_state.get('qcp_covered_count', 0)} of {t_count} true labels included"
                )
                st.caption(
                    "A well-calibrated model balances high reliability with a low ambiguity rate. Ambiguous sets trigger Phase 6 Islanding.")

            st.divider()

            # --- PERFORMANCE COMPARISON ---
            st.markdown("#### Performance Comparison")

            comparison_df = pd.DataFrame(
                [m_raw, m_calibrated, m_qcp],
                index=["Raw Model", "Calibrated Model", "QCP Fail-Safe"]
            )

            st.dataframe(
                comparison_df.style.format("{:.2%}"),
                use_container_width=True
            )


            def get_class_distribution(values, name):
                values = np.asarray(values).reshape(-1).astype(int)

                total = len(values)
                safe_count = int(np.sum(values == 0))
                fail_count = int(np.sum(values == 1))

                return {
                    "Dataset": name,
                    "Safe (0)": safe_count,
                    "Fail (1)": fail_count,
                    "Total": total,
                    "Safe (%)": round(100 * safe_count / total, 2) if total else 0.0,
                    "Fail (%)": round(100 * fail_count / total, 2) if total else 0.0,
                }


            y_test = np.asarray(st.session_state.y_test).reshape(-1)
            y_raw = np.asarray(st.session_state.y_pred_raw).reshape(-1)
            y_calibrated = np.asarray(
                st.session_state.y_pred_calibrated
            ).reshape(-1)
            y_qcp = np.asarray(st.session_state.y_pred_qcp).reshape(-1)

            st.markdown("### Class Distribution Diagnostics")

            distribution_df = pd.DataFrame([
                get_class_distribution(y_test, "Ground Truth"),
                get_class_distribution(y_raw, "Raw Model"),
                get_class_distribution(y_calibrated, "Calibrated Model"),
                get_class_distribution(y_qcp, "QCP Fail-Safe"),
            ])

            st.dataframe(distribution_df, use_container_width=True)

            # Verify that the number of predictions matches the number of labels.
            if not (
                    len(y_test) == len(y_raw) == len(y_calibrated) == len(y_qcp)
            ):
                st.error(
                    "Mismatch between the number of ground-truth labels "
                    "and prediction values."
                )
            else:
                st.success(
                    f"All prediction arrays contain {len(y_test):,} values."
                )

            # Check for unexpected label values.
            st.write("Ground-truth unique labels:", np.unique(y_test))
            st.write("Raw prediction unique labels:", np.unique(y_raw))
            st.write("Calibrated prediction unique labels:", np.unique(y_calibrated))
            st.write("QCP prediction unique labels:", np.unique(y_qcp))

            # --- CONFUSION MATRICES ---
            def plot_cm(cm, title, scale):
                return px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale=scale,
                    x=["Safe", "Failure"],
                    y=["Safe", "Failure"],
                    labels={
                        "x": "Predicted Class",
                        "y": "Actual Class",
                        "color": "Count"
                    },
                    title=title,
                    aspect="auto"
                )


            c1, c2, c3 = st.columns(3)

            c1.plotly_chart(
                plot_cm(
                    confusion_matrix(y_test, y_raw, labels=[0, 1]),
                    "Raw Model",
                    "Blues"
                ),
                use_container_width=True
            )

            c2.plotly_chart(
                plot_cm(
                    confusion_matrix(y_test, y_calibrated, labels=[0, 1]),
                    "Calibrated Model",
                    "Greens"
                ),
                use_container_width=True
            )

            c3.plotly_chart(
                plot_cm(
                    confusion_matrix(y_test, y_qcp, labels=[0, 1]),
                    "QCP Fail-Safe Model",
                    "Reds"
                ),
                use_container_width=True
            )

            # --- ALERT LEDGER ---
            st.markdown("####  Node Alert Report")
            df_res = st.session_state.final_results

            # Restrict displayed columns to avoid overwhelming the view, specifically exposing the Raw Probability
            st.dataframe(df_res[['Snapshot_ID', 'Bus_ID', 'Truth', 'Raw_Prob_Fail', 'Alert_Level']].style.map(
                lambda v: 'background-color: #ff9999;' if v == 'High' else
                ('background-color: #ffe066;' if v == 'Uncertain' else
                 'background-color: #b3ffb3;'),
                subset=['Alert_Level']), use_container_width=True)


            if st.button("Commit Alerts to CTQW Islanding Engine", type="primary"):
                st.session_state.islanding_alerts = df_res[df_res["Alert_Level"] != "Low"]

                # Extract risky buses dynamically for Phase 6
                risky_bus_list = df_res[df_res["Alert_Level"].isin(["High", "Uncertain"])]["Bus_ID"].tolist()
                st.session_state.risky_buses = list(set(risky_bus_list))

                st.success(" Alerts committed. Proceed to Phase 6 (Quantum Islanding).")


# --- TAB 6: PHASE 6 - QUANTUM WALK ISLANDING ---
with tab6:
    st.markdown("###  Phase 6: Post-Disaster Quantum Islanding")
    st.info(
        "Simulate grid fracture using the RAW inference probabilities from Phase 5. Adjust the threshold to explore different disaster severity scenarios. Analyzes cascading failures using Continuous-Time Quantum Walks (CTQW).")

    if not st.session_state.get('inference_complete', False):
        st.warning(" Please complete Out-of-Sample Inference in Tab 5 first to generate the storm probabilities.")
    else:
        col1, col2 = st.columns([1, 2])

        # Fetch the results computed in Tab 5
        df_res = st.session_state.final_results
        max_snap = int(df_res['Snapshot_ID'].max())

        with col1:
            st.markdown("#### Disaster Scenario Setup")

            # Snapshot Slider
            snapshot_idx = st.slider("Select Test Storm Snapshot:", 0, max_snap, 0,
                                     help="Slides through the timeline of the test set typhoons.")

            # Filter data for the selected snapshot
            snapshot_data = df_res[df_res['Snapshot_ID'] == snapshot_idx]

            # --- DYNAMIC UX UPGRADE: Probability Distribution Histogram ---
            st.markdown("#### Node Vulnerability Distribution")
            st.caption("Visualizing the raw probabilities to help you pick the perfect fracture threshold.")

            # Calculate a smart default (e.g., the 85th percentile of risk) to separate safe from fail
            smart_default = float(np.percentile(snapshot_data['Raw_Prob_Fail'], 85))

            fail_threshold = st.slider(
                "Raw Failure Probability Threshold:",
                min_value=0.00,
                max_value=1.00,
                value=min(max(smart_default, 0.01), 0.99),  # Safely bound the default
                step=0.01,
                help="If a bus's RAW predicted failure probability exceeds this threshold, it is considered destroyed."
            )

            # Draw a mini-histogram of the probabilities so the user can see where to put the slider
            fig_hist = px.histogram(
                snapshot_data, x="Raw_Prob_Fail", nbins=20,
                labels={"Raw_Prob_Fail": "Predicted Failure Probability"},
                color_discrete_sequence=["#9b59b6"]
            )
            fig_hist.add_vline(x=fail_threshold, line_dash="dash", line_color="red", annotation_text="Fracture Cutoff")
            fig_hist.update_layout(height=200, margin=dict(t=10, b=0, l=0, r=0), xaxis_title=None,
                                   yaxis_title="Bus Count")
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("#### Quantum Propagation Settings")
            time_evolution = st.slider("Quantum Evolution Time ($t$):", min_value=0.1, max_value=5.0, value=1.0,
                                       step=0.1,
                                       help="Controls how far the quantum cascade wave propagates during Buffer Zone analysis.")

            run_islanding_btn = st.button("Simulate Fracture & Map Islands", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Post-Disaster Microgrid Topology")
            island_plot = st.empty()
            metrics_display = st.empty()

        if run_islanding_btn:
            import networkx as nx
            import plotly.graph_objects as go
            import plotly.express as px
            import scipy.linalg as sl
            from src.mapping import QuantumWalkIslandingMapper

            try:
                with st.spinner("Fracturing topology and calculating quantum wave function spread..."):
                    mapper = st.session_state.mapper

                    # --- 1. DETERMINE FRACTURES (Using RAW Probs & User Threshold) ---
                    failed_nodes = snapshot_data[snapshot_data['Raw_Prob_Fail'] >= fail_threshold]['Bus_ID'].tolist()

                    failed_edges = []
                    for node in failed_nodes:
                        failed_edges.extend(list(mapper.graph.edges(node)))

                    post_disaster_grid = mapper.simulate_typhoon_failures(failed_edges)

                    # --- 2. QUANTUM WALK ISLANDING (PennyLane Macro Map) ---
                    q_mapper = QuantumWalkIslandingMapper(post_disaster_grid)
                    islands = q_mapper.identify_islands()

                    # --- 3. VISUALIZATION (Macro Map) ---
                    fig = go.Figure()

                    for u, v in post_disaster_grid.edges():
                        x0, y0 = mapper.bus_coords[u]
                        x1, y1 = mapper.bus_coords[v]
                        fig.add_trace(go.Scatter(
                            x=[x0, x1, None], y=[y0, y1, None],
                            mode='lines', line=dict(color='lightgrey', width=2), hoverinfo='none', showlegend=False
                        ))

                    colors = px.colors.qualitative.Set1
                    for i, (zone_id, zone_data) in enumerate(islands.items()):
                        color = colors[i % len(colors)]
                        nx_x = [mapper.bus_coords[n][0] for n in zone_data['nodes']]
                        nx_y = [mapper.bus_coords[n][1] for n in zone_data['nodes']]

                        fig.add_trace(go.Scatter(
                            x=nx_x, y=nx_y, mode='markers+text',
                            marker=dict(size=14, color=color, line=dict(width=1, color='DarkSlateGrey')),
                            text=[str(n) for n in zone_data['nodes']], textposition="top center",
                            name=f"{zone_id} ({zone_data['size']} buses)"
                        ))

                    if failed_nodes:
                        fx = [mapper.bus_coords[n][0] for n in failed_nodes]
                        fy = [mapper.bus_coords[n][1] for n in failed_nodes]
                        fig.add_trace(go.Scatter(
                            x=fx, y=fy, mode='markers',
                            marker=dict(size=12, color='black', symbol='x'),
                            name=f'Failed Buses (Raw P ≥ {fail_threshold:.2f})', hoverinfo='text',
                            text=[f"Bus {n}" for n in failed_nodes]
                        ))

                    fig.update_layout(
                        title=f"Post-Disaster Topology (Snapshot {snapshot_idx})",
                        showlegend=True, margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        plot_bgcolor='white'
                    )

                    island_plot.plotly_chart(fig, use_container_width=True)
                    metrics_display.success(
                        f"**Disaster Summary:** {len(failed_nodes)} buses failed. The grid fractured into **{len(islands)}** autonomous microgrid(s).")

                    # --- 4. QUANTUM CASCADE ANALYSIS (Deep Dive) ---
                    if failed_nodes:
                        st.divider()
                        st.markdown("####  Quantum Cascade & Buffer Zone Analysis")
                        st.write(
                            "Using Continuous-Time Quantum Walks (CTQW) on the intact Adjacency Matrix to calculate probability density and identify cascading risk zones before fracture.")

                        # Find the worst bus in this snapshot to act as the Primary Epicenter
                        worst_bus_row = \
                        snapshot_data[snapshot_data['Bus_ID'].isin(failed_nodes)].sort_values(by='Raw_Prob_Fail',
                                                                                              ascending=False).iloc[0]
                        target_bus = int(worst_bus_row['Bus_ID'])

                        col_qc1, col_qc2 = st.columns([1, 2])

                        with col_qc1:
                            st.metric("Primary Epicenter", f"Bus {target_bus}",
                                      f"Raw Fail Prob: {worst_bus_row['Raw_Prob_Fail']:.3f}", delta_color="inverse")
                            st.info(
                                f"The quantum state $|\psi(0)\\rangle$ is initialized at Bus {target_bus}. Evolving for $t={time_evolution}$...")

                            adj_matrix = nx.to_numpy_array(mapper.graph)
                            nodes_list = list(mapper.graph.nodes())

                            H = adj_matrix
                            U = sl.expm(-1j * H * time_evolution)

                            target_idx = nodes_list.index(target_bus)
                            psi_0 = np.zeros(len(nodes_list), dtype=complex)
                            psi_0[target_idx] = 1.0

                            psi_t = U @ psi_0
                            probs = np.abs(psi_t) ** 2

                            fracture_threshold = np.mean(probs) + np.std(probs)
                            buffer_zones = [n for i, n in enumerate(nodes_list) if
                                            probs[i] > fracture_threshold and n != target_bus]

                            st.json({
                                "Epicenter (Isolate)": target_bus,
                                "High-Risk Buffer Zones (Monitor/Shed)": buffer_zones
                            })

                        with col_qc2:
                            fig_ctqw = go.Figure(data=[go.Bar(
                                x=[f"Bus {n}" for n in nodes_list], y=probs,
                                marker_color='#1f77b4', marker_line_color='#0a4a75',
                                marker_line_width=1.5, opacity=0.8
                            )])

                            fig_ctqw.add_hline(y=fracture_threshold, line_dash="dash", line_color="red",
                                               annotation_text="Buffer Threshold (μ + σ)")

                            fig_ctqw.update_layout(
                                title=f"Quantum Probability Amplitude |ψ(t={time_evolution})|²",
                                xaxis_title="Microgrid Nodes", yaxis_title="Probability Density",
                                template="plotly_white", margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_ctqw, use_container_width=True)

            except Exception as e:
                island_plot.error(f" Islanding Error: {e}")


# --- INTEGRATED TAB 7: EVENT-LEVEL QKN BENCHMARKING ---
with tab7:
    st.markdown("### Unseen-Typhoon Generalization & QKN Benchmark")
    st.caption("Integrated evaluation workspace — no separate Streamlit process required.")
    st.markdown(
        "**A-only vs B-only vs A+B comparison.** Approach A uses QKN Pauli-Z expectation features; "
        "Approach B uses the quantum fidelity kernel; Hybrid A+B combines both. A classical RBF-SVC is "
        "included as a control. All approaches use the same train/test split and held-out storm groups. "
        "This tab evaluates the row-wise feature-map/SVC benchmark; the separate Phase 3 training workflow "
        "evaluates the graph-temporal model. Prioritize Recall, then Precision, F1, and Accuracy."
    )

    import json
    import tempfile
    from pathlib import Path
    from experiments.compare_qkn_approaches import run_comparison
    from sklearn.model_selection import GroupShuffleSplit

    eval_col1, eval_col2 = st.columns([1, 2])
    with eval_col1:
        eval_source = st.radio(
            "Evaluation dataset",
            ["Processed CSV", "Upload CSV"],
            key="integrated_eval_source"
        )
        uploaded_eval = None
        if eval_source == "Upload CSV":
            uploaded_eval = st.file_uploader(
                "Upload event-aware feature CSV",
                type=["csv"],
                key="integrated_eval_upload"
            )
            eval_path = None
        else:
            eval_path = st.text_input(
                "Processed CSV path",
                "data/processed/final_quantum_dataset.csv",
                key="integrated_eval_path"
            )

        eval_test_size = st.slider(
            "Held-out storm fraction", 0.10, 0.40, 0.25, 0.05,
            key="integrated_eval_test_size"
        )
        eval_max_samples = st.number_input(
            "Maximum rows for simulator runtime (0 = all)",
            min_value=0, max_value=10000, value=250, step=50,
            key="integrated_eval_max_samples"
        )
        eval_layers = st.slider(
            "Quantum circuit layers", 1, 4, 2,
            key="integrated_eval_layers"
        )
        eval_b_weight = st.slider(
            "Hybrid B-branch weight", 0.0, 1.0, 0.25, 0.05,
            key="integrated_eval_b_weight",
            help="Applied only to the B fidelity-similarity features in A+B. A-only and B-only results remain independent baselines."
        )
        eval_c = st.select_slider(
            "SVM regularization C",
            options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0,
            key="integrated_eval_c"
        )
        eval_seed = st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42,
            key="integrated_eval_seed"
        )

    eval_frame = None
    eval_source_name = ""
    if eval_source == "Upload CSV":
        if uploaded_eval is not None:
            try:
                eval_frame = pd.read_csv(uploaded_eval)
                eval_source_name = uploaded_eval.name
            except Exception as exc:
                st.error(f"Could not read uploaded CSV: {exc}")
    else:
        eval_file = Path(eval_path)
        if eval_file.exists():
            try:
                eval_frame = pd.read_csv(eval_file)
                eval_source_name = str(eval_file)
            except Exception as exc:
                st.error(f"Could not read dataset: {exc}")
        else:
            st.info(
                f"Dataset not found at `{eval_file}`. Complete extraction and export first, "
                "or choose Upload CSV."
            )

    if eval_frame is not None:
        st.write(
            f"Loaded **{len(eval_frame):,} rows** and **{len(eval_frame.columns)} columns** "
            f"from `{eval_source_name}`."
        )
        id_candidates = [
            c for c in eval_frame.columns
            if c.lower() in {"storm_id", "seq_id", "event_id", "typhoon_id", "storm", "event"}
        ]
        label_candidates = [
            c for c in eval_frame.columns
            if c.lower() in {"failure_label", "label", "target", "y"}
        ]
        with eval_col1:
            group_col = st.selectbox(
                "Typhoon/event ID",
                options=list(eval_frame.columns),
                index=(list(eval_frame.columns).index(id_candidates[0]) if id_candidates else 0),
                key="integrated_eval_group_col",
                help="Must identify the complete source typhoon, not a snapshot or bus."
            )
            label_col = st.selectbox(
                "Binary target",
                options=list(eval_frame.columns),
                index=(list(eval_frame.columns).index(label_candidates[0]) if label_candidates else 0),
                key="integrated_eval_label_col"
            )
            confirm_event_id = st.checkbox(
                "Confirm this column identifies the complete source typhoon/event",
                value=False,
                key="integrated_eval_confirm_event"
            )

        group_count = int(eval_frame[group_col].nunique(dropna=True))
        audit1, audit2, audit3 = st.columns(3)
        audit1.metric("Rows", f"{len(eval_frame):,}")
        audit2.metric("Distinct events", group_count)
        audit3.metric("Target classes", int(eval_frame[label_col].nunique(dropna=True)))
        if group_col == label_col:
            st.error("Event ID and target must be different columns.")
        elif group_count < 3:
            st.error("At least three distinct storms are required; more are strongly recommended.")
        else:
            with st.expander("Event audit", expanded=False):
                st.dataframe(
                    eval_frame.groupby(group_col, dropna=False).size()
                    .rename("rows").reset_index(),
                    use_container_width=True
                )
                st.warning(
                    "Storm-level separation is essential. Randomly splitting rows from the same "
                    "typhoon can overstate generalization performance."
                )

        excluded_cols = {group_col, label_col, "snapshot_id", "Snapshot_ID", "bus_id", "Bus_ID"}
        numeric_candidates = [
            c for c in eval_frame.select_dtypes(include=np.number).columns
            if c not in excluded_cols
        ]
        default_features = [c for c in numeric_candidates if c.endswith("(rad)")]
        if len(default_features) < 2:
            default_features = numeric_candidates[:3]
        features = st.multiselect(
            "Numeric model features (one qubit per selected feature; maximum 8)",
            numeric_candidates,
            default=default_features[:3],
            key="integrated_eval_features"
        )
        can_run_eval = (
            confirm_event_id
            and group_col != label_col
            and group_count >= 3
            and len(features) >= 2
            and len(features) <= 8
            and eval_frame[label_col].nunique(dropna=True) == 2
            and eval_frame[group_col].notna().all()
        )
        if len(features) > 8:
            st.warning("Select no more than eight features for this simulator-based benchmark.")

        if st.button(
            "Run storm-level QKN A/B/Hybrid benchmark",
            type="primary",
            disabled=not can_run_eval,
            key="integrated_eval_run"
        ):
            work = eval_frame.dropna(subset=features + [label_col, group_col]).copy()
            try:
                work[label_col] = work[label_col].astype(int)
            except (ValueError, TypeError):
                st.error("Target must be binary and encoded as 0/1.")
                work = None
            if work is not None and set(work[label_col].unique()) != {0, 1}:
                st.error("Target must contain both binary labels 0 and 1.")
                work = None

            if work is not None:
                if eval_max_samples and len(work) > int(eval_max_samples):
                    rng = np.random.RandomState(int(eval_seed))
                    event_ids = work[group_col].drop_duplicates().to_numpy()
                    rng.shuffle(event_ids)
                    selected_events, selected_rows = [], 0
                    for event_id in event_ids:
                        event_rows = int((work[group_col] == event_id).sum())
                        if selected_rows and selected_rows + event_rows > int(eval_max_samples):
                            continue
                        selected_events.append(event_id)
                        selected_rows += event_rows
                        if selected_rows >= int(eval_max_samples):
                            break
                    work = work[work[group_col].isin(selected_events)].copy()

                if work[group_col].nunique() < 3 or work[label_col].nunique() < 2:
                    st.error(
                        "The selected complete events do not contain enough storms or both labels. "
                        "Increase the row budget or use a dataset with more events."
                    )
                else:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", delete=False,
                        encoding="utf-8", newline=""
                    ) as tmp:
                        temp_path = tmp.name
                        work.to_csv(tmp, index=False)
                    try:
                        with st.spinner("Running quantum feature extraction and held-out storm evaluation..."):
                            benchmark = run_comparison(
                                data_path=temp_path,
                                output_path="outputs/qkn_unseen_typhoon_comparison.json",
                                features=features,
                                label=label_col,
                                test_size=float(eval_test_size),
                                random_state=int(eval_seed),
                                n_qubits=len(features),
                                C=float(eval_c),
                                group_column=group_col,
                                max_samples=0,
                                layers=int(eval_layers),
                                b_weight=float(eval_b_weight)
                            )
                        X = work[features].to_numpy(dtype=float)
                        y = work[label_col].to_numpy()
                        groups = work[group_col].to_numpy()
                        train_idx, test_idx = next(
                            GroupShuffleSplit(
                                n_splits=1,
                                test_size=float(eval_test_size),
                                random_state=int(eval_seed)
                            ).split(X, y, groups)
                        )
                        train_events = sorted(map(str, pd.unique(groups[train_idx])))
                        test_events = sorted(map(str, pd.unique(groups[test_idx])))
                        if set(train_events) & set(test_events):
                            raise RuntimeError("Leakage check failed: event IDs overlap.")
                        st.session_state["integrated_qkn_result"] = benchmark
                        st.session_state["integrated_qkn_split"] = {
                            "train_events": train_events,
                            "test_events": test_events,
                            "train_rows": int(len(train_idx)),
                            "test_rows": int(len(test_idx))
                        }
                        st.success("Benchmark complete; train/test event IDs are disjoint.")
                    except Exception as exc:
                        st.exception(exc)
                    finally:
                        Path(temp_path).unlink(missing_ok=True)

        if "integrated_qkn_result" in st.session_state:
            benchmark = st.session_state["integrated_qkn_result"]
            split = st.session_state["integrated_qkn_split"]
            st.subheader("Held-out storm results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Training storms", len(split["train_events"]))
            r2.metric("Held-out storms", len(split["test_events"]))
            r3.metric("Held-out rows", split["test_rows"])
            with st.expander("Show train/test storm IDs"):
                st.write("**Training:**", ", ".join(split["train_events"]))
                st.write("**Testing:**", ", ".join(split["test_events"]))

            metric_table = pd.DataFrame([
                {"model": model_name, **{
                    key: value for key, value in values.items()
                    if key != "per_storm_metrics"
                }}
                for model_name, values in benchmark["models"].items()
            ])
            st.markdown("#### Pooled metrics")
            st.dataframe(metric_table, use_container_width=True)
            preferred_metrics = [
                col for col in ["recall", "precision", "f1", "accuracy"]
                if col in metric_table.columns
            ]
            if preferred_metrics:
                st.plotly_chart(
                    px.bar(
                        metric_table, x="model", y=preferred_metrics,
                        barmode="group", title="Held-out storm performance"
                    ),
                    use_container_width=True
                )

            storm_rows = []
            for model_name, values in benchmark["models"].items():
                for storm_id, scores in values.get("per_storm_metrics", {}).items():
                    storm_rows.append({"model": model_name, "storm_id": storm_id, **scores})
            if storm_rows:
                storm_metrics = pd.DataFrame(storm_rows)
                st.markdown("#### Per-storm metrics")
                st.dataframe(storm_metrics, use_container_width=True)
                st.download_button(
                    "Download per-storm metrics",
                    storm_metrics.to_csv(index=False),
                    file_name="qkn_per_storm_metrics.csv",
                    mime="text/csv",
                    key="integrated_eval_download_storm"
                )
            st.download_button(
                "Download benchmark JSON",
                json.dumps(benchmark, indent=2),
                file_name="qkn_unseen_typhoon_comparison.json",
                mime="application/json",
                key="integrated_eval_download_json"
            )
            st.download_button(
                "Download pooled metrics CSV",
                metric_table.to_csv(index=False),
                file_name="qkn_unseen_typhoon_metrics.csv",
                mime="text/csv",
                key="integrated_eval_download_metrics"
            )
    else:
        st.info(
            "Load a processed CSV containing a true storm/event identifier and binary target. "
            "The updated extraction/export workflow should preserve storm_id, snapshot_id, and bus_id."
        )
