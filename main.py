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

warnings.filterwarnings('ignore')

# Custom Modules
from src.mapping import ChiayiMicrogridMapper, QuantumWalkIslandingMapper
from src.utils import haversine, rankine_vortex, vulnerability_curve
from src.qkn import QuantumSpatiotemporalGNN
from src.qkn_backend import create_qkn
from src.qcp import QuantumConformalPredictor

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Phase 1: Geo-Extraction",
    " Phase 2: Global Optimization",
    " Phase 3: Quantum Training",
    " Phase 4: Uncertainty Calibration",
    " Phase 5: Inference Testing",
    " Phase 6: CTQW Islanding"
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
        ids = int(st.slider("How Many Typhoons to Process:", 1, 10, 3))
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
                                'features': np.array(snapshot_features),
                                'labels': np.array(snapshot_labels),
                                'has_stress': snapshot_has_stress
                            })

                    # --- STRATIFIED SNAPSHOT SELECTION ---
                    # Separate snapshots into stressful (contains at least one failure) and safe
                    df_snapshots = pd.DataFrame(grid_snapshots)
                    df_stress = df_snapshots[df_snapshots['has_stress'] == True]
                    df_safe = df_snapshots[df_snapshots['has_stress'] == False]

                    # Ensure we capture grid failures
                    if df_stress.empty:
                        st.warning("Typhoons were too weak. Inducing synthetic coastal stress on final snapshot.")
                        # Induce synthetic failure on coastal buses (18, 22, 25, 33) in the last snapshot
                        synth_idx = df_snapshots.index[-1]
                        df_snapshots.at[synth_idx, 'labels'][[17, 21, 24, 32]] = 1
                        df_stress = df_snapshots.iloc[[synth_idx]]
                        df_safe = df_snapshots.drop(index=synth_idx)

                    k_stress = min(len(df_stress), target_snapshots // 2)
                    k_safe = target_snapshots - k_stress

                    selected_stress = df_stress.sample(n=k_stress, random_state=42)
                    selected_safe = df_safe.sample(n=k_safe, random_state=42) if not df_safe.empty else pd.DataFrame()

                    final_snapshots = pd.concat([selected_stress, selected_safe]).sample(frac=1,
                                                                                         random_state=42).reset_index(
                        drop=True)

                    # Store as numpy arrays.
                    # X_tensor shape: (Snapshots, 33 Nodes, 4 Timesteps, 4 Features)
                    # Y_tensor shape: (Snapshots, 33 Nodes)
                    st.session_state.X_grid_tensor = np.stack(final_snapshots['features'].values)
                    st.session_state.Y_grid_tensor = np.stack(final_snapshots['labels'].values)

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
                train_b, cal_b = int(0.6 * n_samples), int(0.8 * n_samples)

                # Split data by Full Grid Snapshots
                st.session_state.X_seq_train = X_tensor[:train_b]
                st.session_state.y_train = Y_tensor[:train_b]

                st.session_state.X_seq_cal = X_tensor[train_b:cal_b]
                st.session_state.y_cal = Y_tensor[train_b:cal_b]

                st.session_state.X_seq_test = X_tensor[cal_b:]
                st.session_state.y_test = Y_tensor[cal_b:]

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
                df_export["Failure_Label"] = Y_flat

                os.makedirs("data/processed", exist_ok=True)
                df_export.to_csv("data/processed/final_quantum_dataset.csv", index=False)

                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    df_safe = df_export[df_export["Failure_Label"] == 0]
                    df_fail = df_export[df_export["Failure_Label"] == 1]

                    mean_safe = df_safe.iloc[:, :-1].mean().values.tolist() if not df_safe.empty else [0] * len(
                        selected_features)
                    mean_fail = df_fail.iloc[:, :-1].mean().values.tolist() if not df_fail.empty else [0] * len(
                        selected_features)

                    mean_safe += [mean_safe[0]] if mean_safe else []
                    mean_fail += [mean_fail[0]] if mean_fail else []
                    radial_cols = [f"{feat} (rad)" for feat in selected_features] + [f"{selected_features[0]} (rad)"]

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
                X_cal_raw = st.session_state.X_seq_cal[:proxy_samples]
                y_cal_raw = st.session_state.y_cal[:proxy_samples]

                y_proxy = torch.tensor(y_train_raw, dtype=torch.float32)
                y_cal_t = torch.tensor(y_cal_raw, dtype=torch.float32)

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
                    X_cal_q = qkn.extract_temporal_quantum_features(X_cal_raw)

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
                        val_out = model(X_cal_q, edge_index)
                        # FIX: view_as() applied to validation as well
                        val_loss = criterion(val_out, y_cal_t.view_as(val_out)).item()

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
    st.markdown("###  Phase 3: Quantum-LSTM Training & Hyperparameter Tuning")
    st.info(
        "Configure the network architecture, embed the classical data into a Hilbert space, and train the PyTorch Graph Network.")

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

            st.markdown("#### Step 2: Quantum Backend")
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
                    edge_idx = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                                            dtype=torch.long).t()

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
            telemetry_status.info(" Extracting full dataset through Quantum Kernel Network...")

            qkn = create_qkn(st.session_state.get("qkn_backend", "pennylane"), n_qubits=q_qubits, layers=q_layers, entangling_type=q_entangle)
            X_train_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_train)
            X_cal_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_cal)

            # Save Quantum Embeddings for Visualization
            st.session_state.X_train_q_numpy = X_train_q.detach().numpy()

            # Apply label smoothing to prevent 0.000 logit saturation loss
            y_train_smooth = np.clip(st.session_state.y_train, 0.05, 0.95)
            y_cal_smooth = np.clip(st.session_state.y_cal, 0.05, 0.95)

            y_train_t = torch.tensor(y_train_smooth, dtype=torch.float32)
            y_cal_t = torch.tensor(y_cal_smooth, dtype=torch.float32)

            train_loader = DataLoader(TensorDataset(X_train_q, y_train_t), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_cal_q, y_cal_t), batch_size=batch_size, shuffle=False)

            # Dynamic edge index for main training
            edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                                      dtype=torch.long).t()

            telemetry_status.info(" Initializing PyTorch Graph and Beginning Training...")

            model = QuantumSpatiotemporalGNN(in_channels=q_qubits, seq_len=dynamic_seq_len, conv_out=c_conv,
                                             gat_heads=c_gat, lstm_hidden=c_lstm, dropout=c_drop)

            optimizer = optim.Adam(model.parameters(), lr=c_lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.BCELoss()

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

                    loss = criterion(outputs_clamped, targets.view_as(outputs_clamped))
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
    st.info("Calculate the Non-Conformity Threshold (q_hat) using the calibration dataset. This threshold guarantees our target error rate during out-of-sample inference.")

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
    # Ensure keys exist even if not yet populated
    if 'y_pred_raw' not in st.session_state: st.session_state.y_pred_raw = None
    if 'y_pred_qcp' not in st.session_state: st.session_state.y_pred_qcp = None

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
                X_test_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_test)

                # Rebuild GNN
                model = QuantumSpatiotemporalGNN(
                    in_channels=config['q_qubits'], seq_len=4, conv_out=config['c_conv'],
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

                # Save to Session State
                st.session_state.final_results = pd.DataFrame(results)
                # Fix: Assess raw predictions using raw_probs_flat to avoid scaled distortion
                st.session_state.y_pred_raw = (raw_probs_flat >= 0.5).astype(int)

                # For safety systems: If it's uncertain, we flag it as a 1 (Failure expected) to trigger alarms
                st.session_state.y_pred_qcp = [1 if "Failure" in s else 0 for s in prediction_sets]
                st.session_state.ambiguous_count = ambiguous_count
                st.session_state.certain_count = certain_count
                st.session_state.total_test_samples = total_samples
                st.session_state.inference_complete = True
                st.rerun()

        # 2. VISUALIZATION & METRICS
        if st.session_state.get('inference_complete', False) and st.session_state.y_pred_qcp is not None:
            import plotly.express as px
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            y_test = st.session_state.y_test.flatten()
            y_raw = st.session_state.y_pred_raw
            y_qcp = st.session_state.y_pred_qcp


            def get_metrics(y_true, y_pred):
                return {
                    "Acc": accuracy_score(y_true, y_pred),
                    "Pre": precision_score(y_true, y_pred, zero_division=0),
                    "Rec": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0)
                }


            m_raw = get_metrics(y_test, y_raw)
            m_qcp = get_metrics(y_test, y_qcp)

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
                st.caption(
                    "A well-calibrated model balances high reliability with a low ambiguity rate. Ambiguous sets trigger Phase 6 Islanding.")

            st.divider()

            # --- PERFORMANCE COMPARISON ---
            st.markdown("####  Performance Comparison")
            cols = st.columns(4)
            metrics = ["Acc", "Pre", "Rec", "F1"]
            for i, name in enumerate(metrics):
                delta = (m_qcp[name] - m_raw[name]) * 100
                cols[i].metric(name, f"{m_qcp[name]:.2%}", f"{delta:+.2f}%")


            # --- CONFUSION MATRICES ---
            def plot_cm(cm, title, scale):
                return px.imshow(cm, text_auto=True, color_continuous_scale=scale,
                                 x=['Safe', 'Fail'], y=['Safe', 'Fail'], title=title)


            c1, c2 = st.columns(2)
            c1.plotly_chart(plot_cm(confusion_matrix(y_test, y_raw), "Raw Model", "Blues"), use_container_width=True)
            c2.plotly_chart(plot_cm(confusion_matrix(y_test, y_qcp), "QCP-FailSafe Model", "Reds"),
                            use_container_width=True)

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
