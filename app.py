import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import LineString
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pennylane as qml
import os
import torch
import warnings

warnings.filterwarnings('ignore')

# Custom Modules
from src.mapping import ChiayiMicrogridMapper, QuantumWalkIslandingMapper
from src.utils import haversine, rankine_vortex, vulnerability_curve, circuit_html
from src.qkn import QuantumTemporalConvNet, QuantumKernelNetwork, ScaledQuantumTemporalConvNet
from src.qcp import QuantumConformalPredictor


# --- HELPER: CSV EXPORT ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to a UTF-8 encoded CSV."""
    return df.to_csv(index=True).encode('utf-8')


# --- PAGE CONFIG ---
st.set_page_config(page_title="Q-Rating Chaos", layout="wide")
st.title("⚡ Q-Rating Chaos: A Tri-Partite Quantum Framework for Typhoon Modeling and Microgrid Resilience")
st.subheader("ⓒ Engr. D.J. Medina 2026")
st.markdown(
    "Interactive POC: Quantum Kernel Networks, Conformal Prediction, and Quantum Walks for Typhoon Risk Modeling in Chiayi, Taiwan.")


# --- HELPER: GEOSPATIAL MAPBOX VISUALIZATION ---
def plot_interactive_map(mapper, title="Microgrid Topology", highlighted_nodes=None, node_colors=None):
    """Converts the NetworkX graph into an interactive Plotly Mapbox overlay."""
    edge_lon, edge_lat = [], []
    for edge in mapper.graph.edges():
        x0, y0 = mapper.bus_coords[edge[0]].x, mapper.bus_coords[edge[0]].y
        x1, y1 = mapper.bus_coords[edge[1]].x, mapper.bus_coords[edge[1]].y
        edge_lon.extend([x0, x1, None])
        edge_lat.extend([y0, y1, None])

    edge_trace = go.Scattermapbox(
        lon=edge_lon, lat=edge_lat,
        mode='lines', line=dict(width=2, color='#555'), hoverinfo='none'
    )

    node_lon, node_lat, node_text, node_color_list = [], [], [], []
    for node in mapper.graph.nodes():
        node_lon.append(mapper.bus_coords[node].x)
        node_lat.append(mapper.bus_coords[node].y)
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
                            style="carto-positron",  # Open-source street map, no API key needed
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
st.sidebar.header("Pipeline Controls")
# n_samples = st.sidebar.slider("Sample Size (POC Speed)", 50, 500, 150)
qkn_qubits = st.sidebar.selectbox("QKN Qubits", [1, 2, 3, 4, 5], index=0)
qkn_layers = st.sidebar.slider("QKN Entangling Layers", 1, 5, 2)
target_coverage = st.sidebar.slider("QCP Target Coverage (%)", 80, 99, 90) / 100.0

# --- TAB NAVIGATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Phase 1: Geo-Extraction",
    "⚛️ Phase 2: Quantum Training",
    "🛡️ Phase 3: Uncertainty Calibration",
    "🧪 Phase 4: Inference Testing",
    "🌊 Phase 5: CTQW Islanding"
])

# --- TAB 1: MAPPING & DATA EXTRACTION ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_interactive_map(st.session_state.mapper, "Enhanced IEEE 33-Bus System in Chiayi"),
                        use_container_width=True)
    with col2:
        st.info(
            "**Geospatial Setup**\n\nThe 33-bus system is mapped to Chiayi. Coastal exposure increases vulnerability for buses further West.")

        if st.button("Extract Historical Typhoon & Map Vulnerability"):
            with st.spinner("Applying Rankine Vortex, Fragility Models, and K-Means Distillation"):
                import pandas as pd
                from sklearn.cluster import KMeans
                from sklearn.metrics import pairwise_distances_argmin
                import numpy as np

                try:
                    # 1. LOAD RAW DATA
                    df = pd.read_csv("data/raw/typhoon_data.csv")

                    df_tw = df[(df['lat'] >= 21) & (df['lat'] <= 26) &
                               (df['lng'] >= 118) & (df['lng'] <= 123)].copy()

                    if df_tw.empty:
                        st.error("⚠️ No historical typhoons found near Taiwan in this dataset.")
                        st.stop()

                    taiwan_seq_ids = df_tw['seq_id'].unique()[3:]
                    df_recent_tw = df_tw[df_tw['seq_id'].isin(taiwan_seq_ids)]

                    top_storm_ids = df_recent_tw.groupby('seq_id')['wind'].max().nlargest(5).index
                    df_events = df_recent_tw[df_recent_tw['seq_id'].isin(top_storm_ids)].copy()
                    df_events = df_events.sort_index()

                    chiayi_lat, chiayi_lng = 23.48, 120.44
                    coastline_lng = 120.15

                    np.random.seed(42)
                    bus_coords = {i: (chiayi_lat + np.random.uniform(-0.1, 0.1),
                                      chiayi_lng + np.random.uniform(-0.1, 0.1)) for i in range(1, 34)}

                    records = []
                    time_steps = 4  # Target sliding window size for LSTM

                    # 6. SPATIOTEMPORAL PHYSICAL MAPPING (TEMPORAL CORESET EXTRACTION)
                    for seq_id, storm_track in df_events.groupby('seq_id'):

                        bus_failed_state = {i: 0 for i in range(1, 34)}
                        bus_feature_history = {i: [] for i in range(1, 34)}

                        for _, row in storm_track.iterrows():
                            ty_lat, ty_lng = row['lat'], row['lng']
                            v_max = row['wind'] * 0.51444
                            storm_grade = row['grade']

                            for bus_id in range(1, 34):
                                bus_lat, bus_lng = bus_coords[bus_id]

                                dist_km = haversine(ty_lat, ty_lng, bus_lat, bus_lng)
                                instantaneous_wind = rankine_vortex(v_max, dist_km)
                                coastal_exposure = max(0, 1 - (bus_lng - coastline_lng))

                                # Append current state to timeline
                                bus_feature_history[bus_id].append(
                                    [instantaneous_wind, storm_grade, dist_km, coastal_exposure])

                                # Only extract if we have a full temporal window (4 steps)
                                if len(bus_feature_history[bus_id]) >= time_steps:
                                    window = bus_feature_history[bus_id][-time_steps:]

                                    # Sustained wind is the average across the 4-step window
                                    sustained_wind = np.mean([step[0] for step in window])

                                    if bus_failed_state[bus_id] == 1:
                                        label = 1
                                        fail_prob = 1.0
                                    else:
                                        noisy_sustained = sustained_wind + np.random.normal(0, 2.0)
                                        fail_prob = vulnerability_curve(noisy_sustained)

                                        if fail_prob >= 0.60:
                                            label = 1
                                            bus_failed_state[bus_id] = 1
                                        else:
                                            label = 0

                                    records.append({
                                        'Bus_ID': bus_id,
                                        'Storm_ID': seq_id,
                                        'Wind_Speed': sustained_wind,  # 2D Representation
                                        'Storm_Grade': storm_grade,
                                        'Distance_to_Eye': dist_km,
                                        'Coastal_Exposure': coastal_exposure,
                                        'Failure_Label': label,
                                        'Raw_Prob': fail_prob,
                                        'Sequence': window  # 3D Tensor Representation for LSTM
                                    })

                    df_mapped = pd.DataFrame(records)

                    # 6.5 MARGIN CLEARING
                    df_clean = df_mapped[(df_mapped['Raw_Prob'] <= 0.40) | (df_mapped['Raw_Prob'] >= 0.60)].copy()

                    # 7. MATHEMATICAL CORESET DISTILLATION
                    st.toast("Clustering historical data to build Quantum Coresets", icon="⚛️")
                    target_qpu_budget = 400
                    feature_cols_all = ['Wind_Speed', 'Storm_Grade', 'Distance_to_Eye', 'Coastal_Exposure']

                    if len(df_clean) > target_qpu_budget:
                        df_fails = df_clean[df_clean['Failure_Label'] == 1]
                        df_safes = df_clean[df_clean['Failure_Label'] == 0]

                        k_fails = min(len(df_fails), target_qpu_budget // 2)
                        k_safes = target_qpu_budget - k_fails

                        if k_safes > 0 and len(df_safes) > 0:
                            kmeans_safe = KMeans(n_clusters=k_safes, random_state=42, n_init='auto')
                            kmeans_safe.fit(df_safes[feature_cols_all])
                            safe_indices = pairwise_distances_argmin(kmeans_safe.cluster_centers_,
                                                                     df_safes[feature_cols_all])
                            distilled_safes = df_safes.iloc[safe_indices]
                        else:
                            distilled_safes = df_safes

                        if k_fails > 0 and len(df_fails) > 0:
                            kmeans_fail = KMeans(n_clusters=k_fails, random_state=42, n_init='auto')
                            kmeans_fail.fit(df_fails[feature_cols_all])
                            fail_indices = pairwise_distances_argmin(kmeans_fail.cluster_centers_,
                                                                     df_fails[feature_cols_all])
                            distilled_fails = df_fails.iloc[fail_indices]
                        else:
                            distilled_fails = df_fails

                        df_final = pd.concat([distilled_fails, distilled_safes]).sample(frac=1, random_state=42)
                    else:
                        df_final = df_clean.sample(frac=1, random_state=42)

                    st.session_state.df_final_distilled = df_final
                    st.session_state.feature_cols_all = feature_cols_all
                    st.session_state.coresets_generated = True
                    st.success(
                        f"Distilled dataset to {len(df_final)} representative Temporal Coresets. Please proceed to Feature Selection below.")

                except FileNotFoundError:
                    st.error("⚠️ Could not find 'data/raw/typhoon_data.csv'. Please check the file path.")
                except Exception as e:
                    st.error(f"⚠️ An error occurred during data processing: {e}")

    # --- FEATURE SELECTION & CORRELATION DASHBOARD ---
    if st.session_state.get('coresets_generated', False):
        st.divider()
        st.markdown("### 🎯 Information Bottleneck (Correlation & Feature Selection)")
        st.write(
            "Isolate high-impact meteorological signals and remove redundant collinear features before scaling data into the quantum Hilbert space.")

        import pandas as pd
        import plotly.express as px
        from sklearn.ensemble import RandomForestClassifier

        df_final = st.session_state.df_final_distilled
        feature_cols_all = st.session_state.feature_cols_all

        col_corr, col_gini = st.columns(2)

        with col_corr:
            st.markdown("#### Pearson Correlation Matrix")
            corr_df = df_final[feature_cols_all + ['Failure_Label']].corr()
            fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", zmin=-1,
                                 zmax=1)
            fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_gini:
            st.markdown("#### Random Forest Gini Importance")
            rf = RandomForestClassifier(n_estimators=150, random_state=42)
            rf.fit(df_final[feature_cols_all], df_final['Failure_Label'])

            df_imp = pd.DataFrame({'Feature': feature_cols_all, 'Importance': rf.feature_importances_}).sort_values(
                by='Importance', ascending=True)

            fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance',
                             color_continuous_scale='Blues')
            fig_imp.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Select Quantum Embedding Features")
        st.caption(
            "Limiting variables prevents angular overlap and breaks symmetric manifold rings in Hilbert Space projections.")

        sorted_features = df_imp.sort_values(by='Importance', ascending=False)['Feature'].tolist()

        selected_features = []
        checkbox_cols = st.columns(len(sorted_features))

        for i, feat in enumerate(sorted_features):
            with checkbox_cols[i]:
                if st.checkbox(feat, value=(i < 3)):
                    selected_features.append(feat)

        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features to form a valid Hilbert space projection.")
        else:
            st.info(f"🚀 **Active Quantum Embedding Features:** {', '.join(selected_features)}")

            if st.button("Finalize Features & Prepare Quantum Dataset"):
                from sklearn.preprocessing import MinMaxScaler
                import numpy as np
                import os

                st.session_state.feature_cols = selected_features

                # 1. Prepare 2D Matrix (For QSVM & UI)
                X_raw = df_final[selected_features].values
                Y = df_final['Failure_Label'].values
                bus_ids = df_final['Bus_ID'].values

                scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
                X = scaler.fit_transform(X_raw)

                # 2. Prepare 3D Sequence Tensor (For PyTorch LSTM)
                seq_raw = np.array(df_final['Sequence'].tolist())  # Shape: (N, 4, 4)
                selected_indices = [feature_cols_all.index(f) for f in selected_features]
                seq_filtered = seq_raw[:, :, selected_indices]  # Shape: (N, 4, len(selected_features))

                # Scale sequences using the fitted scaler
                seq_flat = seq_filtered.reshape(-1, len(selected_features))
                seq_scaled_flat = scaler.transform(seq_flat)
                X_seq = seq_scaled_flat.reshape(-1, 4, len(selected_features))

                # Final shuffle keeping sequences and 2D arrays aligned
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X, X_seq, Y, bus_ids = X[indices], X_seq[indices], Y[indices], bus_ids[indices]

                # Split
                n_samples = len(X)
                train_b, cal_b = int(0.6 * n_samples), int(0.8 * n_samples)

                st.session_state.X_train, st.session_state.y_train = X[:train_b], Y[:train_b]
                st.session_state.X_cal, st.session_state.y_cal = X[train_b:cal_b], Y[train_b:cal_b]
                st.session_state.X_test, st.session_state.y_test = X[cal_b:], Y[cal_b:]

                # Save Sequences for PyTorch
                st.session_state.X_seq_train = X_seq[:train_b]
                st.session_state.X_seq_cal = X_seq[train_b:cal_b]
                st.session_state.X_seq_test = X_seq[cal_b:]

                st.session_state.bus_train = bus_ids[:train_b]
                st.session_state.bus_cal = bus_ids[train_b:cal_b]
                st.session_state.bus_test = bus_ids[cal_b:]

                st.session_state.data_loaded = True

                # --- AUTO-SAVE TO DISK ---
                os.makedirs("data/processed", exist_ok=True)
                export_cols = [f"{feat} (rad)" for feat in selected_features]
                df_export = pd.DataFrame(X, columns=export_cols)
                df_export.insert(0, "Bus_ID", bus_ids)
                df_export["Failure_Label"] = Y
                df_export.to_csv("data/processed/final_quantum_dataset.csv", index=False)

                num_safe = np.sum(Y == 0)
                num_fail = np.sum(Y == 1)
                st.success(
                    f"Quantum Matrices Prepared and Auto-Saved to `data/processed/`. Training Distribution: **{num_safe} Safe** | **{num_fail} Failures**")

    # --- DATASET VISUALIZATION ---
    if st.session_state.get('data_loaded', False):
        st.divider()
        st.markdown("### 📊 Distilled Quantum Dataset Profiles")

        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go

        actual_num_cols = st.session_state.X_train.shape[1]
        saved_features = st.session_state.get('feature_cols', [])

        col_headers = []
        theta_vals = []

        for i in range(actual_num_cols):
            if i < len(saved_features):
                feat_name = saved_features[i]
            else:
                feat_name = f"Feature_{i + 1}"

            col_headers.append(f"{feat_name} (rad)")
            theta_vals.append(feat_name.split('_')[0])

        df_train = pd.DataFrame(st.session_state.X_train, columns=col_headers)

        df_train.insert(0, "Bus ID", st.session_state.bus_train)
        df_train.insert(0, "Instance ID", [f"Train_{i}" for i in range(len(df_train))])
        df_train["Failure Label"] = st.session_state.y_train
        df_train["Status"] = df_train["Failure Label"].apply(lambda x: "🔴 Failure" if x == 1 else "🟢 Safe")

        col_table, col_vis = st.columns([2, 1])

        with col_table:
            format_dict = {col: "{:.4f}" for col in col_headers}
            st.dataframe(df_train.drop(columns=["Failure Label"]).style.format(format_dict), height=350,
                         use_container_width=True)

        with col_vis:
            st.markdown("#### Dynamic Feature Profile")
            num_samples = len(st.session_state.X_train)
            sample_options = [f"Train_{i}" for i in range(num_samples)]

            selected_sample_str = st.selectbox("Select Sample to View:", options=sample_options, index=0)
            selected_idx = int(selected_sample_str.split("_")[1])

            sample = st.session_state.X_train[selected_idx]
            label = st.session_state.y_train[selected_idx]
            bus_id = st.session_state.bus_train[selected_idx]

            color = "#d62728" if label == 1 else "#1f77b4"

            r_vals = sample.tolist() + [sample[0]]
            chart_theta = theta_vals.copy()
            chart_theta.append(chart_theta[0])

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=r_vals, theta=chart_theta, fill='toself', fillcolor=color, opacity=0.5,
                line=dict(color=color, width=2), name=selected_sample_str
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-np.pi, np.pi], showticklabels=False)),
                showlegend=False, margin=dict(l=30, r=30, t=20, b=20), height=250)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption(f"**Target Node:** Bus {bus_id} | **Ground Truth:** {'🔴 Failure' if label == 1 else '🟢 Safe'}")
        # --- NEW: TEMPORAL SEQUENCE VISUALIZATION ---
        if 'X_seq_train' in st.session_state:
            st.divider()
            st.markdown("#### 📈 Temporal Sequence Trajectory")
            st.write(
                    "Visualizing the chronological buildup of the selected instance over its 4-step temporal window. This trajectory is exactly what the PyTorch LSTM will analyze to calculate momentum.")

            # Grab the 3D tensor sequence for the selected instance
            # Shape is (4, Num_Features)
            sequence_data = st.session_state.X_seq_train[selected_idx]

            # Create chronological labels (e.g., t-3, t-2, t-1, t)
            time_steps = [f"t-{3 - i}" for i in range(4)]

            fig_seq = go.Figure()

            # Plot a trajectory line for each feature
            for i, feat_name in enumerate(theta_vals):
                fig_seq.add_trace(go.Scatter(
                    x=time_steps,
                    y=sequence_data[:, i],
                    mode='lines+markers',
                    line=dict(width=3),
                    marker=dict(size=8),
                    name=feat_name
                ))

            fig_seq.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Scaled Stress Level (rad)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_seq, use_container_width=True)

# --- TAB 2: HYBRID QUANTUM-DEEP LEARNING PIPELINE ---
with tab2:
    if not st.session_state.get('data_loaded', False):
        st.error("⚠️ Please Load, Extract, and Prepare Data in Phase 1 before proceeding.")
    elif 'X_seq_train' not in st.session_state:
        st.error("⚠️ Sequence data missing. Please go back to Phase 1 and click 'Extract Historical Typhoon'.")
    else:
        st.markdown("### 🧬 Quantum-DL Feature Integration")
        st.write(
            "Using the Quantum Ansatz to extract non-linear topological features, processed by the PyTorch Deep Learning engine.")

        # --- D3.JS CIRCUIT VISUALIZATION ---
        components.html(circuit_html, height=500, scrolling=True)
        st.divider()

        # --- HYBRID OPTIMIZATION ENGINE ---
        st.markdown("### ⚙️ Deep Learning Engine (Convergence Optimized)")

        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            c_epochs = st.number_input("Training Epochs", 10, 1000, 200, step=10)
        with col_hp2:
            c_lr = st.number_input("Initial Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        with col_hp3:
            c_batch = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

        if st.button("Train Q-TCN-LSTM Pipeline (Curriculum Mode)", type="primary"):
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            from src.qkn import QuantumKernelNetwork, ScaledQuantumTemporalConvNet, FocalLoss

            p_text = st.empty()
            p_bar = st.progress(0)
            loss_chart_placeholder = st.empty()

            # 1. RETRIEVE & EXTRACT FEATURES
            p_text.text("⚛️ Extracting Quantum Embeddings...")
            qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)

            X_train_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_train)
            X_val_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_cal)

            y_train = torch.tensor(st.session_state.y_train, dtype=torch.float32).unsqueeze(1)
            y_val = torch.tensor(st.session_state.y_cal, dtype=torch.float32).unsqueeze(1)

            # 2. MODEL, LOSS, OPTIMIZER
            model = ScaledQuantumTemporalConvNet(in_channels=X_train_tensor.shape[1])
            # Focal Loss: Alpha=0.75 gives high importance to minority/difficult class
            criterion = FocalLoss(alpha=0.60, gamma=2.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=c_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train), batch_size=c_batch, shuffle=True)
            loss_history = []

            # 3. CURRICULUM TRAINING LOOP
            for epoch in range(int(c_epochs)):
                model.train()
                epoch_train_loss = 0.0

                for b_x, b_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(b_x)

                    # CURRICULUM: In first 20% epochs, emphasize easier samples or weight differently
                    # Here we simply let FocalLoss handle the hard-example mining automatically
                    loss = criterion(outputs, b_y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    epoch_train_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_tensor), y_val).item()

                scheduler.step(val_loss)
                loss_history.append([epoch_train_loss / len(train_loader), val_loss])

                if epoch % 5 == 0:
                    p_text.text(f"🏃 Epoch {epoch}/{c_epochs} | Train: {loss_history[-1][0]:.4f} | Val: {val_loss:.4f}")
                    loss_chart_placeholder.line_chart(
                        pd.DataFrame(loss_history, columns=["Train Loss", "Validation Loss"]))
                    p_bar.progress(epoch / c_epochs)

            st.session_state.pytorch_model = model
            st.session_state.pytorch_qcnn_active = True
            st.success("✅ Hybrid Training Complete.")

        # --- INFERENCE SUMMARY ---
        if st.session_state.get('pytorch_qcnn_active', False):
            st.markdown("### 📊 Inference Distribution")
            with torch.no_grad():
                # Re-extract test features
                qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)
                X_test_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_test)
                probs = torch.sigmoid(st.session_state.pytorch_model(X_test_tensor)).numpy()

            df_preds = pd.DataFrame({"Prob": probs.flatten(), "True": st.session_state.y_test})
            fig = px.histogram(df_preds, x="Prob", color="True", barmode="overlay", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: UNCERTAINTY CALIBRATION (QCP) ---
with tab3:
    if not st.session_state.get('pytorch_qcnn_active', False):
        st.error("⚠️ Deep Learning model not trained. Please train the model in Phase 2 first.")
    else:
        st.markdown("### 🛡️ Non-Conformity Analysis")
        st.write("""
        We analyze the **Non-Conformity Scores** of the Calibration Set. 
        High scores indicate samples that the Hybrid Model finds 'surprising'. 
        The Conformal Predictor uses these to find a rigorous statistical threshold ($q_{\hat{h}}$).
        """)

        # 1. GENERATE CALIBRATION SCORES
        with st.spinner("Analyzing Calibration Set Surprises..."):
            import torch
            from src.qkn import QuantumKernelNetwork

            model = st.session_state.pytorch_model
            qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)

            # Extract Quantum Features for Calibration Set
            X_cal_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_cal)

            # Run Inference
            model.eval()
            with torch.no_grad():
                raw_logits = model(X_cal_tensor)
                p1_cal = torch.sigmoid(raw_logits).numpy().flatten()

            # Create 2D Probabilities [P(Safe), P(Failure)]
            p0_cal = 1.0 - p1_cal
            probs_cal = np.column_stack((p0_cal, p1_cal))

            # Calculate Non-Conformity Scores: s = 1 - P(true_class)
            cal_scores = []
            for i, true_label in enumerate(st.session_state.y_cal):
                score = 1 - probs_cal[i, int(true_label)]
                cal_scores.append(score)

            st.session_state.cal_scores = np.array(cal_scores)

        # 2. PLOT CALIBRATION DISTRIBUTION
        import pandas as pd
        import plotly.express as px

        df_scores = pd.DataFrame({
            "Non-Conformity Score": st.session_state.cal_scores,
            "Actual Label": ["Failure" if y == 1 else "Safe" for y in st.session_state.y_cal]
        })

        fig_dist = px.histogram(
            df_scores, x="Non-Conformity Score", color="Actual Label",
            marginal="box", barmode="overlay",
            color_discrete_map={"Safe": "#1f77b4", "Failure": "#d62728"},
            nbins=30, title="Calibration Score Distribution"
        )
        fig_dist.update_layout(xaxis_title="Surprise Score (1 - P_hat)", yaxis_title="Frequency",
                               plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()

        # 3. THRESHOLD CALIBRATION
        st.markdown("#### Determine Quantum Safety Threshold")
        st.write(
            f"Target coverage set at {target_coverage * 100:.1f}%. Clicking below calculates the statistical bound.")

        if st.button("Calculate Threshold (q_hat)"):
            from src.qcp import QuantumConformalPredictor

            # Initialize CP
            qcp = QuantumConformalPredictor(model, alpha=(1.0 - target_coverage))

            # Calibrate
            q_hat = qcp.calibrate(
                st.session_state.X_cal,
                st.session_state.y_cal,
                st.session_state.X_train,
                cal_probs=probs_cal
            )

            st.session_state.qcp_model = qcp
            st.session_state.q_hat = q_hat
            st.session_state.qcp_calibrated = True

            # Visual Feedback
            fig_dist.add_vline(x=q_hat, line_dash="dash", line_color="green",
                               annotation_text=f"Threshold (q_hat={q_hat:.3f})")
            st.plotly_chart(fig_dist, use_container_width=True)

            st.success(
                f"Threshold established. At {target_coverage * 100}% reliability, any prediction score above {q_hat:.4f} is considered ambiguous.")

# --- TAB 4: INFERENCE TESTING ---
with tab4:
    # Ensure keys exist even if not yet populated
    if 'y_pred_raw' not in st.session_state: st.session_state.y_pred_raw = None
    if 'y_pred_qcp' not in st.session_state: st.session_state.y_pred_qcp = None

    if not st.session_state.get('qcp_calibrated', False):
        st.error("⚠️ Calibration threshold not found. Please complete Phase 3 first.")
    else:
        st.markdown("### 🧪 Phase 4: Out-of-Sample Inference Testing")
        st.write(
            f"Evaluating model reliability on unseen test data using the established conformal threshold $q_{{\hat{{h}}}}$ = **{st.session_state.q_hat:.4f}**.")

        # 1. EXECUTE INFERENCE
        if st.button("Execute Inference & Generate Metrics", type="primary"):
            import torch
            import numpy as np
            import pandas as pd
            from src.qkn import QuantumKernelNetwork

            with st.spinner("Executing Hybrid Quantum-DL Inference..."):
                model = st.session_state.pytorch_model
                qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)

                X_test_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_test)
                y_test = st.session_state.y_test
                bus_test = st.session_state.bus_test

                model.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(model(X_test_tensor)).numpy().flatten()

                # Apply QCP Sets
                threshold = st.session_state.q_hat
                prediction_sets = []
                results = []
                for i, p in enumerate(probs):
                    s = []
                    if (1 - p) >= (1 - threshold): s.append("Safe")
                    if p >= (1 - threshold): s.append("Failure")
                    prediction_sets.append(s)
                    results.append({
                        "Bus_ID": bus_test[i],
                        "Truth": y_test[i],
                        "Prob_Fail": p,
                        "Alert_Level": "High" if "Failure" in s and "Safe" not in s else (
                            "Uncertain" if len(s) > 1 else "Low")
                    })

                # Save to Session State
                st.session_state.final_results = pd.DataFrame(results)
                st.session_state.y_pred_raw = (probs >= 0.5).astype(int)
                st.session_state.y_pred_qcp = [1 if "Failure" in s else 0 for s in prediction_sets]
                st.session_state.inference_complete = True
                st.rerun()

        # 2. VISUALIZATION & METRICS
        if st.session_state.get('inference_complete', False) and st.session_state.y_pred_qcp is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            y_test = st.session_state.y_test
            y_raw = st.session_state.y_pred_raw
            y_qcp = st.session_state.y_pred_qcp


            # Calculate Metrics
            def get_metrics(y_true, y_pred):
                return {
                    "Acc": accuracy_score(y_true, y_pred),
                    "Pre": precision_score(y_true, y_pred, zero_division=0),
                    "Rec": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0)
                }


            m_raw = get_metrics(y_test, y_raw)
            m_qcp = get_metrics(y_test, y_qcp)

            st.markdown("#### 📊 Performance Comparison")
            cols = st.columns(4)
            metrics = ["Acc", "Pre", "Rec", "F1"]
            for i, name in enumerate(metrics):
                delta = (m_qcp[name] - m_raw[name]) * 100
                cols[i].metric(name, f"{m_qcp[name]:.2%}", f"{delta:+.2f}%")


            # Confusion Matrices
            def plot_cm(cm, title, scale):
                return px.imshow(cm, text_auto=True, color_continuous_scale=scale,
                                 x=['Safe', 'Fail'], y=['Safe', 'Fail'], title=title)


            c1, c2 = st.columns(2)
            c1.plotly_chart(plot_cm(confusion_matrix(y_test, y_raw), "Raw Model", "Blues"), use_container_width=True)
            c2.plotly_chart(plot_cm(confusion_matrix(y_test, y_qcp), "QCP-FailSafe Model", "Reds"),
                            use_container_width=True)

            # Alert Ledger
            st.markdown("#### 📋 Node Alert Report")
            df_res = st.session_state.final_results

            # Use this updated lambda that provides full CSS strings for all conditions
            st.dataframe(df_res.style.map(
                lambda v: 'background-color: #ff9999;' if v == 'High' else
                ('background-color: #ffe066;' if v == 'Uncertain' else
                 'background-color: #b3ffb3;'),
                subset=['Alert_Level']), use_container_width=True)

            if st.button("Commit Alerts to CTQW Islanding Engine"):
                st.session_state.islanding_alerts = df_res[df_res["Alert_Level"] != "Low"]
                st.success("Alerts committed. Proceed to Phase 5.")