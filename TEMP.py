# --- TAB 2: GLOBAL OPTIMIZATION (PSO) ---
with tab2:
    st.markdown("### 🐝 Global Particle Swarm Optimization")
    st.write(
        "Deploys a swarm of particles to identify the optimal Quantum Architecture and Deep Learning hyperparameters.")

    if not st.session_state.get('data_loaded', False):
        st.error("⚠️ Please load and extract the data in Phase 1 first.")
    else:
        col_pso1, col_pso2, col_pso3 = st.columns(3)
        with col_pso1:
            p_swarm_size = st.number_input("Swarm Size", 3, 20, 5)
        with col_pso2:
            p_iterations = st.number_input("Max Iterations", 2, 100, 5)
        with col_pso3:
            p_epochs = st.number_input("Proxy Epochs", 3, 50, 10, help="Epochs run per particle.")

        if st.button("🚀 Launch Unified Swarm (Quantum + DL)", type="primary"):
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            from src.qkn import QuantumKernelNetwork, ScaledQuantumTemporalConvNet, FocalLoss

            pso_status = st.empty()
            pso_progress = st.progress(0)
            col_chart, col_data = st.columns([2, 1])
            with col_chart:
                chart_placeholder = st.empty()
            with col_data:
                data_placeholder = st.empty()

            # Proxy Dataset for Speed (Evaluates on a smaller subset)
            max_proxy_samples = 80
            X_seq_t_proxy = st.session_state.X_seq_train[:max_proxy_samples]
            y_t_proxy = torch.tensor(st.session_state.y_train[:max_proxy_samples], dtype=torch.float32).unsqueeze(1)
            X_seq_v_proxy = st.session_state.X_seq_cal[:40]
            y_v_proxy = torch.tensor(st.session_state.y_cal[:40], dtype=torch.float32).unsqueeze(1)

            # --- 9-DIMENSIONAL DECODER ---
            def decode_particle(pos):
                lr = 10 ** (pos[0] * (np.log10(0.1) - np.log10(0.0001)) + np.log10(0.0001))
                batch_size = int(pos[1] * (64 - 16) + 16)
                alpha = pos[2] * (0.9 - 0.2) + 0.2
                gamma = pos[3] * (4.0 - 0.5) + 0.5
                conv_out = int(pos[4] * (32 - 8) + 8)
                lstm_hidden = int(pos[5] * (64 - 16) + 16)
                n_qubits = int(pos[6] * 2.99) + 2  # Range [2, 3, 4]
                q_type = "StronglyEntangling" if pos[7] >= 0.5 else "BasicEntangling"
                n_layers = int(pos[8] * 4.99) + 1  # Range [1, 2, 3, 4, 5]
                return lr, batch_size, alpha, gamma, conv_out, lstm_hidden, n_qubits, q_type, n_layers

            def evaluate_unified_model(h_params):
                lr, batch_size, alpha, gamma, conv_out, lstm_hidden, n_qubits, q_type, n_layers = h_params

                # 1. Quantum Proxy Extraction (Now includes tuned layers)
                qkn = QuantumKernelNetwork(n_qubits=n_qubits, layers=n_layers, entangling_type=q_type)
                X_t_tensor = qkn.extract_temporal_quantum_features(X_seq_t_proxy)
                X_v_tensor = qkn.extract_temporal_quantum_features(X_seq_v_proxy)

                # 2. PyTorch Training
                model = ScaledQuantumTemporalConvNet(in_channels=n_qubits, conv_out=conv_out, lstm_hidden=lstm_hidden)
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                train_loader = DataLoader(TensorDataset(X_t_tensor, y_t_proxy), batch_size=batch_size, shuffle=True)

                model.train()
                for _ in range(p_epochs):
                    for b_x, b_y in train_loader:
                        optimizer.zero_grad()
                        loss = criterion(model(b_x), b_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_v_tensor), y_v_proxy).item()
                return val_loss

            # --- PSO EXECUTION ---
            dimensions = 9 # Upgraded to 9 dimensions
            particles_pos = np.random.rand(p_swarm_size, dimensions)
            particles_vel = np.random.uniform(-0.1, 0.1, (p_swarm_size, dimensions))

            pbest_pos, pbest_val = np.copy(particles_pos), np.full(p_swarm_size, float('inf'))
            gbest_pos, gbest_val = np.zeros(dimensions), float('inf')
            history_data = []

            for i in range(p_iterations):
                for p in range(p_swarm_size):
                    h_params = decode_particle(particles_pos[p])
                    pso_status.text(
                        f"Iter {i + 1}/{p_iterations} | Particle {p + 1}/{p_swarm_size}\nTesting: {h_params[6]} Qubits | {h_params[8]} Layers | {h_params[7]}")

                    score = evaluate_unified_model(h_params)

                    if score < pbest_val[p]:
                        pbest_val[p], pbest_pos[p] = score, np.copy(particles_pos[p])
                    if score < gbest_val:
                        gbest_val, gbest_pos = score, np.copy(particles_pos[p])

                    history_data.append({
                        "Iteration": i + 1, "Val_Loss": score, "Qubits": h_params[6], "Layers": h_params[8], "Entanglement": h_params[7],
                        "LR": round(h_params[0], 4), "Batch": h_params[1], "Alpha": round(h_params[2], 2),
                        "Gamma": round(h_params[3], 2), "Conv": h_params[4], "LSTM": h_params[5]
                    })

                # Update Velocities
                r1, r2 = np.random.rand(p_swarm_size, dimensions), np.random.rand(p_swarm_size, dimensions)
                particles_vel = 0.5 * particles_vel + 1.5 * r1 * (pbest_pos - particles_pos) + 1.5 * r2 * (
                            gbest_pos - particles_pos)
                particles_pos = np.clip(particles_pos + particles_vel, 0.0, 1.0)

                # UI Update
                df_hist = pd.DataFrame(history_data)
                fig = px.scatter(df_hist, x="Iteration", y="Val_Loss", color="Entanglement",
                                 hover_data=["Qubits", "Layers", "LR"])
                gbest_trend = df_hist.groupby("Iteration")["Val_Loss"].min().cummin().reset_index()
                fig.add_trace(go.Scatter(x=gbest_trend["Iteration"], y=gbest_trend["Val_Loss"], mode='lines',
                                         line=dict(color='yellow', dash='dash'), name='Global Best'))
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                data_placeholder.dataframe(df_hist.sort_values("Val_Loss").head(5), height=250)
                pso_progress.progress((i + 1) / p_iterations)

            # SAVE BEST PARAMETERS TO SESSION
            best = decode_particle(gbest_pos)
            st.session_state.opt_lr, st.session_state.opt_batch = best[0], best[1]
            st.session_state.opt_alpha, st.session_state.opt_gamma = best[2], best[3]
            st.session_state.opt_conv, st.session_state.opt_lstm = best[4], best[5]
            st.session_state.opt_qubits, st.session_state.opt_q_type = best[6], best[7]
            st.session_state.opt_layers = best[8] # Save Optimized Layers
            st.session_state.pso_complete = True

            pso_status.empty()
            pso_progress.empty()
            st.success("✅ Architecture & Hyperparameters Optimized! Values saved for Phase 3.")

# --- TAB 3: QUANTUM TRAINING ---
with tab3:
    if not st.session_state.get('data_loaded', False):
        st.error("⚠️ Please Extract Data in Phase 1.")
    else:
        st.markdown("### ⚛️ Phase 3: Quantum-DL Model Training")
        st.write("Training the full dataset using the optimized pipeline from Phase 2.")

        # --- D3.JS CIRCUIT VISUALIZATION ---
        try:
            from src.utils import circuit_html
            import streamlit.components.v1 as components

            components.html(circuit_html, height=450, scrolling=True)
        except ImportError:
            st.warning("Could not load circuit visualization from utils.")

        st.divider()

        # --- LOAD OPTIMAL HYPERPARAMETERS ---
        # Automatically load optimized parameters, fallback to defaults if PSO wasn't run
        c_lr = st.session_state.get("opt_lr", 0.005)
        c_batch = st.session_state.get("opt_batch", 32)
        c_alpha = st.session_state.get("opt_alpha", 0.60)
        c_gamma = st.session_state.get("opt_gamma", 2.0)
        c_conv = st.session_state.get("opt_conv", 16)
        c_lstm = st.session_state.get("opt_lstm", 32)
        c_qubits = st.session_state.get("opt_qubits", 3)  # Fallback to sidebar
        c_qtype = st.session_state.get("opt_q_type", "StronglyEntangling")
        c_layers = st.session_state.get("opt_layers", 2)

        st.info(f"🧬 **Active Architecture:** `{c_qubits} Qubits` | `{c_qtype}` | `Depth: {c_layers}` | `Conv: {c_conv}` | `LSTM: {c_lstm}`\n"
                f"⚙️ **Active Params:** `LR: {c_lr:.4f}` | `Batch: {c_batch}` | `Alpha: {c_alpha:.2f}` | `Gamma: {c_gamma:.2f}`")

        c_epochs = st.number_input("Full Training Epochs", 50, 1000, 200, step=10)

        # --- HYBRID TRAINING EXECUTION ---
        if st.button("Train Final Pipeline", type="primary"):
            import torch
            import pandas as pd
            from torch.utils.data import TensorDataset, DataLoader
            from src.qkn import QuantumKernelNetwork, ScaledQuantumTemporalConvNet, FocalLoss

            p_text = st.empty()
            p_bar = st.progress(0)
            loss_chart = st.empty()

            p_text.text("⚛️ Extracting Full Dataset Quantum Embeddings...")

            # 1. Use the optimized quantum parameters
            qkn = QuantumKernelNetwork(n_qubits=c_qubits, layers=c_layers, entangling_type=c_qtype)

            X_train_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_train)
            X_val_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_cal)

            y_train = torch.tensor(st.session_state.y_train, dtype=torch.float32).unsqueeze(1)
            y_val = torch.tensor(st.session_state.y_cal, dtype=torch.float32).unsqueeze(1)

            # 2. Use the optimized DL architecture
            model = ScaledQuantumTemporalConvNet(in_channels=c_qubits, conv_out=c_conv, lstm_hidden=c_lstm)
            criterion = FocalLoss(alpha=c_alpha, gamma=c_gamma)
            optimizer = torch.optim.Adam(model.parameters(), lr=c_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train), batch_size=c_batch, shuffle=True)
            loss_hist = []

            # 3. Training Loop
            for epoch in range(int(c_epochs)):
                model.train()
                train_loss = 0.0
                for b_x, b_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(b_x), b_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_tensor), y_val).item()

                scheduler.step(val_loss)
                loss_hist.append([train_loss / len(train_loader), val_loss])

                # Live Visual Updates
                if epoch % 5 == 0 or epoch == c_epochs - 1:
                    p_text.text(
                        f"🏃 Epoch {epoch + 1}/{c_epochs} | Train Loss: {loss_hist[-1][0]:.4f} | Val Loss: {val_loss:.4f}")
                    loss_chart.line_chart(pd.DataFrame(loss_hist, columns=["Train Loss", "Val Loss"]))
                    p_bar.progress((epoch + 1) / c_epochs)

            # 4. Save trained model & architecture rules for Phases 4-6
            st.session_state.pytorch_model = model
            st.session_state.active_q_qubits = c_qubits
            st.session_state.active_q_type = c_qtype
            st.session_state.active_q_depth = c_layers
            st.session_state.pytorch_qcnn_active = True

            p_bar.empty()
            st.success("✅ Full Model Trained Successfully! Architecture states saved to session.")

        # --- INFERENCE SUMMARY VISUALIZATION ---
        if st.session_state.get('pytorch_qcnn_active', False):
            st.divider()
            st.markdown("### 📊 Inference Distribution (Test Set)")

            import torch
            import pandas as pd
            import plotly.express as px
            from src.qkn import QuantumKernelNetwork

            # Fetch the active parameters established during training
            active_qubits = st.session_state.get("active_q_qubits", 3)
            active_qtype = st.session_state.get("active_q_type", "StronglyEntangling")
            active_qlayers = st.session_state.get("active_q_depth", 2)

            with torch.no_grad():
                # Re-extract test features using the matched architecture
                qkn = QuantumKernelNetwork(n_qubits=active_qubits, layers=active_qlayers, entangling_type=active_qtype)
                X_test_tensor = qkn.extract_temporal_quantum_features(st.session_state.X_seq_test)
                probs = torch.sigmoid(st.session_state.pytorch_model(X_test_tensor)).numpy()

            df_preds = pd.DataFrame({"Prob": probs.flatten(),
                                     "True Label": ["Failure" if y == 1 else "Safe" for y in st.session_state.y_test]})

            fig = px.histogram(
                df_preds, x="Prob", color="True Label",
                barmode="overlay", nbins=40, opacity=0.7,
                color_discrete_map={'Safe': '#1f77b4', 'Failure': '#d62728'}
            )
            fig.update_layout(title="Prediction Confidence Separation", margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "💡 **Next Step:** Proceed to Phase 4 (Uncertainty Calibration) to mathematically determine the 'Uncertainty Threshold' for the overlap area in the histogram above.")

# --- TAB 4: UNCERTAINTY CALIBRATION (QCP) ---
with tab4:
    if not st.session_state.get('pytorch_qcnn_active', False):
        st.error("⚠️ Deep Learning model not trained. Please train the model in Phase 3 first.")
    else:
        st.markdown("### 🛡️ Phase 4: Non-Conformity Analysis & Calibration")
        st.write("""
        We analyze the **Non-Conformity Scores** of the Calibration Set. 
        High scores indicate samples that the Hybrid Model finds 'surprising'. 
        The Conformal Predictor uses these to find a rigorous statistical threshold ($q_{\hat{h}}$).
        """)

        # Fetch active architecture rules established during Phase 3 Training
        active_qubits = st.session_state.get("active_q_qubits", 3)
        active_qtype = st.session_state.get("active_q_type", "StronglyEntangling")
        active_qlayers = st.session_state.get("active_q_depth", 2)

        # 1. GENERATE CALIBRATION SCORES
        with st.spinner("Analyzing Calibration Set Surprises..."):
            import torch
            import numpy as np
            from src.qkn import QuantumKernelNetwork

            model = st.session_state.pytorch_model

            # MUST use dynamically optimized parameters to match trained weights
            qkn = QuantumKernelNetwork(n_qubits=active_qubits, layers=active_qlayers, entangling_type=active_qtype)

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
        target_coverage = st.slider("QCP Target Coverage (%)", 80, 99, 90) / 100.0
        st.write(
            f"Target coverage set at {target_coverage * 100:.1f}%. Clicking below calculates the statistical bound.")

        if st.button("Calculate Threshold (q_hat)", type="primary"):
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
                f"✅ Threshold established. At {target_coverage * 100}% reliability, any prediction score above {q_hat:.4f} is considered ambiguous. Proceed to Phase 5.")

# --- TAB 5: INFERENCE TESTING ---
with tab5:
    # Ensure keys exist even if not yet populated
    if 'y_pred_raw' not in st.session_state: st.session_state.y_pred_raw = None
    if 'y_pred_qcp' not in st.session_state: st.session_state.y_pred_qcp = None

    if not st.session_state.get('qcp_calibrated', False):
        st.error("⚠️ Calibration threshold not found. Please complete Phase 4 first.")
    else:
        st.markdown("### 🧪 Phase 5: Out-of-Sample Inference Testing")
        st.write(
            f"Evaluating model reliability on unseen test data using the established conformal threshold $q_{{\hat{{h}}}}$ = **{st.session_state.q_hat:.4f}**.")

        # Fetch active architecture rules established during Phase 3 Training
        active_qubits = st.session_state.get("active_q_qubits", 3)
        active_qtype = st.session_state.get("active_q_type", "StronglyEntangling")
        active_qlayers = st.session_state.get("active_q_depth", 2)

        # 1. EXECUTE INFERENCE
        if st.button("Execute Inference & Generate Metrics", type="primary"):
            import torch
            import numpy as np
            import pandas as pd
            from src.qkn import QuantumKernelNetwork

            with st.spinner("Executing Hybrid Quantum-DL Inference..."):
                model = st.session_state.pytorch_model

                # MUST use dynamically optimized parameters to match trained weights
                qkn = QuantumKernelNetwork(n_qubits=active_qubits, layers=active_qlayers, entangling_type=active_qtype)

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
                ambiguous_count = 0

                for i, p in enumerate(probs):
                    s = []
                    if (1 - p) >= (1 - threshold): s.append("Safe")
                    if p >= (1 - threshold): s.append("Failure")

                    prediction_sets.append(s)

                    if len(s) > 1:
                        ambiguous_count += 1

                    results.append({
                        "Bus_ID": bus_test[i],
                        "Truth": y_test[i],
                        "Prob_Fail": p,
                        "Alert_Level": "High" if "Failure" in s and "Safe" not in s else (
                            "Uncertain" if len(s) > 1 else "Low")
                    })

                # Calculate Certain vs Ambiguous Percentages
                total_samples = len(results)
                certain_count = total_samples - ambiguous_count

                # Save to Session State
                st.session_state.final_results = pd.DataFrame(results)
                st.session_state.y_pred_raw = (probs >= 0.5).astype(int)
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

            # --- PREDICTION CONFIDENCE (CERTAIN VS AMBIGUOUS) ---
            st.markdown("#### 🎯 Conformal Set Efficiency")

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
                st.metric("✅ Certain Predictions (Model is Sure)", f"{c_pct:.1f}%", f"{c_count} samples",
                          delta_color="normal")
                st.metric("⚠️ Ambiguous Sets (Needs Intervention)", f"{a_pct:.1f}%", f"{a_count} samples",
                          delta_color="inverse")
                st.caption(
                    "A well-calibrated model balances high reliability with a low ambiguity rate. Ambiguous sets trigger Phase 6 Islanding.")

            st.divider()

            # --- PERFORMANCE COMPARISON ---
            st.markdown("#### 📊 Performance Comparison")
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
            st.markdown("#### 📋 Node Alert Report")
            df_res = st.session_state.final_results

            st.dataframe(df_res.style.map(
                lambda v: 'background-color: #ff9999;' if v == 'High' else
                ('background-color: #ffe066;' if v == 'Uncertain' else
                 'background-color: #b3ffb3;'),
                subset=['Alert_Level']), use_container_width=True)

            if st.button("Commit Alerts to CTQW Islanding Engine", type="primary"):
                st.session_state.islanding_alerts = df_res[df_res["Alert_Level"] != "Low"]

                # Derive risky buses specifically for Phase 6 input
                risky_bus_list = df_res[df_res["Alert_Level"].isin(["High", "Uncertain"])]["Bus_ID"].tolist()
                st.session_state.risky_buses = list(set(risky_bus_list))

                st.success("✅ Alerts committed. Proceed to Phase 6.")


# --- TAB 6: CTQW ISLANDING ---
with tab6:
    if "risky_buses" not in st.session_state or not st.session_state.risky_buses:
        st.warning("⚠️ Please run Phase 5 to identify high-risk ambiguous buses before proceeding to Islanding.")
    else:
        st.markdown("### 🌊 Phase 6: Continuous-Time Quantum Walk (CTQW) Islanding")
        st.write("""
        This phase simulates a **Continuous-Time Quantum Walk** on the IEEE 33-Bus Microgrid topology. 
        By injecting a quantum state at the high-risk buses identified in Phase 5, we calculate the probability 
        distribution $|\psi(t)|^2$ to determine how cascading failures propagate. This identifies the optimal 
        nodes to decouple, fracturing the grid into resilient islands.
        """)

        # 1. CTQW SIMULATION (MATHEMATICAL ENGINE)
        def run_ctqw(adj_matrix, start_node_idx, time_step=1.0):
            from scipy.linalg import expm
            # Hamiltonian H = Adjacency Matrix
            H = adj_matrix
            # Unitary time evolution operator: U(t) = exp(-i * H * t)
            U = expm(-1j * H * time_step)

            # Initial quantum state: localized entirely at the target bus
            psi_0 = np.zeros(adj_matrix.shape[0], dtype=complex)
            psi_0[start_node_idx] = 1.0

            # Evolved state over time t
            psi_t = U @ psi_0

            # Return probability distribution
            return np.abs(psi_t) ** 2

        # 2. TOPOLOGY & ADJACENCY EXTRACTION
        import networkx as nx

        # Safely access the NetworkX graph from the mapper
        mapper = st.session_state.mapper
        graph_object = getattr(mapper, 'graph', getattr(mapper, 'G', None))

        if graph_object is None:
            st.error("⚠️ Could not locate the microgrid topology graph. Please re-initialize Phase 1.")
        else:
            adj_matrix = nx.to_numpy_array(graph_object)
            nodes_list = list(graph_object.nodes())

            st.markdown("#### ⚛️ Quantum Propagation Parameters")
            col_ctrl1, col_ctrl2 = st.columns(2)

            with col_ctrl1:
                # Allow the user to select which specific risky bus to analyze
                target_bus = st.selectbox("Select High-Risk Bus (Epicenter):",
                                          options=sorted(st.session_state.risky_buses))
            with col_ctrl2:
                time_evolution = st.slider("Quantum Evolution Time ($t$)", min_value=0.1, max_value=5.0, value=1.0,
                                           step=0.1)

            with st.spinner(f"Simulating Quantum Walk from Bus {target_bus}..."):
                # Find the matrix index of the target bus
                if target_bus in nodes_list:
                    target_idx = nodes_list.index(target_bus)
                    probs = run_ctqw(adj_matrix, target_idx, time_evolution)
                else:
                    st.error(f"Bus {target_bus} not found in graph.")
                    probs = np.zeros(len(nodes_list))
                    target_idx = 0

            # 3. VISUALIZATION: PROBABILITY HEATMAP
            st.markdown("#### 📊 Probability Density Distribution")

            import plotly.graph_objects as go
            fig_ctqw = go.Figure(data=[go.Bar(
                x=[f"Bus {n}" for n in nodes_list],
                y=probs,
                marker_color='#1f77b4',
                marker_line_color='#0a4a75',
                marker_line_width=1.5,
                opacity=0.8
            )])

            # Identify the fracture points (nodes with high probability amplitude, excluding the epicenter)
            fracture_threshold = np.mean(probs) + np.std(probs)
            suggested_islands = [n for i, n in enumerate(nodes_list) if
                                 probs[i] > fracture_threshold and n != target_bus]

            fig_ctqw.add_hline(y=fracture_threshold, line_dash="dash", line_color="red",
                               annotation_text="Fracture Threshold (μ + σ)")

            fig_ctqw.update_layout(
                title=f"Quantum Probability Amplitude", #$|\psi(t={time_evolution})|^2$
                xaxis_title="Microgrid Nodes",
                yaxis_title="Probability Density",
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ctqw, use_container_width=True)

            # 4. ISLANDING DECISION & MAPPING
            st.markdown("#### 🗺️ Resilient Islanding Strategy")

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                max_prop_node = nodes_list[np.argmax(probs)]
                st.metric("Maximum Propagation Node", f"Bus {max_prop_node}")
                st.info(
                    f"**Islanding Protocol:** To prevent cascading failure, disconnect **Bus {target_bus}** from the surrounding buffer nodes.")

                st.json({
                    "Epicenter (Isolate)": target_bus,
                    "Buffer Zone (Monitor/Shed)": suggested_islands
                })

                if st.button("Finalize Resiliency Report", type="primary"):
                    st.balloons()
                    st.success("✅ Quantum Resiliency Protocol successfully generated for the Chiayi Microgrid System.")

            with col_res2:
                # Color map for the islanding visualization
                node_colors = {node: "#2ecc71" for node in nodes_list}  # Default Safe Green
                for node in suggested_islands:
                    node_colors[node] = "#f39c12"  # Buffer Orange
                node_colors[target_bus] = "#e74c3c"  # Epicenter Red

                # Call the custom mapping function defined at the top of your script
                fig_map = plot_interactive_map(
                    mapper,
                    title=f"Islanding Boundaries (t={time_evolution})",
                    node_colors=node_colors
                )
                st.plotly_chart(fig_map, use_container_width=True)


### TABS Error
# --- TAB 4: PHASE 4 - CONFORMAL INFERENCE ---
with tab4:
    st.markdown("### 🛡️ Phase 4: Quantum Conformal Predictor & Inference")
    st.info("Applies Conformal Prediction to the locked Q-STGNN. Instead of raw guesses, the model outputs statistically guaranteed prediction sets for the unseen test typhoons.")

    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model in Tab 3 before running Conformal Inference.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Conformal Parameters")
            alpha = st.slider("Target Error Rate (α):", min_value=0.01, max_value=0.20, value=0.10, step=0.01,
                              help="An α of 0.10 guarantees that the true grid state (Safe or Failure) will be contained in the prediction set 90% of the time.")

            run_inference_btn = st.button("Run Guaranteed Inference", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Inference Telemetry & Calibration")
            cp_status = st.empty()

            # Placeholders for results
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Target Coverage", f"{(1 - alpha)*100:.1f}%")
            with metrics_col2:
                empirical_cov_metric = st.empty()
            with metrics_col3:
                uncertainty_metric = st.empty()

            st.caption("Non-Conformity Score Distribution (Calibration Set)")
            dist_chart = st.empty()

        if run_inference_btn:
            import numpy as np
            import pandas as pd
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            import plotly.express as px
            import os

            from src.qkn import QuantumKernelNetwork, QuantumSpatiotemporalGNN

            try:
                cp_status.info("📐 Calculating Non-Conformity Scores on Calibration set...")

                # --- 1. POST-HOC CALIBRATION ---
                cal_probs = st.session_state.cal_probs
                y_cal = st.session_state.y_cal

                probs_flat = cal_probs.flatten()
                y_cal_flat = y_cal.flatten()

                prob_true_class = probs_flat * y_cal_flat + (1 - probs_flat) * (1 - y_cal_flat)
                scores = 1.0 - prob_true_class

                n = len(scores)
                q_level = np.ceil((n + 1) * (1 - alpha)) / n
                q_level = min(q_level, 1.0)
                q_hat = np.quantile(scores, q_level)

                df_scores = pd.DataFrame({'Non-Conformity Score': scores})
                fig_dist = px.histogram(df_scores, x='Non-Conformity Score', nbins=50,
                                        color_discrete_sequence=['#4C78A8'])
                fig_dist.add_vline(x=q_hat, line_dash="dash", line_color="red",
                                   annotation_text=f"q_hat threshold ({q_hat:.3f})")
                dist_chart.plotly_chart(fig_dist, use_container_width=True)

                # --- 2. TEST SET INFERENCE ---
                cp_status.info("⚡ Extracting Test Set features through Quantum Kernel...")
                config = st.session_state.final_model_config

                qkn = QuantumKernelNetwork(n_qubits=config['q_qubits'],
                                           layers=config['q_layers'],
                                           entangling_type=config['q_entangle'])

                X_test_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_test)
                y_test = torch.tensor(st.session_state.y_test, dtype=torch.float32)

                cp_status.info("🧠 Running PyTorch Graph Network...")

                model = QuantumSpatiotemporalGNN(
                    in_channels=config['q_qubits'],
                    seq_len=4,
                    conv_out=config['c_conv'],
                    gat_heads=config['c_gat'],
                    lstm_hidden=config['c_lstm'],
                    dropout=config['c_drop']
                )
                model.load_state_dict(torch.load("data/processed/hybrid_dl_weights.pt", weights_only=True))
                model.eval()

                edge_index, _ = st.session_state.mapper.export_edge_index()

                test_loader = DataLoader(TensorDataset(X_test_q, y_test), batch_size=16, shuffle=False)
                test_probs_list = []

                with torch.no_grad():
                    for inputs, _ in test_loader:
                        outputs = model(inputs, edge_index)
                        test_probs_list.append(outputs.numpy())

                test_probs = np.concatenate(test_probs_list, axis=0)
                y_test_np = y_test.numpy()

                # --- 3. CONFORMAL PREDICTION SETS ---
                cp_status.info("🛡️ Constructing Conformal Prediction Sets...")

                include_1 = test_probs >= (1 - q_hat)
                include_0 = test_probs <= q_hat

                # --- CRITICAL FIX: THE EMPTY SET FALLBACK ---
                # If a prediction falls into the "Death Zone" between thresholds, the set is empty.
                # In safety systems, an empty set MUST be converted to [Safe, Failure] (Uncertainty).
                empty_sets = ~(include_1 | include_0)

                # Force both to True where the set was previously empty
                include_1[empty_sets] = True
                include_0[empty_sets] = True

                # Evaluate Coverage and Set Sizes
                covered = 0
                total_points = y_test_np.size
                uncertain_count = 0

                y_test_flat = y_test_np.flatten()
                inc_1_flat = include_1.flatten()
                inc_0_flat = include_0.flatten()

                for i in range(total_points):
                    true_label = y_test_flat[i]
                    in_1 = inc_1_flat[i]
                    in_0 = inc_0_flat[i]

                    if (true_label == 1 and in_1) or (true_label == 0 and in_0):
                        covered += 1

                    if in_1 and in_0:
                        uncertain_count += 1

                empirical_coverage = covered / total_points
                uncertain_rate = uncertain_count / total_points

                empirical_cov_metric.metric("Empirical Coverage", f"{empirical_coverage*100:.1f}%",
                                            delta=f"{(empirical_coverage - (1 - alpha))*100:.1f}% (vs Target)")
                uncertainty_metric.metric("Uncertainty Rate", f"{uncertain_rate*100:.1f}%",
                                          help="Percentage of predictions where the model returned [Safe, Failure] because the typhoon physics fell outside the confidence threshold.")

                if empirical_coverage >= (1 - alpha):
                    cp_status.success("✅ Conformal Inference Complete! The theoretical coverage guarantee held true on the unseen dataset.")
                else:
                    cp_status.warning("⚠️ Inference Complete. Coverage missed slightly, indicating severe Distribution Shift (Test typhoons are behaving differently than Calibration typhoons).")

            except Exception as e:
                cp_status.error(f"⚠️ Inference Error: {e}")

# --- TAB 5: PHASE 5 - QUANTUM WALK ISLANDING ---
with tab5:
    st.markdown("### 🌊 Phase 5: Post-Disaster Quantum Islanding")
    st.info(
        "Simulates grid fracture based on model predictions and uses Continuous-Time Quantum Walks (CTQW) to map self-sustaining microgrid islands.")

    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model in Tab 3 first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Disaster Scenario Setup")

            # Allow the user to select a specific storm snapshot from the test set
            n_test_snapshots = len(st.session_state.X_seq_test)
            snapshot_idx = st.slider("Select Test Storm Snapshot:", 0, n_test_snapshots - 1, 0,
                                     help="Slides through the timeline of the test set typhoons.")

            # Risk tolerance threshold for failure
            fail_threshold = st.slider("Failure Probability Threshold:", 0.1, 0.9, 0.5, 0.05,
                                       help="If a bus's failure probability exceeds this, it is considered destroyed.")

            run_islanding_btn = st.button("Simulate Fracture & Islanding", type="primary", use_container_width=True)

        with col2:
            st.markdown("#### Post-Disaster Microgrid Topology")
            island_plot = st.empty()
            metrics_display = st.empty()

        if run_islanding_btn:
            import torch
            import networkx as nx
            import plotly.graph_objects as go
            import plotly.express as px
            from src.qkn import QuantumKernelNetwork, QuantumSpatiotemporalGNN
            from src.mapping import QuantumWalkIslandingMapper

            try:
                with st.spinner("Predicting failures and calculating quantum wave function spread..."):
                    # --- 1. RUN INFERENCE ON SINGLE SNAPSHOT ---
                    config = st.session_state.final_model_config
                    qkn = QuantumKernelNetwork(n_qubits=config['q_qubits'], layers=config['q_layers'],
                                               entangling_type=config['q_entangle'])

                    # Extract the single snapshot (Shape: 1, 33, 4, Features)
                    X_single = st.session_state.X_seq_test[snapshot_idx:snapshot_idx + 1]
                    X_q = qkn.extract_temporal_quantum_features(X_single)

                    model = QuantumSpatiotemporalGNN(
                        in_channels=config['q_qubits'],
                        seq_len=4,
                        conv_out=config['c_conv'],
                        gat_heads=config['c_gat'],
                        lstm_hidden=config['c_lstm'],
                        dropout=config['c_drop']
                    )
                    model.load_state_dict(torch.load("data/processed/hybrid_dl_weights.pt", weights_only=True))
                    model.eval()

                    edge_index, node_mapping = st.session_state.mapper.export_edge_index()

                    with torch.no_grad():
                        probs = model(X_q, edge_index).squeeze().numpy()

                    # --- 2. DETERMINE FRACTURES ---
                    mapper = st.session_state.mapper
                    node_list = list(mapper.graph.nodes())

                    # Identify which buses exceeded the threshold
                    failed_nodes = [node_list[i] for i, p in enumerate(probs) if p >= fail_threshold]

                    # If a bus fails, all edges (lines) connected to it are broken
                    failed_edges = []
                    for node in failed_nodes:
                        failed_edges.extend(list(mapper.graph.edges(node)))

                    # Simulate the fracture
                    post_disaster_grid = mapper.simulate_typhoon_failures(failed_edges)

                    # --- 3. QUANTUM WALK ISLANDING ---
                    # Only run CTQW if there are surviving nodes
                    if len(post_disaster_grid.nodes) > 0:
                        q_mapper = QuantumWalkIslandingMapper(post_disaster_grid)
                        islands = q_mapper.identify_islands()
                    else:
                        islands = {}

                    # --- 4. VISUALIZATION ---
                    fig = go.Figure()

                    # Draw surviving edges in light grey
                    for u, v in post_disaster_grid.edges():
                        x0, y0 = mapper.bus_coords[u]
                        x1, y1 = mapper.bus_coords[v]
                        fig.add_trace(go.Scatter(
                            x=[x0, x1, None], y=[y0, y1, None],
                            mode='lines', line=dict(color='lightgrey', width=2), hoverinfo='none', showlegend=False
                        ))

                    # Draw the distinct islands identified by the Quantum Walk
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

                    # Draw the failed nodes in black with an 'X'
                    if failed_nodes:
                        fx = [mapper.bus_coords[n][0] for n in failed_nodes]
                        fy = [mapper.bus_coords[n][1] for n in failed_nodes]
                        fig.add_trace(go.Scatter(
                            x=fx, y=fy, mode='markers',
                            marker=dict(size=12, color='black', symbol='x'),
                            name='Failed Buses', hoverinfo='text', text=[f"Bus {n}" for n in failed_nodes]
                        ))

                    fig.update_layout(
                        title=f"Post-Disaster Topology (Snapshot {snapshot_idx})",
                        showlegend=True, margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        plot_bgcolor='white'
                    )

                    island_plot.plotly_chart(fig, use_container_width=True)

                    # Output Metrics
                    metrics_display.success(
                        f"**Disaster Summary:** {len(failed_nodes)} buses failed. The grid fractured into **{len(islands)}** autonomous microgrid(s).")

            except Exception as e:
                island_plot.error(f"⚠️ Islanding Error: {e}")


# --- TAB 3: PHASE 3 - HYBRID DL TRAINING ---
with tab3:
    st.markdown("### 🧠 Phase 3: Quantum-LSTM Training & Hyperparameter Tuning")
    st.info(
        "Optimize the network using Optuna, then train the PyTorch Graph Network. The classical data is first projected into a Hilbert space via the Quantum Kernel Network.")

    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please complete the Optimization in Tab 2 first.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Step 1: Bayesian Search")
            do_bo = st.checkbox("Run Optuna BO Before Training", value=False)
            if do_bo:
                n_trials = st.slider("Optimization Trials:", 2, 20, 5)
                proxy_samples = st.slider("Proxy Subset Size:", 16, 64, 32)

            st.markdown("#### Step 2: Training Config")
            epochs = st.number_input("Training Epochs:", min_value=5, max_value=200, value=25, step=5)
            batch_size = st.selectbox("Batch Size:", [8, 16, 32], index=1)

            # Manual fallback params
            q_qubits = st.number_input("Qubits:", 2, 4, 3)
            q_layers = st.number_input("Quantum Layers:", 1, 4, 2)
            c_conv = 16
            c_gat = 2
            c_lstm = 32
            c_lr = 0.005
            c_drop = 0.3  # Slightly higher default dropout to combat overfitting

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
            # --- BAYESIAN OPTIMIZATION BLOCK ---
            if do_bo:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                telemetry_status.info("🧬 Running Bayesian Search...")

                dynamic_seq_len = st.session_state.X_seq_train.shape[2]


                def objective(trial):
                    nq = trial.suggest_int("n_qubits", 2, 4)
                    ql = trial.suggest_int("q_layers", 1, 3)
                    cc = trial.suggest_int("conv_out", 8, 32, step=8)
                    gh = trial.suggest_int("gat_heads", 1, 4)
                    cl = trial.suggest_int("lstm_hidden", 16, 64, step=16)
                    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
                    dr = trial.suggest_float("dropout", 0.2, 0.6)

                    qkn_bo = QuantumKernelNetwork(n_qubits=nq, layers=ql, entangling_type='StronglyEntangling')
                    X_p_q = qkn_bo.extract_temporal_quantum_features(st.session_state.X_seq_train[:proxy_samples])
                    y_p_t = torch.tensor(st.session_state.y_train[:proxy_samples], dtype=torch.float32)

                    edge_idx = torch.tensor([[i, j] for i in range(36) for j in range(36) if i != j],
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

                params = study.best_params
                q_qubits, q_layers, c_conv, c_gat, c_lstm, c_lr, c_drop = params['n_qubits'], params['q_layers'], \
                params['conv_out'], params['gat_heads'], params['lstm_hidden'], params['lr'], params['dropout']
                telemetry_status.success(f"BO Found Params: Qubits:{q_qubits}, LR:{c_lr:.4f}, Lstm:{c_lstm}")

            # --- TRAINING BLOCK ---
            dynamic_seq_len = st.session_state.X_seq_train.shape[2]
            telemetry_status.info("⚡ Extracting full dataset through Quantum Kernel Network...")

            qkn = QuantumKernelNetwork(n_qubits=q_qubits, layers=q_layers, entangling_type='StronglyEntangling')
            X_train_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_train)
            X_cal_q = qkn.extract_temporal_quantum_features(st.session_state.X_seq_cal)

            st.session_state.X_train_q_numpy = X_train_q.detach().numpy()

            # [FIX]: APPLY LABEL SMOOTHING TO PREVENT 0.000 LOSS (0 -> 0.05, 1 -> 0.95)
            y_train_smooth = np.clip(st.session_state.y_train, 0.05, 0.95)
            y_cal_smooth = np.clip(st.session_state.y_cal, 0.05, 0.95)

            y_train_t = torch.tensor(y_train_smooth, dtype=torch.float32)
            y_cal_t = torch.tensor(y_cal_smooth, dtype=torch.float32)

            train_loader = DataLoader(TensorDataset(X_train_q, y_train_t), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_cal_q, y_cal_t), batch_size=batch_size, shuffle=False)

            edge_index = torch.tensor([[i, j] for i in range(36) for j in range(36) if i != j], dtype=torch.long).t()

            telemetry_status.info("🚀 Initializing PyTorch Graph and Beginning Training...")

            model = QuantumSpatiotemporalGNN(in_channels=q_qubits, seq_len=dynamic_seq_len, conv_out=c_conv,
                                             gat_heads=c_gat, lstm_hidden=c_lstm, dropout=c_drop)

            # Increased weight decay (L2 Regularization) to keep weights from exploding
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

                    # [FIX]: CLAMP OUTPUTS TO PREVENT MATH ERRORS & SATURATION
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

                        # Calculate accuracy based on original binary logic (> 0.5)
                        predicted = (outputs_clamped >= 0.5).float()
                        hard_targets = (targets >= 0.5).float()  # Revert smoothed labels for hard accuracy

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
                'q_qubits': q_qubits, 'q_layers': q_layers, 'q_entangle': 'StronglyEntangling',
                'c_conv': c_conv, 'c_gat': c_gat, 'c_lstm': c_lstm, 'c_drop': c_drop
            }
            st.session_state.model_trained = True
            telemetry_status.success(f"✅ Training Complete! Best Val Loss ({best_val_loss:.4f}) locked and saved.")

        # --- QUANTUM KERNEL INSPECTOR ---
        if st.session_state.get('model_trained', False) and 'X_train_q_numpy' in st.session_state:
            st.divider()
            st.markdown("### 🔬 Quantum Kernel & Embedding Inspector")
            st.write(
                "Visualize how the Quantum Circuit embedded your topological features into the Hilbert space, and see the resulting **Gram Matrix (Quantum Kernel)**.")

            q_col1, q_col2 = st.columns([1, 2])
            with q_col1:
                snap_q = st.slider("Select Training Snapshot:", 0, len(st.session_state.X_train_q_numpy) - 1, 0,
                                   key='q_snap')
                time_q = st.slider("Select Sequence Timestep:", 0, st.session_state.X_train_q_numpy.shape[2] - 1, 0,
                                   key='q_time')

            embeds = st.session_state.X_train_q_numpy[snap_q, :, time_q, :]

            fig_emb = px.imshow(
                embeds.T,
                labels=dict(x="Grid Nodes (Buses)", y="Qubit Amplitude", color="Value"),
                title="Quantum Feature Embeddings (State Vector)",
                color_continuous_scale="Viridis",
                aspect="auto"
            )

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

# --- TAB 6: PHASE 6 - QUANTUM WALK ISLANDING ---
with tab6:
    st.markdown("### 🌊 Phase 6: Post-Disaster Quantum Islanding")
    st.info(
        "Simulate grid fracture using the RAW inference probabilities from Phase 5. Adjust the threshold to explore different disaster severity scenarios. Analyzes cascading failures using Continuous-Time Quantum Walks (CTQW).")

    if not st.session_state.get('inference_complete', False):
        st.warning("⚠️ Please complete Out-of-Sample Inference in Tab 5 first to generate the storm probabilities.")
    else:
        col1, col2 = st.columns([1, 2])

        # Fetch the results computed in Tab 5
        df_res = st.session_state.final_results
        max_snap = int(df_res['Snapshot_ID'].max())

        with col1:
            st.markdown("#### Disaster Scenario Setup")

            # Snapshot and Threshold Sliders
            snapshot_idx = st.slider("Select Test Storm Snapshot:", 0, max_snap, 0,
                                     help="Slides through the timeline of the test set typhoons.")

            fail_threshold = st.slider("Raw Failure Probability Threshold:", 0.05, 0.95, 0.50, 0.05,
                                       help="If a bus's RAW predicted failure probability exceeds this threshold, it is considered destroyed.")

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

                    # --- 1. DETERMINE FRACTURES (Using RAW Probs) ---
                    snapshot_data = df_res[df_res['Snapshot_ID'] == snapshot_idx]
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
                        st.markdown("#### ⚛️ Quantum Cascade & Buffer Zone Analysis")
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
                                      f"Raw Fail Prob: {worst_bus_row['Raw_Prob_Fail']:.2f}", delta_color="inverse")
                            st.info(
                                f"The quantum state $|\psi(0)\\rangle$ is initialized at Bus {target_bus}. Evolving for $t={time_evolution}$...")

                            # Fetch base Adjacency Matrix from the intact graph to model how the cascade would spread
                            adj_matrix = nx.to_numpy_array(mapper.graph)
                            nodes_list = list(mapper.graph.nodes())

                            # Mathematical Engine (SciPy)
                            H = adj_matrix
                            U = sl.expm(-1j * H * time_evolution)

                            target_idx = nodes_list.index(target_bus)
                            psi_0 = np.zeros(len(nodes_list), dtype=complex)
                            psi_0[target_idx] = 1.0

                            psi_t = U @ psi_0
                            probs = np.abs(psi_t) ** 2

                            # Identify the fracture points (nodes with high probability amplitude)
                            fracture_threshold = np.mean(probs) + np.std(probs)
                            buffer_zones = [n for i, n in enumerate(nodes_list) if
                                            probs[i] > fracture_threshold and n != target_bus]

                            st.json({
                                "Epicenter (Isolate)": target_bus,
                                "High-Risk Buffer Zones (Monitor/Shed)": buffer_zones
                            })

                        with col_qc2:
                            # 3. VISUALIZATION: PROBABILITY HEATMAP BAR CHART
                            fig_ctqw = go.Figure(data=[go.Bar(
                                x=[f"Bus {n}" for n in nodes_list],
                                y=probs,
                                marker_color='#1f77b4',
                                marker_line_color='#0a4a75',
                                marker_line_width=1.5,
                                opacity=0.8
                            )])

                            fig_ctqw.add_hline(y=fracture_threshold, line_dash="dash", line_color="red",
                                               annotation_text="Buffer Threshold (μ + σ)")

                            fig_ctqw.update_layout(
                                title=f"Quantum Probability Amplitude |ψ(t={time_evolution})|²",
                                xaxis_title="Microgrid Nodes",
                                yaxis_title="Probability Density",
                                template="plotly_white",
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_ctqw, use_container_width=True)

            except Exception as e:
                island_plot.error(f"⚠️ Islanding Error: {e}")