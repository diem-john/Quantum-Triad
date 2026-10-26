# Hybrid QKN (A+B) + QSTGNN Workflow

## Purpose

The integrated model combines the two quantum representations before spatiotemporal graph learning. It operates on the application's grid tensors rather than the row-wise CSV used by the lightweight SVM benchmark.

## End-to-end process

1. **Geo-extraction and event-window construction**
   - Build each snapshot as a graph with 33 buses.
   - Raw tensor shape: `(samples, nodes, time_steps, input_features)`.
   - Target shape: `(samples, nodes)`.
   - Retain the source storm ID for event-level splitting and reporting.

2. **Training/calibration/test partition**
   - Split by complete storm IDs before fitting any reference representation or model.
   - Never allow windows from one storm into multiple partitions.
   - The fidelity landmark bank below is fit using training tensors only.

3. **Quantum branch A: expectation values**
   - Apply the existing QKN feature map to each node/time feature vector.
   - Output shape: `(samples, nodes, n_qubits, time_steps)`.

4. **Quantum branch B: fidelity-to-training-landmarks**
   - Flatten node/time feature vectors from the training partition only.
   - Select a reproducible, bounded set of training-only reference vectors (default: 8).
   - Compute quantum fidelity between each node/time vector and each reference vector.
   - Output shape: `(samples, nodes, n_landmarks, time_steps)`.
   - Save the reference vectors to `data/processed/hybrid_qstgnn_encoder.pt` and reuse the same vectors for calibration and inference.

5. **A+B feature fusion**
   - Concatenate the quantum expectation and fidelity channels along the feature-channel axis.
   - Fused shape: `(samples, nodes, n_qubits + n_landmarks, time_steps)`.
   - This is feature-level fusion, not late fusion of two classifiers.

6. **QSTGNN spatiotemporal learning**
   - Temporal Conv1D learns node-wise changes over the input window.
   - Graph Attention Network (GAT) passes information along the mapped power-grid edges.
   - Bidirectional LSTM models temporal dependencies in the spatially informed sequence.
   - Temporal attention weights the sequence representations.
   - A sigmoid prediction head outputs one failure-risk probability per bus.
   - Training uses the actual mapped graph topology from `mapper.export_edge_index()`, not a fully connected synthetic graph.

7. **Model selection and persistence**
   - Optimize binary cross-entropy on the training partition.
   - Select the checkpoint with the best calibration/validation loss.
   - Save weights to `data/processed/hybrid_dl_weights.pt`.
   - Save the hybrid encoder's training-only landmarks separately.

8. **Uncertainty calibration**
   - Use the existing Phase 4 calibration process on a held-out calibration partition.
   - Fit the conformal threshold from calibration predictions only; do not tune it on the test set.

9. **Held-out inference and reporting**
   - Recreate the QKN backend and load the saved landmark bank.
   - Transform the held-out test tensors with the same A+B feature mapping.
   - Load the saved QSTGNN weights and predict bus-level risk.
   - Report Recall first, Precision second, Accuracy third, and F1 as a complementary metric.
   - Report pooled and per-storm results, and confirm the train/calibration/test storm IDs are disjoint.

10. **CTQW islanding analysis**
    - Pass the risk/failure predictions to the existing islanding workflow.
    - Evaluate this downstream stage separately; it does not change the predictive model's held-out labels.

## Run

From the repository root:

```bash
git checkout qdf_qkn_approach
git pull origin qdf_qkn_approach
python -m pytest -q tests/test_hybrid_qstgnn.py tests/test_hybrid_qkn.py tests/test_qkn_kernel.py
streamlit run main.py
```

In Phase 3, select **Hybrid QKN (A+B) + QSTGNN**. The application stores the fitted landmark bank and model weights in `data/processed/`.

## Important evaluation safeguards

- Keep complete storms disjoint across training, calibration, and test.
- The current feature encoder guarantees only that fidelity landmarks are drawn from the training tensor passed to `fit`. It cannot repair an upstream random split that has already mixed the same storm across partitions.
- The existing extraction code creates synthetic vulnerability-based failure labels. Results therefore demonstrate prediction of the modeled proxy target, not validated real-world outage prediction.
- Dense fidelity calculations scale with the number of node/time vectors times the number of landmarks. Increase landmark count only after measuring runtime.
- The current QSTGNN uses a bidirectional LSTM. This is valid for fixed historical windows where all timesteps are observed before the forecast origin; do not feed future observations into the window.
- A passing unit test verifies tensor shape and model wiring, not scientific superiority or quantum advantage.
