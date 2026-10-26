# QDF-QKN: Explicit Quantum Fidelity Kernel Learning

This branch adds an alternative to the existing expectation-value feature extractor in `src/qkn.py`; the existing implementation is unchanged.

## Current architecture

- **Approach A (existing):** `src/qkn.py` produces quantum expectation-value features, which are consumed by the classical spatiotemporal GNN.
- **Approach B (new):** `src/qkn_kernel.py` prepares quantum states and computes `K(x,y) = |<phi(x)|phi(y)>|^2`. The Gram matrix is passed directly to scikit-learn `SVC(kernel="precomputed")`.

These are separate model paths; Approach B does not replace or feed the existing GNN.

## Run the kernel tests

```bash
python -m pip install numpy==2.2.6 pandas==2.3.3 pennylane==0.42.3 scikit-learn==1.7.2 pytest
python -m pytest -q tests/test_qkn_kernel.py tests/test_qkn_kernel_experiment.py
```

## Run a first comparison

From the repository root:

```bash
python -m experiments.compare_qkn_kernel \
  --data data/processed/final_quantum_dataset.csv \
  --output outputs/qkn_kernel_comparison.json \
  --n-qubits 3 --C 1.0 --max-samples 250
```

This runs a reproducible train/test comparison between the quantum fidelity-kernel SVC and a classical RBF-SVC. Both use the same rows and labels; each scaling transformation is fit on training data only. The quantum features are mapped to rotation angles in [-pi, pi]. Results are written to JSON and CSV under `outputs/`.

For an initial full-dataset run, use `--max-samples 0`. This may be slow because the dense quantum training Gram matrix requires O(n_train^2) pairwise state comparisons, each involving circuit simulation.

### Important split limitation

The checked-in `data/processed/final_quantum_dataset.csv` has three feature columns and `Failure_Label`, but no storm/event identifier. By default, the runner uses a reproducible stratified random split and labels it `stratified_random_exploratory`. This split can mix correlated snapshots from the same storm and **must not be reported as unseen-storm generalization**.

When a reliable event identifier is available, include it in the dataset and run:

```bash
python -m experiments.compare_qkn_kernel \
  --data path/to/event_annotated_dataset.csv \
  --group-column storm_id --n-qubits 3
```

The group split holds out whole groups. Confirm that the group column identifies complete storms and that both classes are represented in the training partition. The current runner is a first binary-classification benchmark, not yet a full event-aware three-way train/calibration/test pipeline.

## Outputs and metrics

The JSON and CSV report accuracy, precision, recall, F1, elapsed fit-and-predict time, sample counts, split type, and kernel matrix shape. Timings include quantum Gram-matrix construction and should be interpreted as CPU statevector simulation timing in the current configuration.

## Reproducibility and constraints

- Do not fit scalers on test data.
- Use the same train/test rows and fixed random seed for both classifiers.
- Select hyperparameters using training/validation data only; do not tune on the test set.
- Current kernel uses PennyLane `default.qubit` statevector simulation, not quantum hardware.
- A successful test run establishes implementation checks, not predictive superiority or quantum advantage.


## Matched comparison of Approach A, Approach B, and classical baseline

Run the three-way comparison:

```bash
python -m experiments.compare_qkn_approaches \
  --data data/processed/final_quantum_dataset.csv \
  --output outputs/qkn_approach_comparison.json \
  --n-qubits 3 --layers 2 --C 1.0 --max-samples 250
```

The runner uses one common train/test split and the same labels for all three paths:

1. **Approach A (representation ablation):** QKN-style AngleEmbedding + StronglyEntanglingLayers, measured as Pauli-Z expectation values, followed by an RBF-SVC.
2. **Approach B:** quantum fidelity Gram matrix and precomputed-kernel SVC.
3. **Classical control:** standardized input features followed by an RBF-SVC.

All scalers are fitted on the training partition only. The runner exports JSON and CSV metrics. The expectation-value circuit is simulated with PennyLane `default.qubit` for this comparison; this is not a hardware benchmark.

**Scope caveat:** this is a matched comparison of quantum feature representations, not a complete evaluation of the full Approach A spatiotemporal GNN. The checked-in dataset is row-wise and lacks time-window sequences, node identifiers, and graph edge data needed to instantiate the full GNN without inventing structure. Do not describe the RBF-SVC downstream classifier as the full Approach A architecture. To compare full A versus B, first prepare event-aware sequence tensors and the corresponding graph topology, then evaluate the GNN and kernel model on identical event-level partitions.

The default sample-level split remains exploratory because the checked-in dataset has no storm/event ID. Do not use it to claim unseen-storm generalization.


## Hybrid QKN (A+B): fused quantum representation

The matched benchmark now includes **Hybrid QKN (A+B)** as a fourth model path. It fuses both quantum representations before the downstream classifier:

1. **A branch — expectation-value representation:** apply AngleEmbedding + StronglyEntanglingLayers and measure Pauli-Z expectations.
2. **B branch — fidelity representation:** compute each sample's quantum fidelity similarity to each training sample.
3. **Fusion:** concatenate the A expectation vector with the B similarity-to-training vector for each sample.
4. **Classifier:** standardize the fused vector using statistics fitted on training data only, then train an RBF-SVC. The test representation uses similarities to training samples only; held-out samples are never used as reference points.

The comparison now reports A, B, Hybrid A+B, and classical RBF-SVC metrics on the same split. For storm-aware evaluation, all metrics include pooled and per-storm results. Recall remains the primary metric to inspect, followed by precision and then accuracy.

**Architecture scope:** this is a concrete feature-level hybrid for the current row-wise, event-aware CSV benchmark. It combines A's expectation values and B's fidelity similarities, but it is **not yet a fusion into the full** `QuantumSpatiotemporalGNN` (temporal CNN + GAT + BiLSTM + attention). The row-wise CSV does not carry the full sequence tensor and graph structure required to train and evaluate that end-to-end hybrid. Do not describe this benchmark as the full hybrid GNN or claim predictive superiority before running and validating the experiment.

The fidelity branch produces an n_train-length similarity vector per sample, so the fused dimensionality grows with the number of training samples. The dense kernel computation is quadratic in training-set size; keep the sample budget modest for simulator-based exploration.
