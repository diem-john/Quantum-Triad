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
