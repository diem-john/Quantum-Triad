# QDF-QKN: Explicit Quantum Fidelity Kernel Learning

This branch adds an alternative to the existing expectation-value feature extractor in `src/qkn.py`; the existing implementation is unchanged.

## Method

For each input, `QuantumFidelityKernel` prepares a state using data-dependent rotations and ZZ-style pair interactions, then computes `K(x,y) = |<phi(x)|phi(y)>|^2`. The Gram matrix is used directly by scikit-learn `SVC(kernel="precomputed")`. Inference computes a test-to-training kernel matrix.

## Minimal use

```python
from src.qkn_kernel import QuantumKernelSVC
model = QuantumKernelSVC(n_qubits=4, C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score_metrics(X_test, y_test))
```

## Experiment controls
- Fit preprocessing/scaling on training data only.
- Use event-level train/validation/test partitions; keep complete test storms held out.
- Tune qubit count and SVC C on validation data only.
- Report sample count and Gram-matrix construction cost: dense training requires O(n^2) kernel evaluations.
- Compare with a classical SVC using the same split and tuning budget.
- This initial implementation uses PennyLane statevector simulation; it does not establish hardware execution or quantum advantage.

Run `pytest tests/test_qkn_kernel.py` to exercise the kernel and classifier.